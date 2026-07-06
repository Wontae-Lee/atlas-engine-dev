#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/sink/sink.h>
#include <atlas/spatial/ray.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

Sink::Sink(UniverseHostPtr universe,
           DeviceBuffer<DespawnType> despawn_types,
           DeviceBuffer<Despawn> despawn_operators,
           atlas::host_shared_ptr<atlas::Fluid> fluid,
           const bool flip,
           const float tolerance,
           ObserverHostPtr observer) noexcept
    : _universe(std::move(universe))
    , _despawn_types(std::move(despawn_types))
    , _despawn_operators(std::move(despawn_operators))
    , _fluid(std::move(fluid))
    , _observer(std::move(observer))
    , _flip(flip)
    , _tolerance(tolerance) {
}

Sink::Builder
Sink::builder() noexcept {
    return Builder {};
}

void
Sink::update(const float dt) {
    if (!_universe || _universe->sink_units().empty() || !(dt > 0.0f)) {
        return;
    }
    _universe->sink_units().advance(dt);
    sink(dt);
}

void
Sink::sink(const float dt) {
    const std::size_t step_index = _step_index++;
    auto* sink_sensor_metrics    = _observer ? _observer->sensor_metrics<atlas::SinkSensorMetrics>() : nullptr;
    HostBuffer<std::size_t> removed_per_unit;
    if (sink_sensor_metrics != nullptr) {
        removed_per_unit = HostBuffer<std::size_t>(_universe->sink_units().size(), std::size_t { 0 });
    }
    const auto record_sink_metrics = [&] {
        if (sink_sensor_metrics == nullptr) {
            return;
        }
        for (std::size_t unit_index = 0; unit_index < removed_per_unit.size(); ++unit_index) {
            sink_sensor_metrics->record(step_index, unit_index, removed_per_unit[unit_index]);
        }
    };
    if (!make_probe(dt)) {
        record_sink_metrics();
        return;
    }

    const auto probe                = _probe;
    int* despawned_unit_indices_ptr = nullptr;
    if (sink_sensor_metrics != nullptr) {
        if (_despawned_unit_indices.size() != probe.particle_count) {
            _despawned_unit_indices.resize(probe.particle_count);
        }
        despawned_unit_indices_ptr = atlas::raw_pointer_cast(_despawned_unit_indices.data());
    }

    despawn_particles(probe, despawned_unit_indices_ptr);

    if (sink_sensor_metrics != nullptr) {
        const HostBuffer<int> removed_units(
            _despawned_unit_indices.begin(),
            _despawned_unit_indices.begin() + static_cast<std::ptrdiff_t>(probe.particle_count));
        for (const int unit_index : removed_units) {
            if (unit_index >= 0 && static_cast<std::size_t>(unit_index) < removed_per_unit.size()) {
                ++removed_per_unit[static_cast<std::size_t>(unit_index)];
            }
        }
    }
    record_sink_metrics();
    compact_fluid_particles();
}

bool
Sink::make_probe(const float dt) noexcept {
    _probe = {};

    if (!_fluid || !_universe || _universe->sink_units().empty() || _despawn_operators.empty()) {
        return false;
    }

    refresh_unit_bounds();

    return detail::SinkProbeBuilder::make(_fluid,
                                          _universe->sink_units().units(),
                                          _unit_bounds,
                                          _despawn_operators,
                                          _flip,
                                          _tolerance,
                                          dt,
                                          _probe);
}

void
Sink::refresh_unit_bounds() noexcept {
    _unit_bound_cache.refresh(_universe->sink_units().units(), _unit_bounds, _tolerance);
}

void
Sink::despawn_particles(const SinkProbe& probe, int* removed_unit_indices) {
    const auto device_probe = probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        device_probe.particle_count,
        [=] ATLAS_ALL_DEVICE(const std::size_t i) {
            if (device_probe.active[i] == 0) {
                if (removed_unit_indices != nullptr) {
                    removed_unit_indices[i] = -1;
                }
                return;
            }

            const Float3& position = device_probe.positions[i];
            bool should_despawn    = false;
            int matched_unit_index = -1;

            for (int unit_index = 0; unit_index < device_probe.unit_count; ++unit_index) {
                const int despawn_operator_index
                    = (device_probe.despawn_operator_count == 1 || unit_index >= device_probe.despawn_operator_count)
                    ? 0
                    : unit_index;
                const auto& despawn_operator = device_probe.despawn_operators[despawn_operator_index];

                // Broad-phase reject against the tolerance-expanded AABB
                // before the exact despawn query; an invalid bound (e.g. an
                // unbounded plane unit) skips this cheap test and always
                // falls through to the exact query.
                const auto& unit_bound = device_probe.unit_bounds[unit_index];
                if (unit_bound.is_valid()) {
                    if (despawn_operator.type == DespawnType::tracing) {
                        if (device_probe.velocities == nullptr) {
                            continue;
                        }

                        const Float3& velocity = device_probe.velocities[i];
                        const float speed      = velocity.length();
                        if (!(device_probe.time_step > 0.0f) || !(speed > 0.0f)) {
                            continue;
                        }

                        const auto bound_hit = unit_bound.trace(atlas::Ray(position, velocity));
                        if (!bound_hit.is_intersecting || bound_hit.enter > speed * device_probe.time_step) {
                            continue;
                        }
                    } else if (!unit_bound.contains(position)) {
                        continue;
                    }
                }

                const auto& unit            = device_probe.units[unit_index];
                const auto& sync_op         = unit.sync();
                const auto& geometry_op     = unit.geometry();
                const Float3 local_position = sync_op.sync_to_local(position);
                Float3 despawn_vector       = local_position;
                float despawn_value         = device_probe.tolerance;

                if (despawn_operator.type == DespawnType::tracing) {
                    if (device_probe.velocities == nullptr) {
                        continue;
                    }
                    despawn_vector = sync_op.sync_dir_to_local(device_probe.velocities[i]);
                    despawn_value  = device_probe.time_step;
                }

                if (despawn_operator.despawn(geometry_op, local_position, despawn_vector, despawn_value)) {
                    should_despawn     = true;
                    matched_unit_index = unit_index;
                    break;
                }
            }

            // flip inverts the keep/despawn decision (see sink.h's
            // top-of-file documentation): normally a particle matching any
            // unit's despawn condition is removed, but with flip set only
            // matching particles are *kept* (e.g. using a single unit as a
            // "valid domain" mask). When flipped and nothing matched with a
            // single configured unit, that one unit is still the
            // responsible one for metrics purposes even though no explicit
            // match index was recorded.
            const bool keep_particle = device_probe.flip ? should_despawn : !should_despawn;
            int recorded_unit_index  = matched_unit_index;
            if (!keep_particle && recorded_unit_index < 0 && device_probe.flip && device_probe.unit_count == 1) {
                recorded_unit_index = 0;
            }

            device_probe.active[i] = keep_particle ? 1 : 0;

            if (removed_unit_indices != nullptr) {
                removed_unit_indices[i] = keep_particle ? -1 : recorded_unit_index;
            }
        });
}

void
Sink::compact_fluid_particles() {
    auto* active_state = _fluid->state<atlas::FluidActiveState>();
    if (active_state == nullptr) {
        return;
    }

    auto& active     = active_state->data();
    const auto count = _fluid->particle_count();
    if (count == 0) {
        return;
    }

    if (_keep.size() != count) {
        _keep.resize(count);
    }
    if (_offsets.size() != count) {
        _offsets.resize(count);
    }
    if (_compact_indices.size() != count) {
        _compact_indices.resize(count);
    }

    const auto* active_ptr = atlas::raw_pointer_cast(active.data());
    auto* keep_ptr         = atlas::raw_pointer_cast(_keep.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_ALL_DEVICE(const std::size_t i) {
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });

    // Exclusive prefix sum of keep flags: _offsets[i] becomes the new
    // (packed) index a surviving particle i lands at after compaction —
    // standard stream-compaction via scan.
    atlas::exclusive_scan<ExecutionPolicy::device>(
        _keep.begin(),
        _keep.begin() + static_cast<std::ptrdiff_t>(count),
        _offsets.begin(),
        std::size_t { 0 });

    if (_total_count_buffer.size() < 1) {
        _total_count_buffer.resize(1);
    }

    auto* total_ptr          = atlas::raw_pointer_cast(_total_count_buffer.data());
    const auto* scan_offsets = atlas::raw_pointer_cast(_offsets.data());
    const auto last          = static_cast<std::ptrdiff_t>(count - 1);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(int) {
            total_ptr[0] = scan_offsets[last] + keep_ptr[last];
        });

    std::size_t kept = 0;
    atlas::copy_device_to_host(total_ptr, &kept, 1);

    if (kept == count) {
        _fluid->set_particle_count(kept);
        return;
    }

    if (kept == 0) {
        atlas::parallel_fill<ExecutionPolicy::device>(
            active.begin(),
            active.begin() + static_cast<std::ptrdiff_t>(count),
            0);
        _fluid->set_particle_count(kept);
        return;
    }

    // _compact_indices[new_index] = old_index: scatter each surviving
    // particle's *old* index into its *new* (packed) slot. This is the
    // permutation every FluidState::compact() below gathers through, so
    // every state (position, velocity, species, ...) is reordered
    // identically and stays aligned to the same particle.
    const auto* offsets_ptr = atlas::raw_pointer_cast(_offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(_compact_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_ALL_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });

    // Iterate every *registered* state generically (Fluid doesn't know in
    // advance which states a caller attached) so compaction stays correct
    // regardless of which optional states (internal energy, temperature,
    // ...) this particular fluid happens to track.
    for (auto& entry : _fluid->states()) {
        auto& state = entry.second;
        if (state) {
            state->compact(_compact_indices, kept);
        }
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        active.begin() + static_cast<std::ptrdiff_t>(kept),
        active.begin() + static_cast<std::ptrdiff_t>(_fluid->buffer_size()),
        0);
    _fluid->set_particle_count(kept);
}

Sink
Sink::Builder::build() {
    validate();
    auto despawn_types     = _despawn_types;
    auto despawn_operators = _despawn_operators;
    return Sink(
        _universe,
        DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end()),
        DeviceBuffer<Despawn>(despawn_operators.begin(), despawn_operators.end()),
        _fluid,
        _flip,
        _tolerance,
        _observer);
}

atlas::host_shared_ptr<Sink>
Sink::Builder::make_host_shared() {
    return atlas::make_host_shared<Sink>(build());
}

Sink::Builder&
Sink::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

Sink::Builder&
Sink::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

Sink::Builder&
Sink::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

Sink::Builder&
Sink::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    if (despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }
    _despawn_types.insert(_despawn_types.end(), despawn_types.begin(), despawn_types.end());
    return *this;
}

Sink::Builder&
Sink::Builder::with_despawn_operator(const Despawn& despawn_operator) noexcept {
    _despawn_operators.push_back(despawn_operator);
    return *this;
}

Sink::Builder&
Sink::Builder::with_despawn_operators(const HostBuffer<Despawn>& despawn_operators) {
    if (despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }
    _despawn_operators.insert(
        _despawn_operators.end(),
        despawn_operators.begin(),
        despawn_operators.end());
    return *this;
}

Sink::Builder&
Sink::Builder::with_tolerance(const float tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

Sink::Builder&
Sink::Builder::with_flip(const bool flip) noexcept {
    _flip = flip;
    return *this;
}

void
Sink::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Sink::Builder: fluid must not be null.");
    }
    if (!_universe) {
        throw std::runtime_error("Sink::Builder: universe must not be null.");
    }
    const std::size_t unit_count = _universe->sink_units().size();
    if (unit_count == 0) {
        throw std::runtime_error("Sink::Builder: universe must hold at least one sink unit.");
    }
    if (_despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }
    if (_despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }
    if (_despawn_operators.size() != 1
        && _despawn_operators.size() != unit_count) {
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }
    if (_despawn_types.size() != 1
        && _despawn_types.size() != unit_count) {
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }
    if (!atlas::isfinite(_tolerance)) {
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

}
