#pragma once
#include <atlas/iterator/zip_iterator.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/tuple/tuple.h>
#include <ranges>
#include <stdexcept>
#include <utility>
namespace atlas::fluid {
template <typename T>
Sink<T>::Sink(DeviceBuffer<Unit<T>> units,
              DeviceBuffer<DespawnType> despawn_types,
              DeviceBuffer<DespawnOperator<T>> despawn_operators,
              atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
              const bool flip,
              const T tolerance,
              ObserverHostPtr observer) noexcept
    : _units(std::move(units))
    , _despawn_types(std::move(despawn_types))
    , _despawn_operators(std::move(despawn_operators))
    , _fluid(std::move(fluid))
    , _observer(std::move(observer))
    , _flip(flip)
    , _tolerance(tolerance) {
}

template <typename T>
typename Sink<T>::Builder
Sink<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Sink<T>::update(const T dt) {
    if (_units.empty() || !(dt > T(0))) {
        return;
    }
    auto* units = atlas::raw_pointer_cast(_units.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
    sink(dt);
}

template <typename T>
void
Sink<T>::sink(const T dt) {
    const std::size_t step_index = _step_index++;
    auto* sink_sensor_matrics    = _observer ? _observer->sensor_matrics<atlas::SinkSensorMatrics>() : nullptr;
    HostBuffer<std::size_t> removed_per_unit;
    if (sink_sensor_matrics != nullptr) {
        removed_per_unit = HostBuffer<std::size_t>(_units.size(), std::size_t { 0 });
    }
    const auto record_sink_metrics = [&] {
        if (sink_sensor_matrics == nullptr) {
            return;
        }
        for (std::size_t unit_index = 0; unit_index < removed_per_unit.size(); ++unit_index) {
            sink_sensor_matrics->record(step_index, unit_index, removed_per_unit[unit_index]);
        }
    };
    if (!make_probe(dt)) {
        record_sink_metrics();
        return;
    }
    const auto probe = _probe;
    int* despawned_unit_indices_ptr = nullptr;
    if (sink_sensor_matrics != nullptr) {
        if (_despawned_unit_indices.size() != probe.particle_count) {
            _despawned_unit_indices.resize(probe.particle_count);
        }
        despawned_unit_indices_ptr = atlas::raw_pointer_cast(_despawned_unit_indices.data());
    }
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        probe.particle_count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (probe.active[i] == 0) {
                if (despawned_unit_indices_ptr != nullptr) {
                    despawned_unit_indices_ptr[i] = -1;
                }
                return;
            }
            const Vector3<T>& p    = probe.positions[i];
            bool should_despawn    = false;
            int matched_unit_index = -1;
            for (int unit_index = 0; unit_index < probe.unit_count; ++unit_index) {
                const auto& unit        = probe.units[unit_index];
                const auto& sync_op     = unit.sync_operator();
                const auto& geometry_op = unit.geometry_operator();
                const int despawn_operator_index
                    = (probe.despawn_operator_count == 1 || unit_index >= probe.despawn_operator_count) ? 0 : unit_index;
                const Vector3<T> local_p = sync_op.sync_to_local(p);
                const auto& despawn_operator = probe.despawn_operators[despawn_operator_index];
                Vector3<T> despawn_vector    = local_p;
                T despawn_value              = probe.tolerance;
                if (despawn_operator.type == DespawnType::Tracing) {
                    if (probe.velocities == nullptr) {
                        continue;
                    }
                    despawn_vector = sync_op.sync_dir_to_local(probe.velocities[i]);
                    despawn_value  = probe.time_step;
                }
                if (despawn_operator.despawn(geometry_op, local_p, despawn_vector, despawn_value)) {
                    should_despawn     = true;
                    matched_unit_index = unit_index;
                    break;
                }
            }
            const bool keep_particle = probe.flip ? should_despawn : !should_despawn;
            int recorded_unit_index  = matched_unit_index;
            if (!keep_particle && recorded_unit_index < 0 && probe.flip && probe.unit_count == 1) {
                recorded_unit_index = 0;
            }
            probe.active[i] = keep_particle ? 1 : 0;
            if (despawned_unit_indices_ptr != nullptr) {
                despawned_unit_indices_ptr[i] = keep_particle ? -1 : recorded_unit_index;
            }
        });
    if (sink_sensor_matrics != nullptr) {
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

template <typename T>
bool
Sink<T>::make_probe(const T dt) noexcept {
    _probe = {};
    if (!_fluid || _units.empty() || _despawn_operators.empty()) {
        return false;
    }

    auto& positions = _fluid->template state<atlas::fluid::FluidPositionState<T>>()->data();
    auto& active    = _fluid->template state<atlas::fluid::FluidActiveState<T>>()->data();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    if (positions.empty() || active.empty() || _fluid->particle_count() == 0) {
        return false;
    }

    _probe.units                  = atlas::raw_pointer_cast(_units.data());
    _probe.despawn_operators      = atlas::raw_pointer_cast(_despawn_operators.data());
    _probe.positions              = atlas::raw_pointer_cast(positions.data());
    if (velocity_state != nullptr && !velocity_state->data().empty()) {
        _probe.velocities = atlas::raw_pointer_cast(velocity_state->data().data());
    }
    _probe.active                 = atlas::raw_pointer_cast(active.data());
    _probe.unit_count             = static_cast<int>(_units.size());
    _probe.despawn_operator_count = static_cast<int>(_despawn_operators.size());
    _probe.particle_count         = _fluid->particle_count();
    _probe.flip                   = _flip;
    _probe.tolerance              = _tolerance;
    _probe.time_step              = dt;
    return true;
}

template <typename T>
void
Sink<T>::compact_fluid_particles() {
    auto* active_state = _fluid->template state<atlas::fluid::FluidActiveState<T>>();
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
        [=] ATLAS_DEVICE(const std::size_t i) {
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });
    atlas::exclusive_scan<ExecutionPolicy::device>(
        _keep.begin(),
        _keep.begin() + static_cast<std::ptrdiff_t>(count),
        _offsets.begin(),
        std::size_t { 0 });
    const std::size_t kept = _offsets[count - 1] + _keep[count - 1];
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
    const auto* offsets_ptr = atlas::raw_pointer_cast(_offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(_compact_indices.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });
    for (auto& state : _fluid->states() | std::views::values) {
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

template <typename T>
Sink<T>
Sink<T>::Builder::build() {
    validate();
    auto despawn_types     = _despawn_types;
    auto despawn_operators = _despawn_operators;
    return Sink<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end()),
        DeviceBuffer<DespawnOperator<T>>(despawn_operators.begin(), despawn_operators.end()),
        _fluid,
        _flip,
        _tolerance,
        _observer);
}

template <typename T>
atlas::host_shared_ptr<Sink<T>>
Sink<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Sink<T>>(build());
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        throw std::runtime_error("Sink::Builder: units must not be empty.");
    }
    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    if (despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }
    _despawn_types.insert(_despawn_types.end(), despawn_types.begin(), despawn_types.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {
    _despawn_operators.push_back(despawn_operator);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {
    if (despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }
    _despawn_operators.insert(
        _despawn_operators.end(),
        despawn_operators.begin(),
        despawn_operators.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_flip(const bool flip) noexcept {
    _flip = flip;
    return *this;
}

template <typename T>
void
Sink<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Sink::Builder: fluid must not be null.");
    }
    if (_units.empty()) {
        throw std::runtime_error("Sink::Builder: at least one unit must be provided.");
    }
    if (_despawn_types.empty()) {
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }
    if (_despawn_operators.empty()) {
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }
    if (_despawn_operators.size() != 1
        && _despawn_operators.size() != _units.size()) {
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }
    if (_despawn_types.size() != 1
        && _despawn_types.size() != _units.size()) {
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }
    if (!std::isfinite(_tolerance)) {
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

}
