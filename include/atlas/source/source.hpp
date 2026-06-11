#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/sampling/sampling.h>
#include <atlas/shuffle/shuffle_operator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas::fluid {

template <typename T>
Source<T>::Source(DeviceBuffer<Unit<T>> units,
                  DeviceBuffer<SpawnType> spawn_types,
                  DeviceBuffer<SpawnOperator<T>> spawn_operators,
                  atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
                  const bool flip,
                  const T spacing,
                  const T tolerance,
                  const T temperature,
                  ObserverHostPtr observer) noexcept
    : _units(std::move(units))
    , _spawn_types(std::move(spawn_types))
    , _spawn_operators(std::move(spawn_operators))
    , _fluid(std::move(fluid))
    , _observer(std::move(observer))
    , _flip(flip)
    , _spacing(spacing)
    , _tolerance(tolerance)
    , _temperature(temperature)
    , _is_invalidated_cache(true) {
    // Store source configuration and mark cached spawn samples as dirty.
}

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {
    // Return a fresh builder for fluent Source construction.
    return Builder {};
}

template <typename T>
void
Source<T>::update(const T dt) {
    // Skip source update when there are no units or the time step is invalid.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    // Bind device-accessible unit storage for the update kernel.
    auto* units = atlas::raw_pointer_cast(_units.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            // Advance each source unit by the current time step.
            units[i].update(dt);
        });

    // Emit particles after all source units have been advanced.
    emit();
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {
    // Reuse existing cached spawn samples while the cache remains valid.
    if (!_is_invalidated_cache) {
        return;
    }

    if (_units.empty() || _spawn_types.empty() || _spawn_operators.empty() || !_fluid || _fluid->generators().empty()) {
        // Clear all cached data when required source inputs are unavailable.
        _local_unit_counts.clear();
        _flat_local_positions.clear();
        _flat_unit_indices.clear();
        _local_particle_count = 0;
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const auto& generators = _fluid->generators();

    // Copy CUDA device buffers once for deterministic host-side cache construction.
#if defined(ATLAS_TASKING_CUDA)
    const HostBuffer<Unit<T>> units(_units.begin(), _units.end());

    const HostBuffer<SpawnOperator<T>> spawn_operators(
        _spawn_operators.begin(),
        _spawn_operators.end());
#else
    const auto& units           = _units;
    const auto& spawn_operators = _spawn_operators;
#endif

    // Store per-unit counts for observer metrics while building one flat cache.
    _local_unit_counts.clear();
    _local_unit_counts.resize(units.size(), 0);

    HostBuffer<Vector3<T>> flat_positions;
    HostBuffer<int> flat_unit_indices;

    const std::size_t spawn_operator_count = spawn_operators.size();

    for (std::size_t i = 0; i < units.size(); ++i) {
        // Sample the bounding box of the unit geometry.
        const auto& geometry = units[i].geometry_operator();
        const auto bounds    = geometry.bound();
        const auto& lower    = bounds.lower_corner;
        const auto& upper    = bounds.upper_corner;

        // Use either the shared spawn operator or the unit-specific operator.
        const auto& spawn_op = spawn_operators[(spawn_operator_count == 1) ? 0 : i];

        // Determine the number of sample points along each axis.
        const int nx = atlas::sampling::sample_axis_count(lower.x, upper.x, _spacing);
        const int ny = atlas::sampling::sample_axis_count(lower.y, upper.y, _spacing);
        const int nz = atlas::sampling::sample_axis_count(lower.z, upper.z, _spacing);

        std::size_t accepted_count = 0;

        if (nx > 0 && ny > 0 && nz > 0) {
            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        // Generate a regular-grid sample in the unit's local coordinate space.
                        const Vector3<T> sample(
                            lower.x + static_cast<T>(ix) * _spacing,
                            lower.y + static_cast<T>(iy) * _spacing,
                            lower.z + static_cast<T>(iz) * _spacing);

                        // Test whether the sample should be accepted by the spawn operator.
                        const bool accepted = spawn_op.spawn(geometry, sample, _tolerance);

                        // Flip mode inverts the accepted region.
                        if (_flip ? !accepted : accepted) {
                            flat_positions.push_back(sample);
                            flat_unit_indices.push_back(static_cast<int>(i));
                            ++accepted_count;
                        }
                    }
                }
            }
        }

        _local_unit_counts[i] = static_cast<int>(accepted_count);
    }

    // Reset species shuffle buffers before rebuilding them.
    _species_cache.clear();
    _shuffled_species.clear();
    _shuffle_keys.clear();

    const std::size_t total_count = flat_positions.size();
    _local_particle_count         = total_count;

    if (total_count == 0) {
        // No cached particles are available for emission.
        _flat_local_positions.clear();
        _flat_unit_indices.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    // Move the host-built flat cache to device storage with one transfer per buffer.
#if defined(ATLAS_TASKING_CUDA)
    _flat_local_positions = DeviceBuffer<Vector3<T>>(flat_positions.begin(), flat_positions.end());
    _flat_unit_indices    = DeviceBuffer<int>(flat_unit_indices.begin(), flat_unit_indices.end());
#else
    _flat_local_positions = std::move(flat_positions);
    _flat_unit_indices    = std::move(flat_unit_indices);
#endif

    // Allocate species assignment and shuffle buffers for all cached samples.
    _species_cache.resize(total_count);
    _shuffled_species.resize(total_count);
    _shuffle_keys.resize(total_count);

    const int count         = static_cast<int>(total_count);
    const int species_count = static_cast<int>(generators.size());
    auto* cache             = atlas::raw_pointer_cast(this->_species_cache.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_DEVICE(const int i) {
            // Assign species in a repeating pattern before shuffling.
            cache[i] = static_cast<std::size_t>(i % species_count);
        });

    // Reset shuffle sequence after rebuilding source samples.
    _shuffle_seed         = 0;
    _is_invalidated_cache = false;
}

template <typename T>
void
Source<T>::shuffle_species(const std::size_t count) {
    if (count == 0) {
        // Clear shuffle output when there are no particles to emit.
        _shuffled_species.clear();
        _shuffle_keys.clear();
        return;
    }

    // Advance the seed so each emission step gets a different shuffled order.
    const std::uint64_t seed = _shuffle_seed++;

    const auto* cache = atlas::raw_pointer_cast(this->_species_cache.data());
    auto* shuffled = atlas::raw_pointer_cast(this->_shuffled_species.data());
    auto* keys     = atlas::raw_pointer_cast(this->_shuffle_keys.data());

    const ShuffleOperator shuffle {};

    atlas::parallel_for<ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            // Copy and key only the species range that will be emitted this step.
            shuffled[i] = cache[i];
            keys[i]     = shuffle(static_cast<int>(i), seed);
        });

    // Sort species by pseudo-random keys to distribute species assignments.
    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.begin() + static_cast<std::ptrdiff_t>(count),
        _shuffled_species.begin());
}

template <typename T>
void
Source<T>::emit() {
    // Use a monotonically increasing step index for observer metrics.
    const std::size_t step_index = _step_index++;

    // Retrieve source metrics storage when an observer is attached.
    auto* source_sensor_matrics = _observer ? _observer->sensor_matrics<atlas::SourceSensorMatrics>() : nullptr;

    HostBuffer<std::size_t> emitted_per_unit;

    if (source_sensor_matrics != nullptr) {
        // Allocate per-unit emission counters only when metrics are enabled.
        emitted_per_unit = HostBuffer<std::size_t>(_units.size(), std::size_t { 0 });
    }

    const auto record_source_metrics = [&] {
        if (source_sensor_matrics == nullptr) {
            return;
        }

        // Record emitted particle count for every source unit.
        for (std::size_t unit_index = 0; unit_index < emitted_per_unit.size(); ++unit_index) {
            source_sensor_matrics->record(step_index, unit_index, emitted_per_unit[unit_index]);
        }
    };

    if (!_fluid) {
        // Without a fluid target, no particles can be emitted.
        record_source_metrics();
        return;
    }

    // Ensure cached spawn samples and species buffers are ready.
    rebuild_cache();

    if (_local_particle_count == 0) {
        // Nothing can be emitted when the cached source region contains no samples.
        record_source_metrics();
        return;
    }

    const std::size_t current_particle_count = _fluid->particle_count();

    // Compute remaining capacity in the fluid particle buffers.
    const std::size_t available_slots = current_particle_count < _fluid->buffer_size()
        ? (_fluid->buffer_size() - current_particle_count)
        : std::size_t { 0 };

    // Clamp emission count by both source sample count and available fluid capacity.
    const std::size_t emit_count = _local_particle_count < available_slots
        ? _local_particle_count
        : available_slots;

    if (emit_count == 0) {
        // Warn when the source has particles to emit but the fluid buffer is full.
        atlas::logger::warn() << "\n"
                              << "Source emission skipped: no available slots for "
                              << _local_particle_count
                              << " particles\n";

        record_source_metrics();
        return;
    }

    // Shuffle species assignments for the particles emitted in this step.
    shuffle_species(emit_count);

    // Build the device-side probe required by the emission kernel.
    if (!make_probe()) {
        record_source_metrics();
        return;
    }

    // Copy the probe descriptor for device lambda capture.
    const auto probe = _probe;

    const ShuffleOperator shuffle {};

    // Compute per-unit emission counts on the host for observer metrics using
    // the host-side knowledge of each unit's position range.
    if (source_sensor_matrics != nullptr) {
        int offset = 0;
        for (std::size_t u = 0; u < _local_unit_counts.size(); ++u) {
            const int unit_size = _local_unit_counts[u];
            const std::size_t start = static_cast<std::size_t>(offset);
            const std::size_t end   = start + static_cast<std::size_t>(unit_size);
            if (emit_count > start) {
                emitted_per_unit[u] = std::min(emit_count, end) - start;
            }
            offset += unit_size;
        }
    }

    const int dst_offset = static_cast<int>(current_particle_count);

    // Single flat kernel launch over all emit_count particles. Each thread reads
    // its owning unit directly from the cache, avoiding per-particle offset searches.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(emit_count),
        [=] ATLAS_DEVICE(const int i) {
            const int unit_index = probe.flat_unit_indices[i];
            const int dst    = dst_offset + i;
            const size_t sid = probe.shuffled_species[i];

            if (sid >= static_cast<std::size_t>(probe.property_count)) {
                return;
            }

            const auto sample_seed = static_cast<unsigned int>(shuffle(dst, probe.emission_seed));

            Vector3<T> world_pos;
            probe.units[unit_index].sync_operator().sync_to_world(probe.flat_local_positions[i], world_pos);

            probe.positions[dst]  = world_pos;
            probe.velocities[dst] = probe.generators[sid].generate(
                sample_seed,
                probe.temperature,
                probe.properties[sid].molecular_mass);
            probe.species[dst]    = sid;
            probe.active[dst]     = 1;
        });

    // Publish the new particle count after all emitted particle data has been written.
    _fluid->set_particle_count(current_particle_count + emit_count);

    // Persist observer metrics for this emission step.
    record_source_metrics();
}

template <typename T>
bool
Source<T>::make_probe() noexcept {
    _probe = {};

    if (!_fluid || _units.empty() || _shuffled_species.empty()) {
        return false;
    }

    auto& positions_buf   = _fluid->template state<FluidPositionState<T>>()->data();
    auto& velocities_buf  = _fluid->template state<FluidVelocityState<T>>()->data();
    auto& species_buf     = _fluid->template state<FluidSpeciesState<T>>()->data();
    auto& active_buf      = _fluid->template state<FluidActiveState<T>>()->data();
    const auto& generators_buf = _fluid->generators();
    const auto& properties_buf = _fluid->particle_properties();

    if (positions_buf.empty() || velocities_buf.empty() || species_buf.empty() || active_buf.empty()
        || generators_buf.empty() || properties_buf.empty()
        || _flat_local_positions.empty() || _flat_unit_indices.empty()) {
        return false;
    }

    _probe.units                  = atlas::raw_pointer_cast(_units.data());
    _probe.generators             = atlas::raw_pointer_cast(generators_buf.data());
    _probe.properties             = atlas::raw_pointer_cast(properties_buf.data());
    _probe.shuffled_species       = atlas::raw_pointer_cast(_shuffled_species.data());
    _probe.positions              = atlas::raw_pointer_cast(positions_buf.data());
    _probe.velocities             = atlas::raw_pointer_cast(velocities_buf.data());
    _probe.species                = atlas::raw_pointer_cast(species_buf.data());
    _probe.active                 = atlas::raw_pointer_cast(active_buf.data());
    _probe.flat_local_positions   = atlas::raw_pointer_cast(_flat_local_positions.data());
    _probe.flat_unit_indices      = atlas::raw_pointer_cast(_flat_unit_indices.data());
    _probe.temperature            = _temperature;
    _probe.property_count         = static_cast<int>(properties_buf.size());
    _probe.emission_seed          = _shuffle_seed;

    return true;
}

template <typename T>
Source<T>
Source<T>::Builder::build() {
    // Validate all source dependencies and configuration before construction.
    validate();

    return Source<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<SpawnType>(_spawn_types.begin(), _spawn_types.end()),
        DeviceBuffer<SpawnOperator<T>>(_spawn_operators.begin(), _spawn_operators.end()),
        _fluid,
        _flip,
        _spacing,
        _tolerance,
        _temperature,
        _observer);
}

template <typename T>
atlas::host_shared_ptr<Source<T>>
Source<T>::Builder::make_host_shared() {
    // Build a validated source and store it in host-managed shared ownership.
    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    // Source construction requires at least one unit.
    if (units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    // Append provided source units to the builder configuration.
    _units.insert(_units.end(), units.begin(), units.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {
    // Store the target fluid receiving emitted particles.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {
    // Store optional observer used for source emission metrics.
    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {
    // At least one spawn type must be provided.
    if (spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    // Append provided spawn types to the builder configuration.
    _spawn_types.insert(_spawn_types.end(), spawn_types.begin(), spawn_types.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept {
    // Append a single spawn operator to the builder configuration.
    _spawn_operators.push_back(spawn_operator);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {
    // At least one spawn operator must be provided.
    if (spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    // Append provided spawn operators to the builder configuration.
    _spawn_operators.insert(_spawn_operators.end(), spawn_operators.begin(), spawn_operators.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {
    // Store the geometric tolerance used by spawn operators.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_flip(const bool flip) noexcept {
    // Store whether spawn acceptance should be inverted.
    _flip = flip;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {
    // Store grid spacing used to sample source geometry bounds.
    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_temperature(const T temperature) noexcept {
    // Store the emission temperature used by velocity generators.
    _temperature = temperature;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {
    // At least one source unit is required.
    if (_units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    // A target fluid is required to receive emitted particles.
    if (!_fluid) {
        throw std::runtime_error("Source::Builder: fluid must be provided.");
    }

    // Spawn types must be configured before construction.
    if (_spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    // Spawn operators must be configured before construction.
    if (_spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    // Spawn types can be shared globally or specified per source unit.
    if (_spawn_types.size() != 1 && _spawn_types.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    // Spawn operators can be shared globally or specified per source unit.
    if (_spawn_operators.size() != 1 && _spawn_operators.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
    }

    // Source sampling spacing must be finite and strictly positive.
    if (!atlas::math::isfinite(_spacing) || _spacing <= T(0)) {
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    // Geometric tolerance must be finite.
    if (!atlas::math::isfinite(_tolerance)) {
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    // Emission temperature must be finite.
    if (!atlas::math::isfinite(_temperature)) {
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

} // namespace atlas::fluid
