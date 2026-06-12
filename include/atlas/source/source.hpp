#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

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

    _cache_builder.rebuild(_units,
                           _spawn_types,
                           _spawn_operators,
                           _fluid,
                           _flip,
                           _spacing,
                           _tolerance,
                           _local_unit_counts,
                           _flat_local_positions,
                           _flat_unit_indices,
                           _local_particle_count,
                           _species_cache,
                           _shuffled_species,
                           _shuffle_keys,
                           _shuffle_seed);

    _is_invalidated_cache = false;
}

template <typename T>
void
Source<T>::shuffle_species(const std::size_t count) {
    _species_shuffler.shuffle(_species_cache, _shuffled_species, _shuffle_keys, _shuffle_seed, count);
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
        atlas::warn() << "\n"
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

    _emitter.emit(probe, current_particle_count, emit_count);

    // Publish the new particle count after all emitted particle data has been written.
    _fluid->set_particle_count(current_particle_count + emit_count);

    // Persist observer metrics for this emission step.
    record_source_metrics();
}

template <typename T>
bool
Source<T>::make_probe() noexcept {
    return _probe_builder.build(_fluid,
                                _units,
                                _shuffled_species,
                                _flat_local_positions,
                                _flat_unit_indices,
                                _temperature,
                                _shuffle_seed,
                                _probe);
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
    if (!atlas::isfinite(_spacing) || _spacing <= T(0)) {
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    // Geometric tolerance must be finite.
    if (!atlas::isfinite(_tolerance)) {
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    // Emission temperature must be finite.
    if (!atlas::isfinite(_temperature)) {
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

} // namespace atlas
