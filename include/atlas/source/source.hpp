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
}

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
void
Source<T>::update(const T dt) {

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

    emit();
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {

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

    const std::size_t step_index = _step_index++;

    auto* source_sensor_metrics = _observer ? _observer->sensor_metrics<atlas::SourceSensorMetrics>() : nullptr;

    HostBuffer<std::size_t> emitted_per_unit;

    if (source_sensor_metrics != nullptr) {

        emitted_per_unit = HostBuffer<std::size_t>(_units.size(), std::size_t { 0 });
    }

    const auto record_source_metrics = [&] {
        if (source_sensor_metrics == nullptr) {
            return;
        }

        for (std::size_t unit_index = 0; unit_index < emitted_per_unit.size(); ++unit_index) {
            source_sensor_metrics->record(step_index, unit_index, emitted_per_unit[unit_index]);
        }
    };

    if (!_fluid) {

        record_source_metrics();
        return;
    }

    rebuild_cache();

    if (_local_particle_count == 0) {

        record_source_metrics();
        return;
    }

    const std::size_t current_particle_count = _fluid->particle_count();

    const std::size_t available_slots = current_particle_count < _fluid->buffer_size()
        ? (_fluid->buffer_size() - current_particle_count)
        : std::size_t { 0 };

    const std::size_t emit_count = _local_particle_count < available_slots
        ? _local_particle_count
        : available_slots;

    if (emit_count == 0) {

        atlas::warn() << "\n"
                      << "Source emission skipped: no available slots for "
                      << _local_particle_count
                      << " particles\n";

        record_source_metrics();
        return;
    }

    shuffle_species(emit_count);

    if (!make_probe()) {
        record_source_metrics();
        return;
    }

    const auto probe = _probe;

    if (source_sensor_metrics != nullptr) {
        int offset = 0;
        for (std::size_t u = 0; u < _local_unit_counts.size(); ++u) {
            const int unit_size     = _local_unit_counts[u];
            const std::size_t start = static_cast<std::size_t>(offset);
            const std::size_t end   = start + static_cast<std::size_t>(unit_size);
            if (emit_count > start) {
                emitted_per_unit[u] = std::min(emit_count, end) - start;
            }
            offset += unit_size;
        }
    }

    _emitter.emit(probe, current_particle_count, emit_count);

    _fluid->set_particle_count(current_particle_count + emit_count);

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

    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {

    if (units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    _units.insert(_units.end(), units.begin(), units.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {

    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {

    if (spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    _spawn_types.insert(_spawn_types.end(), spawn_types.begin(), spawn_types.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept {

    _spawn_operators.push_back(spawn_operator);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {

    if (spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    _spawn_operators.insert(_spawn_operators.end(), spawn_operators.begin(), spawn_operators.end());

    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {

    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_flip(const bool flip) noexcept {

    _flip = flip;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {

    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_temperature(const T temperature) noexcept {

    _temperature = temperature;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {

    if (_units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    if (!_fluid) {
        throw std::runtime_error("Source::Builder: fluid must be provided.");
    }

    if (_spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    if (_spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    if (_spawn_types.size() != 1 && _spawn_types.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    if (_spawn_operators.size() != 1 && _spawn_operators.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
    }

    if (!atlas::isfinite(_spacing) || _spacing <= T(0)) {
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    if (!atlas::isfinite(_tolerance)) {
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!atlas::isfinite(_temperature)) {
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

}