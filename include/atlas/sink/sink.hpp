#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
Sink<T>::Sink(DeviceBuffer<Unit<T>> units,
              DeviceBuffer<DespawnType> despawn_types,
              DeviceBuffer<DespawnOperator<T>> despawn_operators,
              atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
              const bool flip,
              const T tolerance,
              ObserverHostPtr observer) noexcept
    : _units(std::move(units))
    , _unit_bounds(_units.size())
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

    _particle_despawner.apply(probe, despawned_unit_indices_ptr);

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

    refresh_unit_bounds();

    return _probe_builder.build(_fluid,
                                _units,
                                _unit_bounds,
                                _despawn_operators,
                                _flip,
                                _tolerance,
                                dt,
                                _probe);
} // namespace atlas

template <typename T>
void
Sink<T>::refresh_unit_bounds() noexcept {
    _unit_bound_cache.refresh(_units, _unit_bounds, _tolerance);
}

template <typename T>
void
Sink<T>::compact_fluid_particles() {
    _fluid_compactor.compact(_fluid, _keep, _offsets, _compact_indices, _total_count_buffer);
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
    if (!atlas::isfinite(_tolerance)) {
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

}
