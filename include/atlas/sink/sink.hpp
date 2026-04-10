#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/tuple/tuple.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Sink<T>::Sink(DeviceBuffer<Unit<T>> units,
              DeviceBuffer<DespawnType> despawn_types,
              DeviceBuffer<DespawnOperator<T>> despawn_operators,
              const bool flip,
              const T tolerance) noexcept
    : _units(std::move(units))
    , _despawn_types(std::move(despawn_types))
    , _despawn_operators(std::move(despawn_operators))
    , _flip(flip)
    , _tolerance(tolerance) { }

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
}

template <typename T>
void
Sink<T>::sink(FluidDeviceProbe<T>& particle_probe) {
    if (particle_probe.empty() || empty()) {
        return;
    }

    const auto* units               = atlas::raw_pointer_cast(_units.data());
    const auto* despawn_operators   = atlas::raw_pointer_cast(_despawn_operators.data());
    const int unit_count            = static_cast<int>(_units.size());
    const int despawn_operator_count = static_cast<int>(_despawn_operators.size());
    const bool flip                 = _flip;
    const T tol                     = _tolerance;

    auto pos       = particle_probe.pos;
    auto vel       = particle_probe.vel;
    auto temperature = particle_probe.temperature;
    auto species   = particle_probe.species;
    auto zip_begin = atlas::make_zip_iterator(
        atlas::make_tuple(pos, vel, temperature, species));
    auto zip_end = zip_begin + particle_probe.particle_count;

    auto new_end = atlas::remove_if(
        atlas::device,
        zip_begin,
        zip_end,
        [=] ATLAS_DEVICE(const atlas::tuple<Vector3<T>, Vector3<T>, T, size_t>& t) {
            const Vector3<T>& p       = atlas::get<0>(t);
            bool should_despawn = false;

            for (int i = 0; i < unit_count; ++i) {
                const auto& unit = units[i];
                const auto& sync_op = unit.sync_operator();
                const auto& geometry_op = unit.geometry_operator();
                const int despawn_operator_index
                    = (despawn_operator_count == 1 || i >= despawn_operator_count) ? 0 : i;
                const Vector3<T> local_p = sync_op.sync_to_local(p);

                if (despawn_operators[despawn_operator_index].despawn(geometry_op, local_p, tol)) {
                    should_despawn = true;
                    break;
                }
            }

            return flip ? !should_despawn : should_despawn;
        });

    particle_probe.particle_count = static_cast<int>(new_end - zip_begin);
}

template <typename T>
void
Sink<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    _units = std::move(units);
}

template <typename T>
void
Sink<T>::set_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        atlas::logger::error()
            << "Sink: units must not be empty.";
        throw std::runtime_error("Sink: units must not be empty.");
    }

    _units = DeviceBuffer<Unit<T>>(units.begin(), units.end());
}

template <typename T>
void
Sink<T>::set_despawn_operators(DeviceBuffer<DespawnOperator<T>> despawn_operators) noexcept {
    _despawn_operators = std::move(despawn_operators);
}

template <typename T>
void
Sink<T>::set_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators) {
    if (despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink: despawn operators must not be empty.";
        throw std::runtime_error("Sink: despawn operators must not be empty.");
    }

    _despawn_operators = DeviceBuffer<DespawnOperator<T>>(
        despawn_operators.begin(),
        despawn_operators.end());
}

template <typename T>
void
Sink<T>::set_despawn_types(DeviceBuffer<DespawnType> despawn_types) noexcept {
    _despawn_types = std::move(despawn_types);
}

template <typename T>
void
Sink<T>::set_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    if (despawn_types.empty()) {
        atlas::logger::error()
            << "Sink: despawn types must not be empty.";
        throw std::runtime_error("Sink: despawn types must not be empty.");
    }

    _despawn_types = DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end());
}

template <typename T>
void
Sink<T>::set_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept {
    set_despawn_operators(DeviceBuffer<DespawnOperator<T>>(1, despawn_operator));
}

template <typename T>
void
Sink<T>::set_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
}

template <typename T>
void
Sink<T>::set_flip(const bool flip) noexcept {
    _flip = flip;
}

template <typename T>
DeviceBuffer<Unit<T>>&
Sink<T>::units() noexcept {
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Sink<T>::units() const noexcept {
    return _units;
}

template <typename T>
DeviceBuffer<DespawnOperator<T>>&
Sink<T>::despawn_operators() noexcept {
    return _despawn_operators;
}

template <typename T>
const DeviceBuffer<DespawnOperator<T>>&
Sink<T>::despawn_operators() const noexcept {
    return _despawn_operators;
}

template <typename T>
DeviceBuffer<DespawnType>&
Sink<T>::despawn_types() noexcept {
    return _despawn_types;
}

template <typename T>
const DeviceBuffer<DespawnType>&
Sink<T>::despawn_types() const noexcept {
    return _despawn_types;
}

template <typename T>
T
Sink<T>::tolerance() const noexcept {
    return _tolerance;
}

template <typename T>
bool
Sink<T>::flip() const noexcept {
    return _flip;
}

template <typename T>
bool
Sink<T>::empty() const noexcept {
    return _units.empty() || _despawn_types.empty() || _despawn_operators.empty();
}

template <typename T>
Sink<T>
Sink<T>::Builder::build() {
    validate();

    auto despawn_types = _despawn_types;
    auto despawn_operators = _despawn_operators;

    return Sink<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<DespawnType>(despawn_types.begin(), despawn_types.end()),
        DeviceBuffer<DespawnOperator<T>>(despawn_operators.begin(), despawn_operators.end()),
        _flip,
        _tolerance);
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
        atlas::logger::error()
            << "Sink::Builder: units must not be empty.";
        throw std::runtime_error("Sink::Builder: units must not be empty.");
    }

    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_types(const HostBuffer<DespawnType>& despawn_types) {
    if (despawn_types.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must not be empty.";
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
        atlas::logger::error()
            << "Sink::Builder: despawn operators must not be empty.";
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
    if (_units.empty()) {
        atlas::logger::error()
            << "Sink::Builder: at least one unit must be provided.";
        throw std::runtime_error("Sink::Builder: at least one unit must be provided.");
    }

    if (_despawn_types.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn types must not be empty.");
    }

    if (_despawn_operators.empty()) {
        atlas::logger::error()
            << "Sink::Builder: despawn operators must not be empty.";
        throw std::runtime_error("Sink::Builder: despawn operators must not be empty.");
    }

    if (!_despawn_operators.empty()
        && _despawn_operators.size() != 1
        && _despawn_operators.size() != _units.size()) {
        atlas::logger::error()
            << "Sink::Builder: despawn operators must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn operators must have size 1 or match the unit count.");
    }

    if (!_despawn_types.empty()
        && _despawn_types.size() != 1
        && _despawn_types.size() != _units.size()) {
        atlas::logger::error()
            << "Sink::Builder: despawn types must have size 1 or match the unit count.";
        throw std::runtime_error(
            "Sink::Builder: despawn types must have size 1 or match the unit count.");
    }

    if (!std::isfinite(_tolerance)) {
        atlas::logger::error()
            << "Sink::Builder: tolerance must be finite.";
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

}
