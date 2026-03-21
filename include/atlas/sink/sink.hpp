#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/tuple/tuple.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Sink<T>::Sink(Unit<T> unit,
              const DespawnType despawn_type,
              const T tolerance) noexcept
    : _unit(std::move(unit))
    , _despawn_operator(despawn_type)
    , _tolerance(tolerance) { }

template <typename T>
typename Sink<T>::Builder
Sink<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Sink<T>::sink(ParticleDeviceProbe<T>& particle_probe) {
    const auto sync_op          = _unit.sync_operator();
    const auto query            = _unit.query_operator();
    const auto despawn_operator = _despawn_operator;
    const T tol                 = _tolerance;

    auto pos       = particle_probe.pos;
    auto vel       = particle_probe.vel;
    auto species   = particle_probe.species;
    auto zip_begin = atlas::make_zip_iterator(
        atlas::make_tuple(pos, vel, species));
    auto zip_end = zip_begin + particle_probe.particle_count;

    auto new_end = atlas::remove_if(
        atlas::device,
        zip_begin,
        zip_end,
        [=] ATLAS_DEVICE(const atlas::tuple<Vector3<T>, Vector3<T>, size_t>& t) {
            const Vector3<T>& p      = atlas::get<0>(t);
            const Vector3<T> local_p = sync_op.sync_to_local(p);
            return despawn_operator.despawn(query, local_p, tol);
        });

    particle_probe.particle_count = static_cast<int>(new_end - zip_begin);
}

template <typename T>
void
Sink<T>::set_unit(Unit<T> unit) noexcept {
    _unit = std::move(unit);
}

template <typename T>
void
Sink<T>::set_despawn_type(const DespawnType despawn_type) noexcept {
    _despawn_operator = DespawnOperator<T>(despawn_type);
}

template <typename T>
void
Sink<T>::set_despawn_operator(const DespawnOperator<T> despawn_operator) noexcept {
    _despawn_operator = despawn_operator;
}

template <typename T>
void
Sink<T>::set_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
}

template <typename T>
const Unit<T>&
Sink<T>::unit() const noexcept {
    return _unit;
}

template <typename T>
DespawnType
Sink<T>::despawn_type() const noexcept {
    return _despawn_operator.type;
}

template <typename T>
const DespawnOperator<T>&
Sink<T>::despawn_operator() const noexcept {
    return _despawn_operator;
}

template <typename T>
T
Sink<T>::tolerance() const noexcept {
    return _tolerance;
}

template <typename T>
Sink<T>
Sink<T>::Builder::build() {
    validate();
    return Sink<T>(std::move(*_unit), _despawn_type, _tolerance);
}

template <typename T>
atlas::host_shared_ptr<Sink<T>>
Sink<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Sink<T>>(build());
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_unit(const Unit<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_unit(Unit<T>&& unit) noexcept {
    _unit = std::move(unit);
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_despawn_type(const DespawnType despawn_type) noexcept {
    _despawn_type = despawn_type;
    return *this;
}

template <typename T>
typename Sink<T>::Builder&
Sink<T>::Builder::with_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

template <typename T>
void
Sink<T>::Builder::validate() const {
    if (!_unit.has_value()) {
        atlas::logger::error()
            << "Sink::Builder: unit must be provided.";
        throw std::runtime_error("Sink::Builder: unit must be provided.");
    }

    if (!std::isfinite(_tolerance)) {
        atlas::logger::error()
            << "Sink::Builder: tolerance must be finite.";
        throw std::runtime_error("Sink::Builder: tolerance must be finite.");
    }
}

} // namespace atlas::system
