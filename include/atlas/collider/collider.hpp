#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Collider<T>::Collider(UnitHostPtr<T> unit,
                      atlas::host_shared_ptr<ColliderSurfaceInteraction<T>> surface_interaction) noexcept
    : _unit(std::move(unit))
    , _surface_interaction(std::move(surface_interaction)) { }

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Collider<T>::set_unit(const UnitHostPtr<T>& unit) {
    if (!unit) {
        atlas::logger::error()
            << "Collider: unit must not be null.";
        throw std::runtime_error("Collider: unit must not be null.");
    }

    _unit = unit;
}

template <typename T>
void
Collider<T>::set_surface_interaction(
    const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>& surface_interaction) {
    if (!surface_interaction) {
        atlas::logger::error()
            << "Collider: surface interaction must not be null.";
        throw std::runtime_error("Collider: surface interaction must not be null.");
    }

    _surface_interaction = surface_interaction;
}

template <typename T>
const UnitHostPtr<T>&
Collider<T>::unit() const noexcept {
    return _unit;
}

template <typename T>
const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>&
Collider<T>::surface_interaction() const noexcept {
    return _surface_interaction;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_unit(const UnitHostPtr<T>& unit) {
    if (!unit) {
        atlas::logger::error()
            << "Collider::Builder: unit must not be null.";
        throw std::runtime_error("Collider::Builder: unit must not be null.");
    }

    _unit = unit;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction(
    const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>& surface_interaction) {
    if (!surface_interaction) {
        atlas::logger::error()
            << "Collider::Builder: surface interaction must not be null.";
        throw std::runtime_error("Collider::Builder: surface interaction must not be null.");
    }

    _surface_interaction = surface_interaction;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    validate();

    if (!_surface_interaction) {
        _surface_interaction = atlas::make_host_shared<ColliderSurfaceInteraction<T>>();
    }

    Collider<T> collider(std::move(_unit), std::move(_surface_interaction));
    _unit.reset();
    _surface_interaction.reset();
    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Collider<T>>(build());
}

template <typename T>
void
Collider<T>::Builder::validate() const {
    if (!_unit) {
        atlas::logger::error()
            << "Collider::Builder: unit must be provided.";
        throw std::runtime_error("Collider::Builder: unit must be provided.");
    }
}

}
