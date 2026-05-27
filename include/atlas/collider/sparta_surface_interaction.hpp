#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <tuple>

namespace atlas::system {

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder
SpartaSurfaceInteraction<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_accommodation(const T accommodation) noexcept {
    _accommodation = accommodation;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    _temperature = temperature;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_particle_properties(const MaterialProperties<T>& properties) noexcept {
    _particle_properties = properties;
    if (properties.molecular_mass > T(0)) {
        _molecular_mass = properties.molecular_mass;
    }
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_wall_properties(const MaterialProperties<T>& properties) noexcept {
    _wall_properties = properties;
    if (properties.reference_temperature.has_value()) {
        _temperature = properties.reference_temperature.value();
    }
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_wall_velocity(const Vector3<T>& velocity) noexcept {
    _wall_velocity = velocity;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept {
    _wall_angular_velocity = angular_velocity;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_wall_rotation_origin(const Vector3<T>& origin) noexcept {
    _wall_rotation_origin = origin;
}

template <typename T>
void
SpartaSurfaceInteraction<T>::set_no_slip(const bool no_slip) noexcept {
    _no_slip = no_slip;
}

template <typename T>
T
SpartaSurfaceInteraction<T>::accommodation() const noexcept {
    return _accommodation;
}

template <typename T>
T
SpartaSurfaceInteraction<T>::temperature() const noexcept {
    return _temperature;
}

template <typename T>
T
SpartaSurfaceInteraction<T>::molecular_mass() const noexcept {
    return _molecular_mass;
}

template <typename T>
const MaterialProperties<T>&
SpartaSurfaceInteraction<T>::particle_properties() const noexcept {
    return _particle_properties;
}

template <typename T>
const MaterialProperties<T>&
SpartaSurfaceInteraction<T>::wall_properties() const noexcept {
    return _wall_properties;
}

template <typename T>
const Vector3<T>&
SpartaSurfaceInteraction<T>::wall_velocity() const noexcept {
    return _wall_velocity;
}

template <typename T>
const Vector3<T>&
SpartaSurfaceInteraction<T>::wall_angular_velocity() const noexcept {
    return _wall_angular_velocity;
}

template <typename T>
const Vector3<T>&
SpartaSurfaceInteraction<T>::wall_rotation_origin() const noexcept {
    return _wall_rotation_origin;
}

template <typename T>
bool
SpartaSurfaceInteraction<T>::no_slip() const noexcept {
    return _no_slip;
}

template <typename T>
Vector3<T>
SpartaSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                        const Vector3<T>& normal) const noexcept {
    if (!(incident.length_squared() > T(atlas::tol))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    if (_no_slip) {
        return -incident;
    }

    const Vector3<T> specular = atlas::math::reflected(incident, normal);
    if (_accommodation <= T(0)) {
        return specular;
    }

    const T mix = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));
    if (mix > _accommodation) {
        return specular;
    }

    return diffuse_reflection(incident, normal);
}

template <typename T>
typename SpartaSurfaceInteraction<T>::ParticleState
SpartaSurfaceInteraction<T>::collide(const ParticleState& state,
                                     const Vector3<T>& point_of_intersection,
                                     const Vector3<T>& normal) const noexcept {
    ParticleState result = state;

    const Vector3<T> local_wall_velocity = wall_velocity_at(point_of_intersection);
    const Vector3<T> relative_incident = state.velocity - local_wall_velocity;
    const Vector3<T> reflected_relative = (*this)(relative_incident, normal);

    result.position = point_of_intersection;
    result.velocity = reflected_relative + local_wall_velocity;

    if (!_no_slip && _accommodation > T(0)) {
        const T mix = atlas::sampling::sample_hashed_unit_interval(
            relative_incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
            T(atlas::seed::RANDOM_HASH_SALT_MIX));
        if (mix <= _accommodation) {
            result.internal_energy = diffuse_internal_energy(state.internal_energy);
        }
    }

    return result;
}

template <typename T>
T
SpartaSurfaceInteraction<T>::effective_molecular_mass() const noexcept {
    return _particle_properties.molecular_mass > T(0)
        ? _particle_properties.molecular_mass
        : _molecular_mass;
}

template <typename T>
Vector3<T>
SpartaSurfaceInteraction<T>::wall_velocity_at(const Vector3<T>& point) const noexcept {
    return _wall_velocity + atlas::math::cross(
        _wall_angular_velocity,
        point - _wall_rotation_origin);
}

template <typename T>
Vector3<T>
SpartaSurfaceInteraction<T>::diffuse_reflection(const Vector3<T>& incident,
                                                const Vector3<T>& normal) const noexcept {
    const T molecular_mass = effective_molecular_mass();
    const T vrm = static_cast<T>(std::sqrt(static_cast<double>(
        T(2) * static_cast<T>(atlas::boltzmann_constant) * _temperature / molecular_mass)));

    const T u_perp = std::max(
        atlas::sampling::sample_hashed_unit_interval(
            incident,
            T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1)),
        static_cast<T>(atlas::eps));
    const T u_theta = atlas::sampling::sample_hashed_unit_interval(
        incident + normal,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));
    const T u_tangent = std::max(
        atlas::sampling::sample_hashed_unit_interval(
            incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
            T(atlas::seed::RANDOM_HASH_SALT_MIX)),
        static_cast<T>(atlas::eps));

    const T vperp = vrm * static_cast<T>(std::sqrt(static_cast<double>(-std::log(u_perp))));
    const T vtangent = vrm * static_cast<T>(std::sqrt(static_cast<double>(-std::log(u_tangent))));
    const T theta = T(2) * static_cast<T>(std::numbers::pi_v<double>) * u_theta;
    const T vtan1 = vtangent * static_cast<T>(std::sin(static_cast<double>(theta)));
    const T vtan2 = vtangent * static_cast<T>(std::cos(static_cast<double>(theta)));

    const Vector3<T> unit_normal = atlas::math::normalize(normal);
    const T normal_component = atlas::math::dot(incident, unit_normal);
    Vector3<T> tangent1 = incident - unit_normal * normal_component;

    if (!(tangent1.length_squared() > T(atlas::tol))) {
        tangent1 = std::get<0>(unit_normal.tangential());
    } else {
        tangent1 = atlas::math::normalize(tangent1);
    }
    const Vector3<T> tangent2 = atlas::math::cross(unit_normal, tangent1);

    return unit_normal * vperp + tangent1 * vtan1 + tangent2 * vtan2;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::InternalEnergyState
SpartaSurfaceInteraction<T>::diffuse_internal_energy(const InternalEnergyState& old_energy) const noexcept {
    InternalEnergyState energy = old_energy;
    energy.rotational = _wall_properties.rotational_energy.value_or(
        _particle_properties.rotational_energy.value_or(energy.rotational));
    energy.vibrational = _wall_properties.vibrational_energy.value_or(
        _particle_properties.vibrational_energy.value_or(energy.vibrational));
    return energy;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_accommodation(const T accommodation) noexcept {
    _accommodation = accommodation;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_particle_properties(const MaterialProperties<T>& properties) noexcept {
    _particle_properties = properties;
    if (properties.molecular_mass > T(0)) {
        _molecular_mass = properties.molecular_mass;
    }
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_wall_properties(const MaterialProperties<T>& properties) noexcept {
    _wall_properties = properties;
    if (properties.reference_temperature.has_value()) {
        _temperature = properties.reference_temperature.value();
    }
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_wall_velocity(const Vector3<T>& velocity) noexcept {
    _wall_velocity = velocity;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept {
    _wall_angular_velocity = angular_velocity;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_wall_rotation_origin(const Vector3<T>& origin) noexcept {
    _wall_rotation_origin = origin;
    return *this;
}

template <typename T>
typename SpartaSurfaceInteraction<T>::Builder&
SpartaSurfaceInteraction<T>::Builder::with_no_slip(const bool no_slip) noexcept {
    _no_slip = no_slip;
    return *this;
}

template <typename T>
SpartaSurfaceInteraction<T>
SpartaSurfaceInteraction<T>::Builder::build() const {
    validate();

    SpartaSurfaceInteraction<T> interaction {};
    interaction.set_accommodation(_accommodation);
    interaction.set_temperature(_temperature);
    interaction.set_molecular_mass(_molecular_mass);
    interaction.set_particle_properties(_particle_properties);
    interaction.set_wall_properties(_wall_properties);
    interaction.set_wall_velocity(_wall_velocity);
    interaction.set_wall_angular_velocity(_wall_angular_velocity);
    interaction.set_wall_rotation_origin(_wall_rotation_origin);
    interaction.set_no_slip(_no_slip);
    return interaction;
}

template <typename T>
atlas::host_shared_ptr<SpartaSurfaceInteraction<T>>
SpartaSurfaceInteraction<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<SpartaSurfaceInteraction<T>>(build());
}

template <typename T>
void
SpartaSurfaceInteraction<T>::Builder::validate() const {
    if (!std::isfinite(_accommodation) || _accommodation < T(0) || _accommodation > T(1)) {
        throw std::runtime_error(
            "SpartaSurfaceInteraction::Builder: accommodation must be finite and within [0, 1].");
    }
    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        throw std::runtime_error(
            "SpartaSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
    if (!std::isfinite(_molecular_mass) || !(_molecular_mass > T(0))) {
        throw std::runtime_error(
            "SpartaSurfaceInteraction::Builder: molecular mass must be finite and positive.");
    }
}

} // namespace atlas::system
