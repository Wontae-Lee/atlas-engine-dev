#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <stdexcept>

namespace atlas {

template <typename T>
typename IsothermalSurfaceInteraction<T>::Builder
IsothermalSurfaceInteraction<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
void
IsothermalSurfaceInteraction<T>::set_diffuse_sampling(const DiffuseSampling mode) noexcept {

    _diffuse_sampling = mode;
}

template <typename T>
void
IsothermalSurfaceInteraction<T>::set_restitution(const T restitution_coeff) noexcept {

    _restitution_coeff = restitution_coeff;
}

template <typename T>
void
IsothermalSurfaceInteraction<T>::set_momentum_acc(const T momentum_acc) noexcept {

    _momentum_acc = momentum_acc;
}

template <typename T>
void
IsothermalSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {

    _temperature = temperature;
}

template <typename T>
DiffuseSampling
IsothermalSurfaceInteraction<T>::diffuse_sampling() const noexcept {

    return _diffuse_sampling;
}

template <typename T>
T
IsothermalSurfaceInteraction<T>::restitution() const noexcept {

    return _restitution_coeff;
}

template <typename T>
T
IsothermalSurfaceInteraction<T>::momentum_acc() const noexcept {

    return _momentum_acc;
}

template <typename T>
T
IsothermalSurfaceInteraction<T>::temperature() const noexcept {

    return _temperature;
}

template <typename T>
Vector3<T>
IsothermalSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                            const Vector3<T>& normal) const noexcept {
    const T incident_speed = incident.length();

    if (incident_speed <= T(atlas::tol)) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const Vector3<T> specular_dir  = atlas::reflected(incident, normal);
    const Vector3<T> specular_unit = atlas::normalize(specular_dir);

    if (_momentum_acc <= T(0)) {
        return specular_unit * (incident_speed * _restitution_coeff);
    }

    const T u1 = atlas::sample_hashed_unit_interval(
        incident,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U1));

    const T u2 = atlas::sample_hashed_unit_interval(
        normal + incident,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U2));

    Vector3<T> diffuse_dir {};

    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {
        diffuse_dir = atlas::sample_cosine_hemisphere(normal, u1, u2);
    } else {
        diffuse_dir = atlas::sample_uniform_hemisphere(normal, u1, u2);
    }

    const T mix = atlas::sample_hashed_unit_interval(
        incident + normal * T(atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::RANDOM_HASH_SALT_MIX));

    const Vector3<T> out_unit = (mix < _momentum_acc)
        ? atlas::normalize(diffuse_dir)
        : specular_unit;

    return out_unit * (incident_speed * _restitution_coeff);
}

template <typename T>
FluidInternalEnergy<T>
IsothermalSurfaceInteraction<T>::internal_energy(
    const FluidInternalEnergy<T>& incident_energy,
    const Vector3<T>& incident_velocity,
    const Vector3<T>& normal,
    const MaterialProperties<T>& material) const noexcept {
    static_cast<void>(incident_velocity);
    static_cast<void>(normal);
    static_cast<void>(material);
    return incident_energy;
}

template <typename T>
typename IsothermalSurfaceInteraction<T>::Builder&
IsothermalSurfaceInteraction<T>::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {

    _diffuse_sampling = mode;
    return *this;
}

template <typename T>
typename IsothermalSurfaceInteraction<T>::Builder&
IsothermalSurfaceInteraction<T>::Builder::with_restitution(const T restitution) noexcept {

    _restitution = restitution;
    return *this;
}

template <typename T>
typename IsothermalSurfaceInteraction<T>::Builder&
IsothermalSurfaceInteraction<T>::Builder::with_momentum_acc(const T momentum_acc) noexcept {

    _momentum_acc = momentum_acc;
    return *this;
}

template <typename T>
typename IsothermalSurfaceInteraction<T>::Builder&
IsothermalSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {

    _temperature = temperature;
    return *this;
}

template <typename T>
IsothermalSurfaceInteraction<T>
IsothermalSurfaceInteraction<T>::Builder::build() const {

    validate();

    IsothermalSurfaceInteraction<T> interaction {};

    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_momentum_acc(_momentum_acc);
    interaction.set_temperature(_temperature);

    return interaction;
}

template <typename T>
atlas::host_shared_ptr<IsothermalSurfaceInteraction<T>>
IsothermalSurfaceInteraction<T>::Builder::make_host_shared() const {

    return atlas::make_host_shared<IsothermalSurfaceInteraction<T>>(build());
}

template <typename T>
void
IsothermalSurfaceInteraction<T>::Builder::validate() const {

    if (!atlas::isfinite(_restitution) || _restitution < T(0)) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }

    if (!atlas::isfinite(_momentum_acc) || _momentum_acc < T(0) || _momentum_acc > T(1)) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: momentum_acc must be finite and within [0, 1].");
    }

    if (!atlas::isfinite(_temperature) || _temperature < T(0)) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

}