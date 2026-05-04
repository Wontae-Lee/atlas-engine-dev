#pragma once
#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>
#include <cmath>
#include <stdexcept>
namespace atlas::system {
template <typename T>
typename ColliderSurfaceInteraction<T>::Builder
ColliderSurfaceInteraction<T>::builder() noexcept {
    return Builder {}

    ;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_diffuse_sampling(const DiffuseSampling mode) noexcept {
    _diffuse_sampling = mode;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_restitution(const T restitution_coeff) noexcept {
    _restitution_coeff = restitution_coeff;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_tangential_momentum_accommodation(const T tmac) noexcept {
    _tmac = tmac;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    _temperature = temperature;
}

template <typename T>
DiffuseSampling
ColliderSurfaceInteraction<T>::diffuse_sampling() const noexcept {
    return _diffuse_sampling;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::restitution() const noexcept {
    return _restitution_coeff;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::tangential_momentum_accommodation() const noexcept {
    return _tmac;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::temperature() const noexcept {
    return _temperature;
}

template <typename T>
Vector3<T>
ColliderSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                          const Vector3<T>& normal) const noexcept {
    const T incident_speed = incident.length();
    if (incident_speed <= T(atlas::tol)) {
        return Vector3<T>(T(0), T(0), T(0));
    }
    const Vector3<T> specular_dir  = atlas::math::reflected(incident, normal);
    const Vector3<T> specular_unit = atlas::math::normalize(specular_dir);
    if (_tmac <= T(0)) {
        return specular_unit * (incident_speed * _restitution_coeff);
    }
    const T u1 = atlas::sampling::sample_hashed_unit_interval(incident,
                                                              T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u2 = atlas::sampling::sample_hashed_unit_interval(normal + incident,
                                                              T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));
    Vector3<T> diffuse_dir {}

    ;
    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {
        diffuse_dir = atlas::sampling::sample_cosine_hemisphere(normal, u1, u2);
    } else {
        diffuse_dir = atlas::sampling::sample_uniform_hemisphere(normal, u1, u2);
    }
    const T mix = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));
    const Vector3<T> out_unit = (mix < _tmac)
        ? atlas::math::normalize(diffuse_dir)
        : specular_unit;
    return out_unit * (incident_speed * _restitution_coeff);
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {
    _diffuse_sampling = mode;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_restitution(const T restitution) noexcept {
    _restitution = restitution;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_tangential_momentum_accommodation(const T tmac) noexcept {
    _tmac = tmac;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
ColliderSurfaceInteraction<T>
ColliderSurfaceInteraction<T>::Builder::build() const {
    validate();
    ColliderSurfaceInteraction<T> interaction {}

    ;
    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_tangential_momentum_accommodation(_tmac);
    interaction.set_temperature(_temperature);
    return interaction;
}

template <typename T>
atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
ColliderSurfaceInteraction<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<ColliderSurfaceInteraction<T>>(build());
}

template <typename T>
void
ColliderSurfaceInteraction<T>::Builder::validate() const {
    if (!std::isfinite(_restitution) || _restitution < T(0)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }
    if (!std::isfinite(_tmac) || _tmac < T(0) || _tmac > T(1)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].");
    }
    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

}