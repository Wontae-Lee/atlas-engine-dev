#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder
ColliderSurfaceInteraction<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the interaction model
    // through a fluent API.
    return Builder {};
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_diffuse_sampling(const DiffuseSampling mode) noexcept {
    // Select the angular distribution used when a diffuse reflection is sampled.
    _diffuse_sampling = mode;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_restitution(const T restitution_coeff) noexcept {
    // Store the velocity scaling factor applied after surface reflection.
    _restitution_coeff = restitution_coeff;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_tangential_momentum_accommodation(const T tmac) noexcept {
    // Store the probability-like mixing factor between specular and diffuse reflection.
    _tmac = tmac;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    // Store the wall temperature for models that require thermal surface interaction.
    _temperature = temperature;
}

template <typename T>
DiffuseSampling
ColliderSurfaceInteraction<T>::diffuse_sampling() const noexcept {
    // Return the currently selected diffuse reflection sampling mode.
    return _diffuse_sampling;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::restitution() const noexcept {
    // Return the restitution coefficient used to scale outgoing speed.
    return _restitution_coeff;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::tangential_momentum_accommodation() const noexcept {
    // Return the diffuse/specular mixing coefficient.
    return _tmac;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::temperature() const noexcept {
    // Return the configured wall temperature.
    return _temperature;
}

template <typename T>
Vector3<T>
ColliderSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                          const Vector3<T>& normal) const noexcept {
    const T incident_speed = incident.length();

    // Degenerate incident velocities do not define a reliable reflection direction.
    if (incident_speed <= T(atlas::tol)) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Compute the deterministic specular reflection direction from the surface normal.
    const Vector3<T> specular_dir  = atlas::math::reflected(incident, normal);
    const Vector3<T> specular_unit = atlas::math::normalize(specular_dir);

    // With zero accommodation, the interaction reduces to pure specular reflection.
    if (_tmac <= T(0)) {
        return specular_unit * (incident_speed * _restitution_coeff);
    }

    // Generate deterministic pseudo-random samples from the local collision state.
    const T u1 = atlas::sampling::sample_hashed_unit_interval(
        incident,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));

    const T u2 = atlas::sampling::sample_hashed_unit_interval(
        normal + incident,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    Vector3<T> diffuse_dir {};

    // Sample a diffuse outgoing direction from the selected hemisphere distribution.
    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {
        diffuse_dir = atlas::sampling::sample_cosine_hemisphere(normal, u1, u2);
    } else {
        diffuse_dir = atlas::sampling::sample_uniform_hemisphere(normal, u1, u2);
    }

    // Draw the reflection branch: diffuse with probability _tmac, otherwise specular.
    const T mix = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));

    const Vector3<T> out_unit = (mix < _tmac)
        ? atlas::math::normalize(diffuse_dir)
        : specular_unit;

    // Preserve the incident speed up to the restitution coefficient.
    return out_unit * (incident_speed * _restitution_coeff);
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {
    // Configure the diffuse angular sampling mode used by the constructed interaction.
    _diffuse_sampling = mode;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_restitution(const T restitution) noexcept {
    // Configure the restitution coefficient used to scale reflected particle speed.
    _restitution = restitution;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_tangential_momentum_accommodation(const T tmac) noexcept {
    // Configure the diffuse/specular mixing factor for surface reflection.
    _tmac = tmac;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    // Configure the wall temperature stored by the interaction model.
    _temperature = temperature;
    return *this;
}

template <typename T>
ColliderSurfaceInteraction<T>
ColliderSurfaceInteraction<T>::Builder::build() const {
    // Validate physical parameters before creating the interaction object.
    validate();

    ColliderSurfaceInteraction<T> interaction {};

    // Transfer all builder-side parameters into the constructed interaction.
    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_tangential_momentum_accommodation(_tmac);
    interaction.set_temperature(_temperature);

    return interaction;
}

template <typename T>
atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
ColliderSurfaceInteraction<T>::Builder::make_host_shared() const {
    // Build a validated interaction and store it in host-managed shared ownership.
    return atlas::make_host_shared<ColliderSurfaceInteraction<T>>(build());
}

template <typename T>
void
ColliderSurfaceInteraction<T>::Builder::validate() const {
    // Restitution scales outgoing speed, so it must be finite and non-negative.
    if (!std::isfinite(_restitution) || _restitution < T(0)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }

    // TMAC is used as a mixing probability, so it must lie within the closed unit interval.
    if (!std::isfinite(_tmac) || _tmac < T(0) || _tmac > T(1)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].");
    }

    // Temperature is a physical scalar and must not be negative or non-finite.
    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

} // namespace atlas::system