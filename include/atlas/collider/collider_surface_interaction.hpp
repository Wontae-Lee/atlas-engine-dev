#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder
ColliderSurfaceInteraction<T>::builder() noexcept {

    // Return a default-initialized builder so callers can configure the
    // interaction model through a fluent API before constructing the final object.
    return Builder {};
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_diffuse_sampling(const DiffuseSampling mode) noexcept {

    // Select which hemisphere sampling strategy is used when the interaction
    // chooses a diffuse reflection event.
    _diffuse_sampling = mode;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_restitution(const T restitution_coeff) noexcept {

    // Store the restitution coefficient used to scale the outgoing velocity
    // magnitude after the reflection direction has been chosen.
    _restitution_coeff = restitution_coeff;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_tangential_momentum_accommodation(const T tmac) noexcept {

    // Store the tangential momentum accommodation coefficient.
    //
    // Conceptually:
    // - tmac = 0   -> purely specular reflection
    // - tmac = 1   -> purely diffuse reflection
    // - 0 < tmac < 1 -> stochastic mixture of the two behaviors
    _tmac = tmac;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {

    // Store the surface temperature associated with this interaction model.
    //
    // The current operator() implementation does not explicitly use this value,
    // but it remains part of the interaction state for future extensions or
    // coupled thermal/surface models.
    _temperature = temperature;
}

template <typename T>
DiffuseSampling
ColliderSurfaceInteraction<T>::diffuse_sampling() const noexcept {

    // Return the configured diffuse hemisphere sampling mode.
    return _diffuse_sampling;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::restitution() const noexcept {

    // Return the restitution coefficient applied to outgoing velocity.
    return _restitution_coeff;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::tangential_momentum_accommodation() const noexcept {

    // Return the probability-like mixing parameter controlling diffuse vs.
    // specular reflection selection.
    return _tmac;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::temperature() const noexcept {

    // Return the stored surface temperature parameter.
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

    // Compute the ideal mirror-reflection direction around the surface normal.
    //
    // This preserves the incident speed and changes only direction.
    const Vector3<T> specular_dir = atlas::math::reflected(incident, normal);
    const Vector3<T> specular_unit = atlas::math::normalize(specular_dir);

    // Fast path for purely specular reflection.
    //
    // When TMAC is zero or negative, no diffuse scattering is introduced.
    // The outgoing speed is the incident speed scaled by restitution.
    if (_tmac <= T(0)) {
        return specular_unit * (incident_speed * _restitution_coeff);
    }

    // Generate two deterministic pseudo-random samples in [0, 1] from the input
    // vectors. Using hashed sampling keeps the interaction reproducible for the
    // same incident/normal pair while still providing stochastic-looking behavior.
    const T u1 = atlas::sampling::sample_hashed_unit_interval(incident,
                                                              T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u2 = atlas::sampling::sample_hashed_unit_interval(normal + incident,
                                                              T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    Vector3<T> diffuse_dir {};

    // Sample a diffuse outgoing direction over the hemisphere oriented by the
    // surface normal.
    //
    // Cosine-weighted sampling favors directions closer to the normal and is a
    // common model for Lambertian-like diffuse reflection.
    //
    // Uniform sampling treats all hemisphere directions equally.
    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {

        diffuse_dir = atlas::sampling::sample_cosine_hemisphere(normal, u1, u2);
    } else {

        diffuse_dir = atlas::sampling::sample_uniform_hemisphere(normal, u1, u2);
    }

    // Generate another deterministic pseudo-random sample used to choose between
    // the diffuse and specular branches.
    //
    // The probability threshold is TMAC:
    // - mix < TMAC  -> choose diffuse reflection
    // - otherwise   -> choose specular reflection
    const T mix = atlas::sampling::sample_hashed_unit_interval(
        incident + normal * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX),
        T(atlas::seed::RANDOM_HASH_SALT_MIX));

    // Select the final outgoing unit direction by stochastically mixing the
    // diffuse and specular models according to TMAC. Speed scaling is handled
    // separately through restitution.
    const Vector3<T> out_unit = (mix < _tmac)
        ? atlas::math::normalize(diffuse_dir)
        : specular_unit;

    return out_unit * (incident_speed * _restitution_coeff);
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {

    // Configure the diffuse hemisphere sampling strategy to use in the final
    // interaction object.
    _diffuse_sampling = mode;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_restitution(const T restitution) noexcept {

    // Configure the restitution coefficient that scales outgoing speed relative
    // to the incident speed.
    _restitution = restitution;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_tangential_momentum_accommodation(const T tmac) noexcept {

    // Configure the diffuse/specular mixing factor.
    _tmac = tmac;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {

    // Configure the surface temperature stored in the final interaction object.
    _temperature = temperature;
    return *this;
}

template <typename T>
ColliderSurfaceInteraction<T>
ColliderSurfaceInteraction<T>::Builder::build() const {

    // Validate all user-provided parameters before constructing the final
    // interaction object.
    validate();

    ColliderSurfaceInteraction<T> interaction {};

    // Transfer builder configuration into the concrete interaction instance.
    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_tangential_momentum_accommodation(_tmac);
    interaction.set_temperature(_temperature);

    return interaction;
}

template <typename T>
atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
ColliderSurfaceInteraction<T>::Builder::make_host_shared() const {

    // Convenience helper that constructs the interaction object first and then
    // places it into host-managed shared ownership.
    return atlas::make_host_shared<ColliderSurfaceInteraction<T>>(build());
}

template <typename T>
void
ColliderSurfaceInteraction<T>::Builder::validate() const {

    // Restitution must be a valid finite value and cannot be negative.
    //
    // Negative restitution would invert the post-collision scaling semantics and
    // is therefore rejected.
    if (!std::isfinite(_restitution) || _restitution < T(0)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }

    // TMAC must lie in the closed interval [0, 1].
    //
    // It acts as a mixing probability / accommodation factor, so values outside
    // this range would not have a physically meaningful interpretation in the
    // current model.
    if (!std::isfinite(_tmac) || _tmac < T(0) || _tmac > T(1)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].");
    }

    // Temperature must be finite and non-negative.
    //
    // Even though the current reflection operator does not directly use the
    // stored temperature, the builder still enforces a physically meaningful
    // domain for the parameter.
    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

}
