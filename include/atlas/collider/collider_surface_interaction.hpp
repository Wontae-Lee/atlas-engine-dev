#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder
ColliderSurfaceInteraction<T>::builder() noexcept {
    // Return a fresh builder object for staged construction of
    // `ColliderSurfaceInteraction<T>`.
    //
    // This entry point is useful when interaction parameters such as
    // restitution, accommodation coefficient, and temperature are
    // configured incrementally before the final object is created.
    return Builder {};
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_diffuse_sampling(const DiffuseSampling mode) noexcept {
    // Store the diffuse hemisphere sampling strategy that will be used
    // whenever the interaction selects a diffuse scattering event.
    //
    // Typical interpretation:
    // - UniformHemisphere   : all hemisphere directions are sampled evenly
    // - CosineWeighted      : directions near the surface normal are favored
    _diffuse_sampling = mode;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_restitution(const T restitution_coeff) noexcept {
    // Store the restitution coefficient that scales the outgoing velocity magnitude.
    //
    // Intended physical role:
    // - values near 1 preserve more post-collision speed
    // - values near 0 strongly damp the reflected/scattered response
    _restitution_coeff = restitution_coeff;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_tangential_momentum_accommodation(const T tmac) noexcept {
    // Store the tangential momentum accommodation coefficient.
    //
    // Intended interpretation:
    // - `0` means fully specular behavior
    // - `1` means fully diffuse behavior
    // - intermediate values probabilistically mix the two responses
    _tmac = tmac;
}

template <typename T>
void
ColliderSurfaceInteraction<T>::set_temperature(const T temperature) noexcept {
    // Store the surface temperature associated with the interaction model.
    //
    // The current implementation does not yet use this value directly during
    // scattering, but it is preserved as part of the interaction state for
    // future physically richer wall models.
    _temperature = temperature;
}

template <typename T>
DiffuseSampling
ColliderSurfaceInteraction<T>::diffuse_sampling() const noexcept {
    // Return the currently configured diffuse sampling mode.
    return _diffuse_sampling;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::restitution() const noexcept {
    // Return the currently configured restitution coefficient.
    return _restitution_coeff;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::tangential_momentum_accommodation() const noexcept {
    // Return the currently configured tangential momentum accommodation coefficient.
    return _tmac;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::temperature() const noexcept {
    // Return the currently configured surface temperature.
    return _temperature;
}

template <typename T>
T
ColliderSurfaceInteraction<T>::hashed_unit_interval(const Vector3<T>& seed, const T salt) noexcept {
    // Map the input seed and salt deterministically into the unit interval [0, 1).
    //
    // Design intent:
    // - produce inexpensive pseudo-random variation without maintaining RNG state
    // - generate decorrelated values from geometric inputs such as directions
    // - keep the result deterministic for identical inputs
    //
    // The specific constants are chosen only to scramble the input phase space;
    // they do not imply any rigorous statistical guarantee.
    const T phase = seed.x * T(12.9898) + seed.y * T(78.233) + seed.z * T(37.719) + salt;

    // Apply a nonlinear transform so nearby seeds do not map linearly.
    const T value = std::sin(phase) * T(43758.5453);

    // Remove the integer part and keep only the fractional component,
    // yielding a value in [0, 1).
    return value - std::floor(value);
}

template <typename T>
Vector3<T>
ColliderSurfaceInteraction<T>::operator()(const Vector3<T>& incident,
                                          const Vector3<T>& normal) const noexcept {
    // Compute the ideal specular reflection direction first.
    //
    // This is the baseline answer for the interaction model and becomes
    // the final result whenever the accommodation coefficient is zero.
    const Vector3<T> specular_dir = atlas::math::reflected(incident, normal);

    // Fast path for purely specular interaction:
    // - no diffuse sampling is needed
    // - the outgoing direction is just the reflected direction scaled
    //   by the restitution coefficient
    if (_tmac <= T(0)) {
        return specular_dir * _restitution_coeff;
    }

    // Generate two deterministic pseudo-random numbers from the local
    // geometric state. These values drive hemisphere sampling.
    //
    // `u1` and `u2` serve as canonical random inputs for diffuse-direction samplers.
    const T u1 = hashed_unit_interval(incident, T(0.31));
    const T u2 = hashed_unit_interval(normal + incident, T(1.73));

    Vector3<T> diffuse_dir {};

    if (_diffuse_sampling == DiffuseSampling::CosineWeighted) {
        // Sample a diffuse direction with cosine weighting around the surface normal.
        //
        // This biases directions toward the normal and approximates a Lambertian-style
        // diffuse scattering distribution.
        diffuse_dir = atlas::sampling::sample_cosine_hemisphere(normal, u1, u2);
    } else {
        // Sample a diffuse direction uniformly over the hemisphere oriented by `normal`.
        //
        // Every hemisphere direction is equally likely under this mode.
        diffuse_dir = atlas::sampling::sample_uniform_hemisphere(normal, u1, u2);
    }

    // Draw a second deterministic mixing sample to choose whether this event
    // behaves as diffuse or specular.
    //
    // Interpretation:
    // - probability of diffuse event  = `_tmac`
    // - probability of specular event = `1 - _tmac`
    const T mix = hashed_unit_interval(incident + normal * T(17), T(2.41));

    // Select the outgoing direction according to the accommodation coefficient.
    const Vector3<T> out_dir = (mix < _tmac) ? diffuse_dir : specular_dir;

    // Apply restitution after the directional choice so both specular and diffuse
    // outcomes are attenuated consistently by the same energy-loss factor.
    return out_dir * _restitution_coeff;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {
    // Store the diffuse sampling mode in the builder's staged state.
    _diffuse_sampling = mode;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_restitution(const T restitution) noexcept {
    // Store the restitution coefficient in the builder.
    //
    // Final validity is checked later in `validate()`.
    _restitution = restitution;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_tangential_momentum_accommodation(const T tmac) noexcept {
    // Store the accommodation coefficient in the builder.
    //
    // Final validity is checked later in `validate()`.
    _tmac = tmac;
    return *this;
}

template <typename T>
typename ColliderSurfaceInteraction<T>::Builder&
ColliderSurfaceInteraction<T>::Builder::with_temperature(const T temperature) noexcept {
    // Store the surface temperature in the builder.
    //
    // Final validity is checked later in `validate()`.
    _temperature = temperature;
    return *this;
}

template <typename T>
ColliderSurfaceInteraction<T>
ColliderSurfaceInteraction<T>::Builder::build() const {
    // Validate all staged parameters before materializing the final object.
    validate();

    // Construct the interaction object using its setter interface so all field
    // assignment goes through the same public configuration path.
    ColliderSurfaceInteraction<T> interaction {};
    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_tangential_momentum_accommodation(_tmac);
    interaction.set_temperature(_temperature);
    return interaction;
}

template <typename T>
atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>
ColliderSurfaceInteraction<T>::Builder::make_host_shared() const {
    // Build the validated interaction object by value and place it into
    // host-shared managed storage.
    return atlas::make_host_shared<ColliderSurfaceInteraction<T>>(build());
}

template <typename T>
void
ColliderSurfaceInteraction<T>::Builder::validate() const {
    // Validate restitution.
    //
    // Requirements:
    // - must be finite
    // - must be non-negative
    //
    // Negative restitution would be non-physical for this model, and non-finite
    // values would make runtime scattering behavior ill-defined.
    if (!std::isfinite(_restitution) || _restitution < T(0)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }

    // Validate tangential momentum accommodation coefficient.
    //
    // Requirements:
    // - must be finite
    // - must lie in the closed interval [0, 1]
    //
    // This range is required because `_tmac` is interpreted as a mixing probability
    // between specular and diffuse scattering.
    if (!std::isfinite(_tmac) || _tmac < T(0) || _tmac > T(1)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: tmac must be finite and within [0, 1].");
    }

    // Validate temperature.
    //
    // Requirements:
    // - must be finite
    // - must be non-negative
    //
    // Although temperature is not yet used directly in `operator()`, the builder
    // still enforces physically meaningful state for future model extensions.
    if (!std::isfinite(_temperature) || _temperature < T(0)) {
        atlas::logger::error()
            << "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.";
        throw std::runtime_error(
            "ColliderSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

} // namespace atlas::system