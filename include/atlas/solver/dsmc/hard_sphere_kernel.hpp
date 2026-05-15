#pragma once

#include <atlas/math/vector/vector.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <numbers>

namespace atlas::system {

template <typename T>
T
HardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                   const MaterialProperties<T>& rhs) noexcept {
    // Both species must provide reference diameters to define a hard-sphere area.
    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()) {
        return T(0);
    }

    // Use the arithmetic mean pair diameter for mixed-species collisions.
    const T diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);

    // A non-positive diameter disables the pair collision cross section.
    if (!(diameter > T(0))) {
        return T(0);
    }

    // Hard-sphere total cross section: sigma = pi * d^2.
    return static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
}

template <typename T>
void
HardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                Vector3<T>& rhs_velocity,
                                const MaterialProperties<T>& lhs,
                                const MaterialProperties<T>& rhs) const noexcept {
    // Positive molecular masses are required for a valid center-of-mass update.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return;
    }

    // Compute the incoming relative velocity g = v_i - v_j and its magnitude.
    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();

    // Zero relative speed has no well-defined scattering direction.
    if (!(speed > T(0))) {
        return;
    }

    // Compute the center-of-mass velocity, which is preserved by the elastic collision.
    const Vector3<T> center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    // Build an orthonormal basis around the incoming relative-velocity direction.
    const Vector3<T> axis = relative / speed;
    const auto tangents   = axis.tangential();
    const Vector3<T> t1   = std::get<0>(tangents);
    const Vector3<T> t2   = std::get<1>(tangents);

    // Build a deterministic pair-dependent seed.
    // This avoids per-thread RNG state while still producing reproducible scattering samples.
    const Vector3<T> sample_seed = relative + center * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX)
        + Vector3<T>(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    // Generate two hash-based pseudo-random samples in [0, 1].
    const T u1 = atlas::sampling::sample_hashed_unit_interval(
        sample_seed,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));

    const T u2 = atlas::sampling::sample_hashed_unit_interval(
        sample_seed + axis,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    // Sample an isotropic post-collision direction:
    // cos(chi) is uniform in [-1, 1], and phi is uniform in [0, 2*pi].
    const T cos_chi    = T(2) * u1 - T(1);
    const T sin_chi_sq = T(1) - cos_chi * cos_chi;
    const T sin_chi    = sin_chi_sq > T(0)
           ? static_cast<T>(std::sqrt(static_cast<double>(sin_chi_sq)))
           : T(0);

    const T phi     = T(2) * static_cast<T>(std::numbers::pi_v<double>) * u2;
    const T cos_phi = static_cast<T>(std::cos(static_cast<double>(phi)));
    const T sin_phi = static_cast<T>(std::sin(static_cast<double>(phi)));

    // Rotate the relative-velocity direction while preserving the relative speed.
    const Vector3<T> scattered_axis     = axis * cos_chi + (t1 * cos_phi + t2 * sin_phi) * sin_chi;
    const Vector3<T> scattered_relative = scattered_axis * speed;

    // Reconstruct post-collision velocities from the center-of-mass frame.
    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

} // namespace atlas::system