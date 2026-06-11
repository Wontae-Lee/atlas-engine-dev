#pragma once

#include <atlas/math/math.h>
#include <atlas/sampling/sampling.h>

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
    return static_cast<T>(atlas::pi) * diameter * diameter;
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

    // Build the incoming relative-velocity direction.
    const Vector3<T> axis = relative / speed;

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
    const T cos_chi = T(2) * u1 - T(1);
    const T phi     = T(2) * static_cast<T>(atlas::pi) * u2;

    // Rotate the relative-velocity direction while preserving the relative speed.
    const Vector3<T> scattered_axis     = atlas::math::spherical_direction(axis, cos_chi, phi);
    const Vector3<T> scattered_relative = scattered_axis * speed;

    // Reconstruct post-collision velocities from the center-of-mass frame.
    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

} // namespace atlas::system
