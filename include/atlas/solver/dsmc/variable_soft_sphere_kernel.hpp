#pragma once

#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <cmath>
#include <numbers>

namespace atlas::system {

template <typename T>
T
VariableSoftSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                           const MaterialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // VSS uses the same total collision cross-section law as VHS.
    // The scattering parameter affects only the angular scattering step.
    return VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed);
}

template <typename T>
void
VariableSoftSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs) const noexcept {
    // Use the arithmetic mean scattering parameter for the species pair.
    // Missing values fall back to 1, which gives isotropic hard-sphere scattering.
    const T scattering_parameter = (lhs.scattering_parameter.value_or(T(1))
                                    + rhs.scattering_parameter.value_or(T(1)))
        * T(0.5);

    // Positive molecular masses are required for a valid center-of-mass update.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))
        || !(scattering_parameter > T(0))) {
        return;
    }

    // Compute incoming relative velocity and speed.
    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();

    if (!(speed > T(0))) {
        return;
    }

    // Compute the center-of-mass velocity, which is preserved by the collision.
    const Vector3<T> center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    // Build an orthonormal basis around the incoming relative-velocity direction.
    const Vector3<T> axis = relative / speed;
    const auto tangents   = axis.tangential();
    const Vector3<T> t1   = std::get<0>(tangents);
    const Vector3<T> t2   = std::get<1>(tangents);

    // Build a deterministic pair-dependent seed.
    // This avoids per-thread RNG state while still producing reproducible samples.
    const Vector3<T> sample_seed = relative + center * T(atlas::seed::RANDOM_HASH_NORMAL_SCALE_FOR_MIX)
        + Vector3<T>(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    // Generate two hash-based pseudo-random samples in [0, 1].
    const T u1 = atlas::sampling::sample_hashed_unit_interval(
        sample_seed,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u2 = atlas::sampling::sample_hashed_unit_interval(
        sample_seed + axis,
        T(atlas::seed::RANDOM_HASH_SALT_DIFFUSE_U2));

    // Sample the VSS polar scattering angle:
    //
    //   cos_chi = 2 * u1^(1 / alpha) - 1
    //
    // where alpha is the averaged scattering parameter.
    const T cos_chi = T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);

    const T sin_chi_sq = T(1) - cos_chi * cos_chi;
    const T sin_chi    = sin_chi_sq > T(0)
           ? static_cast<T>(std::sqrt(static_cast<double>(sin_chi_sq)))
           : T(0);

    // Sample the azimuthal angle uniformly in [0, 2*pi].
    const T phi     = T(2) * static_cast<T>(std::numbers::pi_v<double>) * u2;
    const T cos_phi = static_cast<T>(std::cos(static_cast<double>(phi)));
    const T sin_phi = static_cast<T>(std::sin(static_cast<double>(phi)));

    // Rotate the relative-velocity direction while preserving its magnitude.
    const Vector3<T> scattered_axis     = axis * cos_chi + (t1 * cos_phi + t2 * sin_phi) * sin_chi;
    const Vector3<T> scattered_relative = scattered_axis * speed;

    // Reconstruct post-collision velocities from the center-of-mass frame.
    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

} // namespace atlas::system