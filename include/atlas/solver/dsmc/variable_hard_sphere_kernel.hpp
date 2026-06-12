#pragma once

#include <atlas/math/math.h>
#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas {

template <typename T>
T
VariableHardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                           const MaterialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // Bird VHS requires reference diameter and reference temperature for both species.
    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()
        || !lhs.reference_temperature.has_value() || !rhs.reference_temperature.has_value()) {
        return T(0);
    }

    // Molecular masses and relative speed must be positive.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))
        || !(relative_speed > T(0))) {
        return T(0);
    }

    // Use arithmetic pair averages for VHS reference parameters.
    const T reference_diameter    = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
    const T reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * T(0.5);
    const T viscosity_index       = (lhs.viscosity_index.value_or(T(0.5)) + rhs.viscosity_index.value_or(T(0.5)))
        * T(0.5);

    if (!(reference_diameter > T(0)) || !(reference_temperature > T(0))) {
        return T(0);
    }

    // Bird VHS denominator contains Gamma(2.5 - omega).
    const T gamma_argument = T(2.5) - viscosity_index;
    if (!(gamma_argument > T(0))) {
        return T(0);
    }

    // Reduced mass for the species pair.
    const T reduced_mass = lhs_mass * rhs_mass / mass_sum;

    // Dimensionless thermal ratio:
    //
    //   2 k_B T_ref / (m_r g^2)
    //
    // where g is the relative speed.
    const T thermal_ratio = (T(2) * static_cast<T>(atlas::boltzmann_constant) * reference_temperature)
        / (reduced_mass * relative_speed * relative_speed);

    if (!(thermal_ratio > T(0))) {
        return T(0);
    }

    // Reference cross-sectional area: pi * d_ref^2.
    const T reference_area = static_cast<T>(atlas::pi) * reference_diameter * reference_diameter;

    // Evaluate Gamma(2.5 - omega).
    const T gamma_value = static_cast<T>(std::tgamma(static_cast<double>(gamma_argument)));
    if (!(gamma_value > T(0))) {
        return T(0);
    }

    // Bird VHS total collision cross section.
    return reference_area * std::pow(thermal_ratio, viscosity_index - T(0.5)) / gamma_value;
}

template <typename T>
void
VariableHardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs) const noexcept {
    // Positive molecular masses are required for a valid center-of-mass update.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    // VHS uses hard-sphere-style isotropic scattering, equivalent to alpha = 1 here.
    const T scattering_parameter = T(1);

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

    // Build the incoming relative-velocity direction.
    const Vector3<T> axis = relative / speed;

    // Build a deterministic pair-dependent seed.
    // This avoids per-thread RNG state while still producing reproducible samples.
    const Vector3<T> sample_seed = relative + center * T(atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX)
        + Vector3<T>(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    // Generate two hash-based pseudo-random samples in [0, 1].
    const T u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U2));

    // Sample the polar scattering angle.
    // With scattering_parameter = 1, this reduces to isotropic HS scattering:
    //   cos_chi = 2u1 - 1.
    const T cos_chi = T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);

    // Sample the azimuthal angle uniformly in [0, 2*pi].
    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    // Rotate the relative-velocity direction while preserving its magnitude.
    const Vector3<T> scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Vector3<T> scattered_relative = scattered_axis * speed;

    // Reconstruct post-collision velocities from the center-of-mass frame.
    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

} // namespace atlas
