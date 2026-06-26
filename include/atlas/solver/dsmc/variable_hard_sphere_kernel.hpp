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

    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()
        || !lhs.reference_temperature.has_value() || !rhs.reference_temperature.has_value()) {
        return T(0);
    }

    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))
        || !(relative_speed > T(0))) {
        return T(0);
    }

    const T reference_diameter    = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
    const T reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * T(0.5);
    const T viscosity_index       = (lhs.viscosity_index.value_or(T(0.5)) + rhs.viscosity_index.value_or(T(0.5)))
        * T(0.5);

    if (!(reference_diameter > T(0)) || !(reference_temperature > T(0))) {
        return T(0);
    }

    const T gamma_argument = T(2.5) - viscosity_index;
    if (!(gamma_argument > T(0))) {
        return T(0);
    }

    const T reduced_mass = lhs_mass * rhs_mass / mass_sum;

    const T thermal_ratio = (T(2) * static_cast<T>(atlas::boltzmann_constant) * reference_temperature)
        / (reduced_mass * relative_speed * relative_speed);

    if (!(thermal_ratio > T(0))) {
        return T(0);
    }

    const T reference_area = static_cast<T>(atlas::pi) * reference_diameter * reference_diameter;

    const T gamma_value = static_cast<T>(std::tgamma(static_cast<double>(gamma_argument)));
    if (!(gamma_value > T(0))) {
        return T(0);
    }

    return reference_area * std::pow(thermal_ratio, viscosity_index - T(0.5)) / gamma_value;
}

template <typename T>
void
VariableHardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs) const noexcept {

    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    const T scattering_parameter = T(1);

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))
        || !(scattering_parameter > T(0))) {
        return;
    }

    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();

    if (!(speed > T(0))) {
        return;
    }

    const Vector3<T> center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    const Vector3<T> axis = relative / speed;

    const Vector3<T> sample_seed = relative + center * T(atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX)
        + Vector3<T>(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    const T u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U1));
    const T u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U2));

    const T cos_chi = T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);

    const T phi = T(2) * static_cast<T>(atlas::pi) * u2;

    const Vector3<T> scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Vector3<T> scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}