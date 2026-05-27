#pragma once

#include <atlas/math/constants.h>

#include <cmath>
#include <numbers>

namespace atlas::system {

template <typename T>
T
SpartaVssKernel<T>::collision_frequency_factor(const MaterialProperties<T>& lhs,
                                               const MaterialProperties<T>& rhs) noexcept {
    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()
        || !lhs.reference_temperature.has_value() || !rhs.reference_temperature.has_value()) {
        return T(0);
    }

    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return T(0);
    }

    const T reference_diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
    const T reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * T(0.5);
    const T viscosity_index = (lhs.viscosity_index.value_or(T(0.5)) + rhs.viscosity_index.value_or(T(0.5))) * T(0.5);

    if (!(reference_diameter > T(0)) || !(reference_temperature > T(0))) {
        return T(0);
    }

    const T gamma_argument = T(2.5) - viscosity_index;
    if (!(gamma_argument > T(0))) {
        return T(0);
    }

    const T reduced_mass = lhs_mass * rhs_mass / mass_sum;
    const T thermal_factor = (T(2) * static_cast<T>(atlas::boltzmann_constant) * reference_temperature) / reduced_mass;
    if (!(thermal_factor > T(0))) {
        return T(0);
    }

    const T cross_section = static_cast<T>(std::numbers::pi_v<double>) * reference_diameter * reference_diameter;
    const T gamma_value = static_cast<T>(std::tgamma(static_cast<double>(gamma_argument)));
    if (!(gamma_value > T(0))) {
        return T(0);
    }

    return cross_section * std::pow(thermal_factor, viscosity_index - T(0.5)) / gamma_value;
}

template <typename T>
T
SpartaVssKernel<T>::collision_frequency(const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs,
                                        const T relative_speed_squared) noexcept {
    const T viscosity_index = (lhs.viscosity_index.value_or(T(0.5)) + rhs.viscosity_index.value_or(T(0.5))) * T(0.5);
    if (!(relative_speed_squared > T(0))) {
        return T(0);
    }

    if (relative_speed_squared < T(1.0e-30) && viscosity_index >= T(1)) {
        return T(0);
    }

    const T factor = collision_frequency_factor(lhs, rhs);
    if (!(factor > T(0))) {
        return T(0);
    }

    return factor * std::pow(relative_speed_squared, T(1) - viscosity_index);
}

template <typename T>
T
SpartaVssKernel<T>::collision_frequency(const MaterialProperties<T>* properties_ptr,
                                        const std::size_t species_i,
                                        const std::size_t species_j,
                                        const T relative_speed_squared) const noexcept {
    return SpartaVssKernel<T>::collision_frequency(
        properties_ptr[species_i],
        properties_ptr[species_j],
        relative_speed_squared);
}

template <typename T>
void
SpartaVssKernel<T>::operator()(Vector3<T>& lhs_velocity,
                               Vector3<T>& rhs_velocity,
                               const MaterialProperties<T>& lhs,
                               const MaterialProperties<T>& rhs) const noexcept {
    VariableSoftSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system
