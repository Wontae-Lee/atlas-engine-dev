#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas {

class VariableHardSphereKernel final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const MaterialProperties& lhs,
               const MaterialProperties& rhs) const noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
VariableHardSphereKernel::cross_section(const MaterialProperties& lhs,
                                        const MaterialProperties& rhs,
                                        const float relative_speed) noexcept {

    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()
        || !lhs.reference_temperature.has_value() || !rhs.reference_temperature.has_value()) {
        return 0.0f;
    }

    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
        || !(relative_speed > 0.0f)) {
        return 0.0f;
    }

    const float reference_diameter    = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * 0.5f;
    const float reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * 0.5f;
    const float viscosity_index       = (lhs.viscosity_index.value_or(0.5f) + rhs.viscosity_index.value_or(0.5f))
        * 0.5f;

    if (!(reference_diameter > 0.0f) || !(reference_temperature > 0.0f)) {
        return 0.0f;
    }

    const float gamma_argument = 2.5f - viscosity_index;
    if (!(gamma_argument > 0.0f)) {
        return 0.0f;
    }

    const float reduced_mass = lhs_mass * rhs_mass / mass_sum;

    const float thermal_ratio = (2.0f * atlas::boltzmann_constant * reference_temperature)
        / (reduced_mass * relative_speed * relative_speed);

    if (!(thermal_ratio > 0.0f)) {
        return 0.0f;
    }

    const float reference_area = atlas::pi * reference_diameter * reference_diameter;

    const float gamma_value = static_cast<float>(std::tgamma(static_cast<double>(gamma_argument)));
    if (!(gamma_value > 0.0f)) {
        return 0.0f;
    }

    return reference_area * std::pow(thermal_ratio, viscosity_index - 0.5f) / gamma_value;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
VariableHardSphereKernel::operator()(Float3& lhs_velocity,
                                     Float3& rhs_velocity,
                                     const MaterialProperties& lhs,
                                     const MaterialProperties& rhs) const noexcept {

    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;

    const float scattering_parameter = 1.0f;

    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
        || !(scattering_parameter > 0.0f)) {
        return;
    }

    const Float3 relative = lhs_velocity - rhs_velocity;
    const float speed     = relative.length();

    if (!(speed > 0.0f)) {
        return;
    }

    const Float3 center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    const Float3 axis = relative / speed;

    const Float3 sample_seed = relative + center * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX
        + Float3(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    const float u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U1);
    const float u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

    const float cos_chi = 2.0f * std::pow(u1, 1.0f / scattering_parameter) - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    const Float3 scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Float3 scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}
