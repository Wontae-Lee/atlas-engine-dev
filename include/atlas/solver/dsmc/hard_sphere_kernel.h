#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

namespace atlas {

class HardSphereKernel final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs) noexcept;

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
HardSphereKernel::cross_section(const MaterialProperties& lhs,
                                const MaterialProperties& rhs) noexcept {

    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()) {
        return 0.0f;
    }

    const float diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * 0.5f;

    if (!(diameter > 0.0f)) {
        return 0.0f;
    }

    return atlas::pi * diameter * diameter;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
HardSphereKernel::cross_section(const MaterialProperties& lhs,
                                const MaterialProperties& rhs,
                                const float relative_speed) noexcept {
    static_cast<void>(relative_speed);
    return cross_section(lhs, rhs);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
HardSphereKernel::operator()(Float3& lhs_velocity,
                             Float3& rhs_velocity,
                             const MaterialProperties& lhs,
                             const MaterialProperties& rhs) const noexcept {

    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)) {
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

    const float cos_chi = 2.0f * u1 - 1.0f;
    const float phi     = 2.0f * atlas::pi * u2;

    const Float3 scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Float3 scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}
