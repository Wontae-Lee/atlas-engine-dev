#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas {

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
dsmc_scatter(Float3& lhs_velocity,
             Float3& rhs_velocity,
             const Material& lhs,
             const Material& rhs,
             const float scattering_parameter) noexcept {
    const float lhs_mass = lhs.mass();
    const float rhs_mass = rhs.mass();
    const float mass_sum = lhs_mass + rhs_mass;

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
        + Float3(lhs_mass, rhs_mass, mass_sum);

    const float u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

    const float u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

    const float cos_chi = (scattering_parameter == 1.0f)
        ? 2.0f * u1 - 1.0f
        : 2.0f * std::pow(u1, 1.0f / scattering_parameter) - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    const Float3 scattered_relative = atlas::spherical_direction(axis, cos_chi, phi) * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}