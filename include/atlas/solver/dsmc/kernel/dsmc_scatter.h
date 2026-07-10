#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

namespace atlas {

/**
 * @brief Elastically scatters a colliding pair in place, shared by every DSMC kernel leaf.
 *
 * Works in the pair's centre-of-mass frame: the centre-of-mass velocity and the relative
 * speed are conserved, only the *direction* of the relative velocity is redrawn, which
 * automatically conserves both momentum and kinetic energy. The deflection angle chi is
 * drawn from the VSS law `cos(chi) = 2*u1^(1/alpha) - 1`; with `alpha == 1` this collapses
 * to the isotropic hard-sphere / VHS case `cos(chi) = 2*u1 - 1`. The azimuth is uniform.
 *
 * The two uniform variates come from the low-quality sine hash keyed on the pair's own
 * geometry (relative velocity, centre-of-mass, and masses), so the scatter is stateless and
 * reproducible without carrying an RNG per thread. Distinct salts decorrelate the two draws.
 *
 * @param lhs_velocity        First partner's velocity (m/s); overwritten with the post-collision value.
 * @param rhs_velocity        Second partner's velocity (m/s); overwritten with the post-collision value.
 * @param lhs                 First partner's material, queried for its mass.
 * @param rhs                 Second partner's material, queried for its mass.
 * @param scattering_parameter The VSS exponent alpha (1 = isotropic); must be positive.
 *
 * @note A no-op when either mass, the mass sum, the scattering parameter, or the relative
 *       speed is non-positive — a degenerate pair carries no well-defined scatter direction.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
dsmc_scatter(Float3& lhs_velocity,
             Float3& rhs_velocity,
             const Material& lhs,
             const Material& rhs,
             const float scattering_parameter) noexcept {
    const float lhs_mass = lhs.mass();
    const float rhs_mass = rhs.mass();
    const float mass_sum = lhs_mass + rhs_mass;

    // Guard against massless species and a non-positive scattering exponent, either of which
    // makes the centre-of-mass split or the angular law undefined.
    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
        || !(scattering_parameter > 0.0f)) {
        return;
    }

    const Float3 relative = lhs_velocity - rhs_velocity;
    const float speed     = relative.length();

    // No relative motion means no collision to resolve, and no axis to rotate about.
    if (!(speed > 0.0f)) {
        return;
    }

    const Float3 center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    const Float3 axis = relative / speed;

    // Seed the stateless hash from the collision's own geometry and masses so identical pairs
    // reproduce and unrelated pairs decorrelate; the `axis` offset separates the two draws.
    const Float3 sample_seed = relative + center * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX
        + Float3(lhs_mass, rhs_mass, mass_sum);

    const float u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

    const float u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

    // VSS deflection cosine; the alpha == 1 branch skips the pow and is exact isotropic scatter.
    const float cos_chi = (scattering_parameter == 1.0f)
        ? 2.0f * u1 - 1.0f
        : 2.0f * std::pow(u1, 1.0f / scattering_parameter) - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    // Rotate the relative velocity to the sampled direction, preserving its magnitude (energy).
    const Float3 scattered_relative = atlas::spherical_direction(axis, cos_chi, phi) * speed;

    // Redistribute about the conserved centre of mass, weighting each partner by the other's mass.
    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}