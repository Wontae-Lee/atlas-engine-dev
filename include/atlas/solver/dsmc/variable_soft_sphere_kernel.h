#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <cmath>

/**
 * @file variable_soft_sphere_kernel.h
 * @brief DSMC collision model that reuses `VariableHardSphereKernel`'s
 *        cross-section but replaces isotropic scattering with an
 *        anisotropic law tunable per material pair, better matching real
 *        (non-hard-sphere) molecular deflection functions.
 *
 * @details
 * ### Background
 * VHS gets the cross-section (and hence transport-coefficient)
 * temperature dependence right but still scatters every collision
 * isotropically, which real intermolecular potentials do not: softer
 * potentials deflect trajectories preferentially in the forward
 * direction (small-angle scattering is more likely than large-angle),
 * an effect isotropic scattering cannot represent. Koura and Matsumoto's
 * VSS model (1991) keeps VHS's cross-section formula exactly (this file
 * delegates `cross_section` straight to
 * `VariableHardSphereKernel::cross_section`) and only changes how the
 * post-collision deflection angle `chi` is sampled, introducing a
 * per-pair *scattering parameter* `alpha` that controls how forward-
 * peaked the scattering is.
 *
 * ### Derivation — anisotropic deflection-angle sampling
 * `HardSphereKernel`/`VariableHardSphereKernel` draw
 * `cos(chi) = 2u - 1 ~ Uniform(-1, 1)` (isotropic; see
 * `hard_sphere_kernel.h`'s derivation). VSS instead draws
 * `cos(chi) = 2 * u^(1/alpha) - 1`, `u ~ Uniform(0, 1)`. This is again
 * inverse-CDF sampling: solving for the CDF `F` this transform
 * implements, `u = ((cos(chi) + 1) / 2)^alpha`, i.e.
 * `F(chi) = ((1 + cos(chi))/2)^alpha`, which is a valid CDF on
 * `chi in [0, pi]` (monotonic, `F(0) = 1`, `F(pi) = 0` — note the
 * convention here has `u` map to `1 - cos(chi)`'s complement, so `chi`
 * decreasing as `u` increases; this only relabels which end is which and
 * does not change the resulting distribution's shape). At `alpha = 1`
 * this reduces exactly to `cos(chi) = 2u - 1`, recovering
 * `HardSphereKernel`/`VariableHardSphereKernel`'s isotropic scattering —
 * consistent with VSS being a strict generalization. For `alpha > 1`,
 * `u^(1/alpha) > u` (since `1/alpha < 1` and `u < 1`), which pushes
 * `cos(chi)` toward `+1`, i.e. `chi` toward `0` — biasing the deflection
 * distribution toward small (forward) scattering angles, matching softer
 * real-molecule potentials. The azimuthal angle `phi = 2*pi*u2` stays
 * uniform (no direction in the plane perpendicular to the relative
 * velocity is preferred), and the elastic-collision kinematics
 * (center-of-mass velocity/relative speed conservation) that turn
 * `(chi, phi)` into new individual velocities are identical to
 * `HardSphereKernel::operator()`.
 *
 * `_scattering_parameter` (`alpha`) is read from
 * `MaterialProperties::scattering_parameter` per species (default `1.0`,
 * i.e. isotropic if unset) and pairwise-averaged, the same mixing
 * convention as VHS's `viscosity_index`/`reference_temperature`.
 *
 * ### References
 * - K. Koura and H. Matsumoto, "Variable soft sphere molecular model for
 *   inverse-power-law or Lennard-Jones potential," Physics of Fluids A
 *   3(10), 1991, pp. 2459-2465. (the VSS scattering-angle model)
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994. (VHS cross-section this model
 *   reuses; see `variable_hard_sphere_kernel.h`)
 */

namespace atlas {

/**
 * @brief VHS cross-section, VSS anisotropic (forward-peaked-tunable)
 *        center-of-mass scattering. See this file's top-of-file
 *        documentation for the deflection-angle derivation.
 */
class VariableSoftSphereKernel final {
public:
    /** @brief Identical to `VariableHardSphereKernel::cross_section`
     *  (VSS reuses the VHS cross-section formula unchanged). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    /**
     * @brief Applies one elastic VSS collision: same center-of-mass
     *        conservation kinematics as `HardSphereKernel`, but samples
     *        the deflection angle from the anisotropic law
     *        `cos(chi) = 2*u1^(1/alpha) - 1`, `alpha` the pairwise-
     *        averaged `MaterialProperties::scattering_parameter`. See
     *        this file's top-of-file documentation for the derivation.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const MaterialProperties& lhs,
               const MaterialProperties& rhs) const noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
VariableSoftSphereKernel::cross_section(const MaterialProperties& lhs,
                                        const MaterialProperties& rhs,
                                        const float relative_speed) noexcept {

    return VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
VariableSoftSphereKernel::operator()(Float3& lhs_velocity,
                                     Float3& rhs_velocity,
                                     const MaterialProperties& lhs,
                                     const MaterialProperties& rhs) const noexcept {

    const float scattering_parameter = (lhs.scattering_parameter.value_or(1.0f)
                                        + rhs.scattering_parameter.value_or(1.0f))
        * 0.5f;

    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
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
