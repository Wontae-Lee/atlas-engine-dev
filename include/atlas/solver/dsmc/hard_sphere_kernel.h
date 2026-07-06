#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

/**
 * @file hard_sphere_kernel.h
 * @brief Simplest DSMC binary collision model: rigid, elastic spheres of
 *        constant diameter, isotropically scattered in the
 *        center-of-mass frame.
 *
 * @details
 * ### Background
 * DSMC (Direct Simulation Monte Carlo, G. A. Bird) replaces the
 * intractable exact molecular dynamics of a real gas with a
 * statistically-representative set of simulator particles, each standing
 * in for a large number of real molecules. Rather than integrating exact
 * intermolecular force fields, DSMC probabilistically selects candidate
 * collision pairs within a cell each timestep (see
 * `DsmcSolver`/`DsmcKernel::sigma_g`, the no-time-counter selection
 * scheme) and, for each selected pair, applies a *binary collision model*
 * — a simplified molecular interaction law that determines the collision
 * cross-section (how "big" a target the pair presents, controlling
 * selection probability) and the post-collision velocity direction. This
 * file, `variable_hard_sphere_kernel.h`, and `variable_soft_sphere_kernel.h`
 * implement the three collision models most commonly used in DSMC
 * codes — HS, VHS, VSS — of increasing physical fidelity and cost; see
 * `dsmc_kernel.h` for how a solver selects between them at runtime.
 *
 * `HardSphereKernel` is the original, simplest model: molecules are
 * treated as rigid spheres of a constant reference diameter `d`, so the
 * total collision cross-section `sigma = pi * d^2` never depends on
 * their relative speed. Every elastic hard-sphere collision, regardless
 * of impact parameter, is assumed a priori equally likely to scatter
 * into any solid angle in the center-of-mass frame — i.e. the
 * post-collision relative-velocity direction is *isotropic*. This is
 * the crudest of the three models (it does not reproduce the
 * viscosity-temperature dependence of real gases, corrected by
 * `VariableHardSphereKernel`), but is cheap and still widely used for
 * qualitative/pedagogical DSMC.
 *
 * ### Operating principle — elastic binary collision kinematics
 * `operator()` implements the standard elastic two-body collision update
 * shared by all three kernels in this family:
 * 1. Center-of-mass velocity `v_cm = (m1 v1 + m2 v2) / (m1 + m2)` is
 *    invariant under an elastic collision (momentum conservation) and is
 *    computed once up front.
 * 2. The relative velocity `g = v1 - v2` has its *magnitude* `|g|`
 *    preserved by an elastic collision (kinetic energy conservation in
 *    the CM frame, where all relative kinetic energy is carried by `g`);
 *    only its *direction* changes, to some new unit vector
 *    `g_hat_scattered`.
 * 3. The new relative velocity is decomposed back into individual
 *    velocities the same way the incident ones combined:
 *    `v1' = v_cm + g_scattered * (m2 / (m1+m2))`,
 *    `v2' = v_cm - g_scattered * (m1 / (m1+m2))` — this is exactly the
 *    inverse of step 1/2, and is easily checked to conserve momentum
 *    (`m1 v1' + m2 v2' = (m1+m2) v_cm = m1 v1 + m2 v2`) and kinetic
 *    energy (`|g_scattered| = |g|` by construction).
 * 4. `HardSphereKernel` draws the new direction isotropically: with two
 *    hashed uniforms `u1, u2 ~ Uniform(0,1)`,
 *    `cos(chi) = 2*u1 - 1 ~ Uniform(-1, 1)` and `phi = 2*pi*u2`. Sampling
 *    `cos(chi)` *uniformly* (not `chi` itself) is what makes the
 *    resulting direction uniform over the *sphere*: the solid-angle
 *    element `dOmega = sin(chi) dchi dphi = -d(cos(chi)) dphi` is
 *    already uniform in `(cos(chi), phi)`, so drawing both coordinates
 *    uniformly reproduces a uniform (isotropic) distribution on the
 *    sphere directly, without needing a separate Jacobian correction.
 *    `atlas::spherical_direction(axis, cos_chi, phi)` builds the
 *    resulting unit vector in a local frame around `axis = g / |g|`.
 *
 * Cross section and scattering are decoupled: `cross_section()` only
 * affects how often a pair is *selected* to collide (see `DsmcSolver`),
 * while `operator()` only determines the outcome *given* a collision was
 * selected.
 *
 * ### References
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994. (canonical DSMC reference;
 *   hard-sphere, VHS, and VSS collision models, and the isotropic
 *   center-of-mass scattering construction, are all from this text)
 */

namespace atlas {

/**
 * @brief Rigid-sphere DSMC collision model: constant cross-section,
 *        isotropic center-of-mass scattering. See this file's
 *        top-of-file documentation for the physical model and the
 *        shared elastic-collision kinematics used by the whole
 *        HS/VHS/VSS family.
 */
class HardSphereKernel final {
public:
    /**
     * @brief Total collision cross-section `pi * d^2`, `d` the mean of
     *        `lhs`/`rhs`'s `reference_diameter`. `0` if either species
     *        lacks a `reference_diameter` or the mean is non-positive.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs) noexcept;

    /**
     * @brief Speed-independent overload for interface parity with
     *        `VariableHardSphereKernel`/`VariableSoftSphereKernel` (whose
     *        cross-sections genuinely depend on `relative_speed`);
     *        `relative_speed` is ignored and this simply forwards to the
     *        two-argument `cross_section`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    /**
     * @brief Applies one elastic hard-sphere collision to `lhs_velocity`/
     *        `rhs_velocity` in place: isotropic center-of-mass
     *        scattering. See this file's top-of-file documentation for
     *        the full kinematic derivation. No-op if either mass is
     *        non-positive or the pair's relative speed is zero (nothing
     *        to scatter).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3& lhs_velocity,
               Vector3& rhs_velocity,
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
HardSphereKernel::operator()(Vector3& lhs_velocity,
                             Vector3& rhs_velocity,
                             const MaterialProperties& lhs,
                             const MaterialProperties& rhs) const noexcept {

    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)) {
        return;
    }

    const Vector3 relative = lhs_velocity - rhs_velocity;
    const float speed      = relative.length();

    if (!(speed > 0.0f)) {
        return;
    }

    const Vector3 center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    const Vector3 axis = relative / speed;

    const Vector3 sample_seed = relative + center * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX
        + Vector3(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    const float u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

    const float u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

    const float cos_chi = 2.0f * u1 - 1.0f;
    const float phi     = 2.0f * atlas::pi * u2;

    const Vector3 scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Vector3 scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}
