#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

/**
 * @file variable_hard_sphere_kernel.h
 * @brief DSMC collision model with a temperature-dependent cross-section
 *        that reproduces the correct viscosity-temperature exponent of a
 *        real gas, while keeping `HardSphereKernel`'s isotropic
 *        scattering.
 *
 * @details
 * ### Background
 * Real-gas viscosity scales with temperature roughly as `mu ~ T^omega`,
 * where `omega` (`viscosity_index`) is close to `0.5` for a true rigid
 * hard sphere but empirically higher (`~0.7`-`1.0`) for most real gases —
 * `HardSphereKernel`'s temperature-independent cross-section reproduces
 * only `omega = 0.5` and so gets transport properties (viscosity,
 * diffusion, thermal conductivity) wrong away from the reference
 * condition. The VHS model (Bird) keeps the *scattering law* of a hard
 * sphere (still isotropic, same as `HardSphereKernel::operator()`) but
 * lets the *apparent* sphere diameter — and hence the cross-section —
 * shrink with increasing relative speed, calibrated so the resulting
 * viscosity matches the real gas's `omega` at a reference temperature
 * `T_ref`. This decouples cross-section physics from scattering-angle
 * physics, which `VariableSoftSphereKernel` (VSS) then also
 * generalizes.
 *
 * ### Derivation — the VHS cross-section formula
 * Bird derives the VHS total cross-section by requiring the model's
 * viscosity coefficient, computed from Chapman-Enskog kinetic theory for
 * a power-law cross-section `sigma ~ g^(1 - 2*omega)` (`g` = relative
 * speed), to equal the real gas's viscosity `mu = mu_ref * (T/T_ref)^omega`
 * at the reference temperature. The result (this file's `cross_section`):
 * ```
 * sigma(g) = pi * d_ref^2 * [ (2 k_B T_ref) / (mu_r * g^2) ]^(omega - 1/2)
 *            / Gamma(5/2 - omega)
 * ```
 * where `d_ref` is the mean reference diameter at `T_ref`, `mu_r` the
 * reduced mass `m1 m2 / (m1+m2)`, and `Gamma` the Euler gamma function
 * (`std::tgamma`) — the normalization constant that makes the resulting
 * effective viscosity exponent come out to exactly `omega` at `T_ref`.
 * Two sanity checks: at `omega = 0.5` (a true hard sphere), the exponent
 * `omega - 1/2` vanishes and `sigma` collapses to the constant
 * `pi * d_ref^2 / Gamma(2) = pi * d_ref^2` — exactly
 * `HardSphereKernel`'s cross-section, as it must. And `sigma` decreases
 * with `g` for `omega > 0.5` (the usual case for real gases): faster
 * pairs present a smaller apparent target, which is what raises the
 * effective viscosity exponent above the pure-hard-sphere value of
 * `0.5`.
 *
 * `viscosity_index`/`reference_temperature`/`reference_diameter` are
 * averaged pairwise (arithmetic mean of the two species' values) — the
 * conventional VHS mixing rule for unlike-species pairs.
 *
 * ### References
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994, ch. 4 (VHS model derivation
 *   and cross-section formula).
 */

namespace atlas {

/**
 * @brief Temperature-dependent (power-law) cross-section, isotropic
 *        center-of-mass scattering (same as `HardSphereKernel`). See
 *        this file's top-of-file documentation for the cross-section
 *        derivation.
 */
class VariableHardSphereKernel final {
public:
    /**
     * @brief VHS total cross-section at relative speed `relative_speed`;
     *        see this file's Derivation section. Returns `0` if either
     *        species is missing `reference_diameter`/
     *        `reference_temperature`, if either mass/the mean diameter/
     *        the mean temperature is non-positive, if
     *        `relative_speed <= 0`, if `omega >= 2.5` (the Gamma-function
     *        argument `2.5 - omega` would be non-positive), or if the
     *        Gamma evaluation itself is non-positive.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    cross_section(const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    /**
     * @brief Applies one elastic VHS collision: identical isotropic
     *        center-of-mass scattering kinematics to
     *        `HardSphereKernel::operator()` (see that file's
     *        derivation) — only the cross-section (and hence selection
     *        probability, not the outcome here) differs between the two
     *        models. The local `scattering_parameter = 1` fixes the
     *        scattering law to isotropic; compare
     *        `VariableSoftSphereKernel`, which instead reads a
     *        per-material scattering parameter to bias it.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3& lhs_velocity,
               Vector3& rhs_velocity,
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
VariableHardSphereKernel::operator()(Vector3& lhs_velocity,
                                     Vector3& rhs_velocity,
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

    const float cos_chi = 2.0f * std::pow(u1, 1.0f / scattering_parameter) - 1.0f;

    const float phi = 2.0f * atlas::pi * u2;

    const Vector3 scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Vector3 scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}
