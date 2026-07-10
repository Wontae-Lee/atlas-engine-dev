#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>

#include <cmath>

namespace atlas {

/**
 * @brief Variable Hard Sphere (VHS) kernel: speed-dependent cross section, isotropic scatter.
 *
 * Refines the hard-sphere model so the cross section falls off with relative speed as a power
 * law set by the viscosity index omega, reproducing a realistic temperature-dependent
 * viscosity while keeping scattering isotropic (same @ref dsmc_scatter with `alpha = 1` as
 * @ref HardSphereKernel). Every pair property is the arithmetic mean of the two species'
 * values. Stateless and trivially copyable so it fits inside @ref DsmcKernel's union.
 */
class VariableHardSphereKernel final {
public:
    /**
     * @brief Speed-dependent VHS cross section for a material pair.
     *
     * Evaluates the VHS law
     * `sigma = pi d_ref^2 * (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega)`,
     * with `d_ref`, `T_ref`, and the viscosity index `omega` taken as pair means, `m_r` the
     * reduced mass, `g` the relative speed, and `k` Boltzmann's constant. Each intermediate is
     * guarded so a degenerate species (zero mass, diameter, temperature, or an out-of-range
     * viscosity index making the Gamma argument non-positive) yields a `0` cross section
     * rather than a NaN.
     *
     * @warning The reduced mass is evaluated as `m_l * (m_r / (m_l + m_r))`, never as
     *          `m_l * m_r / (m_l + m_r)`. A molecular mass is around `1e-26` kg, so the
     *          product of two of them underflows float to zero and the whole expression
     *          returns `inf` for every relative speed. `m_r * g^2` is guarded for the same
     *          reason: it underflows once `g` drops below roughly `1e-10` m/s.
     *
     * @param lhs           First partner's material.
     * @param rhs           Second partner's material.
     * @param relative_speed Magnitude of the relative velocity (m/s); must be positive.
     * @return Cross section in m^2, or `0` for any degenerate or non-positive input.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs,
                  const Material& rhs,
                  const float relative_speed) noexcept {
        const float lhs_mass = lhs.mass();
        const float rhs_mass = rhs.mass();
        const float mass_sum = lhs_mass + rhs_mass;

        // A massless species or zero relative speed makes the reduced mass or the thermal ratio
        // undefined; reject the pair.
        if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
            || !(relative_speed > 0.0f)) {
            return 0.0f;
        }

        const float reference_diameter    = (lhs.reference_diameter() + rhs.reference_diameter()) * 0.5f;
        const float reference_temperature = (lhs.reference_temperature() + rhs.reference_temperature()) * 0.5f;
        const float viscosity_index       = (lhs.viscosity_index() + rhs.viscosity_index()) * 0.5f;

        if (!(reference_diameter > 0.0f) || !(reference_temperature > 0.0f)) {
            return 0.0f;
        }

        // The VHS normalization carries Gamma(5/2 - omega); its argument must stay positive.
        const float gamma_argument = 2.5f - viscosity_index;

        if (!(gamma_argument > 0.0f)) {
            return 0.0f;
        }

        // Divide before multiplying. A molecular mass is ~1e-26 kg, so `lhs_mass * rhs_mass`
        // is ~1e-52 and flushes to zero in float long before the division can rescale it.
        const float reduced_mass = lhs_mass * (rhs_mass / mass_sum);

        const float thermal_energy = reduced_mass * relative_speed * relative_speed;

        // Still zero for a relative speed small enough to underflow the square; the pair
        // contributes no cross-section either way, since sigma * g tends to zero with g.
        if (!(thermal_energy > 0.0f)) {
            return 0.0f;
        }

        const float thermal_ratio
            = (2.0f * atlas::boltzmann_constant * reference_temperature) / thermal_energy;

        // tgamma is only available in double precision; evaluate there and narrow back.
        const float gamma_value = static_cast<float>(std::tgamma(static_cast<double>(gamma_argument)));

        if (!(gamma_value > 0.0f)) {
            return 0.0f;
        }

        const float reference_area = atlas::pi * reference_diameter * reference_diameter;

        return reference_area * std::pow(thermal_ratio, viscosity_index - 0.5f) / gamma_value;
    }

    /**
     * @brief Scatters the pair isotropically via @ref dsmc_scatter with `alpha = 1`.
     *
     * VHS shares the isotropic angular law with hard spheres; only its cross section differs.
     *
     * @param lhs_velocity First partner's velocity (m/s); updated in place.
     * @param rhs_velocity Second partner's velocity (m/s); updated in place.
     * @param lhs          First partner's material.
     * @param rhs          Second partner's material.
     * @param engine       Generator supplying the scatter variates; advanced twice.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs,
               default_random_engine& engine) const noexcept {
        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);
    }
};

}