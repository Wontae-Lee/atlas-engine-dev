#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>

namespace atlas {

/**
 * @brief Variable Soft Sphere (VSS) kernel: the VHS cross section with anisotropic scatter.
 *
 * VSS keeps the same speed-dependent cross section as @ref VariableHardSphereKernel but
 * replaces isotropic scattering with the soft-sphere angular law parameterized by each
 * species' scattering parameter alpha, letting a model match both viscosity and diffusion
 * coefficients. Stateless and trivially copyable so it fits inside @ref DsmcKernel's union.
 */
class VariableSoftSphereKernel final {
public:
    /**
     * @brief VSS cross section — identical to the VHS cross section.
     *
     * VSS and VHS differ only in the scattering angle, not in the total cross section, so this
     * simply forwards to @ref VariableHardSphereKernel::cross_section.
     *
     * @param lhs           First partner's material.
     * @param rhs           Second partner's material.
     * @param relative_speed Magnitude of the relative velocity (m/s).
     * @return The VHS cross section in m^2.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs,
                  const Material& rhs,
                  const float relative_speed) noexcept {
        return VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed);
    }

    /**
     * @brief Scatters the pair anisotropically via @ref dsmc_scatter with the pair's alpha.
     *
     * Uses the mean of the two species' scattering parameters as the VSS exponent; a mean of
     * `1` reproduces isotropic (VHS) scatter.
     *
     * @param lhs_velocity First partner's velocity (m/s); updated in place.
     * @param rhs_velocity Second partner's velocity (m/s); updated in place.
     * @param lhs          First partner's material, queried for its scattering parameter.
     * @param rhs          Second partner's material, queried for its scattering parameter.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs) const noexcept {
        const float scattering_parameter = (lhs.scattering_parameter() + rhs.scattering_parameter()) * 0.5f;

        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, scattering_parameter);
    }
};

}