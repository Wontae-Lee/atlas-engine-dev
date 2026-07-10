#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>

namespace atlas {

/**
 * @brief Hard-sphere collision kernel: a speed-independent cross section, isotropic scatter.
 *
 * The simplest DSMC leaf. Models each species as a rigid sphere whose diameter never
 * changes, so the pair cross section is `pi * d^2` with `d` the arithmetic mean of the two
 * reference diameters, independent of the collision speed. Scattering is isotropic. Stateless
 * and trivially copyable, so it lives inside @ref DsmcKernel's device-capturable union.
 */
class HardSphereKernel final {
public:
    /**
     * @brief Speed-independent hard-sphere cross section for a material pair.
     *
     * Uses the mean of the two reference diameters; the relative-speed argument is ignored,
     * which is the defining property of the hard-sphere model.
     *
     * @param lhs First partner's material.
     * @param rhs Second partner's material.
     * @return `pi * d^2` in m^2 with `d` the mean reference diameter, or `0` if that mean is
     *         non-positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs, const Material& rhs, const float) noexcept {
        const float diameter = (lhs.reference_diameter() + rhs.reference_diameter()) * 0.5f;

        if (!(diameter > 0.0f)) {
            return 0.0f;
        }

        return atlas::pi * diameter * diameter;
    }

    /**
     * @brief Scatters the pair isotropically via @ref dsmc_scatter with `alpha = 1`.
     * @param lhs_velocity First partner's velocity (m/s); updated in place.
     * @param rhs_velocity Second partner's velocity (m/s); updated in place.
     * @param lhs          First partner's material.
     * @param rhs          Second partner's material.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs) const noexcept {
        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f);
    }
};

}