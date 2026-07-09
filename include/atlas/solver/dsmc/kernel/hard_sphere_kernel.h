#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>
#include <atlas/material/material.h>

namespace atlas {

// A speed-independent cross-section from the pair's mean diameter, and
// isotropic scattering.
class HardSphereKernel final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs, const Material& rhs, const float) noexcept {
        const float diameter = (lhs.reference_diameter() + rhs.reference_diameter()) * 0.5f;

        if (!(diameter > 0.0f)) {
            return 0.0f;
        }

        return atlas::pi * diameter * diameter;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs) const noexcept {
        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f);
    }
};

}
