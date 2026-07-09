#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>

namespace atlas {

class VariableSoftSphereKernel final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs,
                  const Material& rhs,
                  const float relative_speed) noexcept {
        return VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed);
    }

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