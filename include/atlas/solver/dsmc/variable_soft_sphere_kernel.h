#pragma once

/**
 * @file variable_soft_sphere_kernel.h
 * @brief Declares a variable-soft-sphere collision kernel for DSMC solvers.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

template <typename T>
class VariableSoftSphereKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MatrialProperties<T>& lhs,
                  const MatrialProperties<T>& rhs,
                  T relative_speed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MatrialProperties<T>& lhs,
               const MatrialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_soft_sphere_kernel.hpp>
