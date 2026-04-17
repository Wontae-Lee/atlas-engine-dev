#pragma once

/**
 * @file hard_sphere_kernel.h
 * @brief Declares a simple hard-sphere collision kernel for DSMC solvers.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

template <typename T>
class HardSphereKernel final {
public:
    /**
     * @brief Computes an effective hard-sphere collision cross section.
     *
     * @param lhs Left species material properties.
     * @param rhs Right species material properties.
     * @return Effective cross section. Returns zero when the diameter data is unavailable.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MatrialProperties<T>& lhs,
                  const MatrialProperties<T>& rhs) noexcept;

    /**
     * @brief Applies a simple elastic hard-sphere collision to two particle velocities.
     *
     * @param lhs_velocity Velocity of the first particle.
     * @param rhs_velocity Velocity of the second particle.
     * @param lhs Left species material properties.
     * @param rhs Right species material properties.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MatrialProperties<T>& lhs,
               const MatrialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/hard_sphere_kernel.hpp>
