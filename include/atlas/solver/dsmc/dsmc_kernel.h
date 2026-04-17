#pragma once

/**
 * @file dsmc_kernel.h
 * @brief Declares tagged DSMC collision kernels for device-side runtime dispatch.
 */

#include <atlas/solver/dsmc/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

namespace atlas::system {

enum struct DsmcKernelType : int {
    hard_sphere,
    variable_hard_sphere,
    variable_soft_sphere
};

template <typename T>
struct DsmcKernel final {
    DsmcKernelType type = DsmcKernelType::hard_sphere;

    union {
        HardSphereKernel<T> hard_sphere;
        VariableHardSphereKernel<T> variable_hard_sphere;
        VariableSoftSphereKernel<T> variable_soft_sphere;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DsmcKernel(DsmcKernelType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel(const DsmcKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DsmcKernel() noexcept;

    ATLAS_HOST
    DsmcKernel(const HardSphereKernel<T>& op);

    ATLAS_HOST
    DsmcKernel(const VariableHardSphereKernel<T>& op);

    ATLAS_HOST
    DsmcKernel(const VariableSoftSphereKernel<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(DsmcKernelType type,
                  const MatrialProperties<T>& lhs,
                  const MatrialProperties<T>& rhs,
                  T relative_speed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MatrialProperties<T>& lhs,
               const MatrialProperties<T>& rhs) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DsmcKernel& other) noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/dsmc_kernel.hpp>
