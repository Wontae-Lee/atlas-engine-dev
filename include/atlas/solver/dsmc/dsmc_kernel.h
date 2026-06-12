#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/dsmc/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <cstdint>

namespace atlas {

enum struct DsmcKernelType : int {
    hard_sphere,
    variable_hard_sphere,
    variable_soft_sphere
};

template <typename T>
struct DsmcPairParameters final {
    T reference_diameter {};
    T reference_temperature {};
    T viscosity_index { T(0.5) };
    T scattering_parameter { T(1) };
    T reduced_mass {};
    bool valid {};
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
                  const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static DsmcPairParameters<T>
    pair_parameters(const MaterialProperties<T>& lhs,
                    const MaterialProperties<T>& rhs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sigma_g(const MaterialProperties<T>* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            T relative_speed_squared) const noexcept;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DsmcKernel& other) noexcept;
};

}

#include <atlas/solver/dsmc/dsmc_kernel.hpp>