#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/sph/cubic_spline_sph_kernel.h>
#include <atlas/solver/sph/standard_sph_kernel.h>
#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

#include <type_traits>

namespace atlas {

enum struct SphKernelType : int {
    standard,
    cubic_spline,
    wendland_quintic
};

template <typename T>
struct SphKernel final {

    SphKernelType type = SphKernelType::standard;

    union {

        StandardSphKernel<T> standard;

        CubicSplineSphKernel<T> cubic_spline;

        WendlandQuinticSphKernel<T> wendland_quintic;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SphKernel(SphKernelType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SphKernel(const SphKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SphKernel&
    operator=(const SphKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SphKernel() noexcept;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel>, int> = 0>
    ATLAS_HOST SphKernel(const Payload& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(SphKernelType type,
                   T radius,
                   T cell_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(SphKernelType type,
                      const Vector3<T>& delta,
                      T radius,
                      T cell_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(SphKernelType type,
                        T radius,
                        T cell_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    density_weight(T radius, T cell_size) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T cell_size) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    viscosity_laplacian(T radius, T cell_size) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SphKernel& other) noexcept;
};

}

#include <atlas/solver/sph/sph_kernel.hpp>