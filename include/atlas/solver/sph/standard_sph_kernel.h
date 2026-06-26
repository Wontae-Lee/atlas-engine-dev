#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas {

template <typename T>
struct StandardSphKernel final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(T radius, T cell_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T cell_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(T radius, T cell_size) noexcept;
};

}

#include <atlas/solver/sph/standard_sph_kernel.hpp>