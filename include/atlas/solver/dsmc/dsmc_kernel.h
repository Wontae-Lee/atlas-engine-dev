#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <cstdint>
#include <type_traits>

namespace atlas::system {

enum class DsmcModelType : int {
    hs,
    vhs,
    vss
};

template <typename T>
struct DsmcHsKernel final {
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DsmcHsKernel(
        T probability_scale = T(1),
        unsigned int seed   = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_kernel(T effective_collision_diameter,
                     T relative_speed) const noexcept;
};

template <typename T>
struct DsmcVhsKernel final {
    T reference_temperature { T(273.15) };
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DsmcVhsKernel(
        T reference_temperature = T(273.15),
        T probability_scale     = T(1),
        unsigned int seed       = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_kernel(T effective_collision_diameter,
                     T reduced_mass,
                     T effective_viscosity_index,
                     T local_temperature,
                     T relative_speed) const noexcept;
};

template <typename T>
struct DsmcVssKernel final {
    T reference_temperature { T(273.15) };
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DsmcVssKernel(
        T reference_temperature = T(273.15),
        T probability_scale     = T(1),
        unsigned int seed       = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_kernel(T effective_collision_diameter,
                     T reduced_mass,
                     T effective_viscosity_index,
                     T effective_scattering_parameter,
                     T local_temperature,
                     T relative_speed) const noexcept;
};

template <typename T>
struct DsmcKernel final {
    static_assert(std::is_floating_point_v<T>, "DsmcKernel requires a floating-point T");

    DsmcModelType type = DsmcModelType::hs;
    union {
        DsmcHsKernel<T> hard_sphere;
        DsmcVhsKernel<T> vhs;
        DsmcVssKernel<T> vss;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel(DsmcModelType type,
               T param0          = T(273.15),
               T probability     = T(1),
               unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcKernel(const DsmcKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DsmcKernel() noexcept;

    ATLAS_HOST
    DsmcKernel(const DsmcHsKernel<T>& op);

    ATLAS_HOST
    DsmcKernel(const DsmcVhsKernel<T>& op);

    ATLAS_HOST
    DsmcKernel(const DsmcVssKernel<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    should_collide(T effective_collision_diameter,
                   T reduced_mass,
                   T effective_viscosity_index,
                   T effective_scattering_parameter,
                   T local_temperature,
                   const Vector3<T>& relative_velocity,
                   std::uint64_t pair_id) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    scatter_relative_velocity(T effective_scattering_parameter,
                              const Vector3<T>& relative_velocity,
                              std::uint64_t pair_id) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_kernel(T effective_collision_diameter,
                     T reduced_mass,
                     T effective_viscosity_index,
                     T effective_scattering_parameter,
                     T local_temperature,
                     T relative_speed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE unsigned int
    base_seed() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DsmcKernel& other) noexcept;
};

}

namespace atlas {

template <typename T>
using DsmcOperator = atlas::system::DsmcKernel<T>;

template <typename T>
using HardSphereDsmcOperator = atlas::system::DsmcHsKernel<T>;

template <typename T>
using VhsDsmcOperator = atlas::system::DsmcVhsKernel<T>;

template <typename T>
using VssDsmcOperator = atlas::system::DsmcVssKernel<T>;

using DsmcModelType = atlas::system::DsmcModelType;

}

#include <atlas/solver/dsmc/dsmc_kernel.hpp>