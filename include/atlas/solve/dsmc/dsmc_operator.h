#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/solve/solve.h>

#include <cstdint>
#include <type_traits>

namespace atlas::system {

enum class DsmcModelType : int {
    hard_sphere,
    vhs,
    vss
};

template <typename T>
struct HardSphereDsmcOperator final {
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit HardSphereDsmcOperator(
        T probability_scale = T(1),
        unsigned int seed   = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_kernel(T effective_collision_diameter,
                     T relative_speed) const noexcept;
};

template <typename T>
struct VhsDsmcOperator final {
    T reference_temperature { T(273.15) };
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit VhsDsmcOperator(
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
struct VssDsmcOperator final {
    T reference_temperature { T(273.15) };
    T probability_scale { T(1) };
    unsigned int seed = 0u;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit VssDsmcOperator(
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
struct DsmcOperator final {
    static_assert(std::is_floating_point_v<T>, "DsmcOperator requires a floating-point T");

    DsmcModelType type = DsmcModelType::hard_sphere;
    union {
        HardSphereDsmcOperator<T> hard_sphere;
        VhsDsmcOperator<T> vhs;
        VssDsmcOperator<T> vss;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcOperator(DsmcModelType type,
                 T param0          = T(273.15),
                 T probability     = T(1),
                 unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DsmcOperator(const DsmcOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcOperator&
    operator=(const DsmcOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DsmcOperator() noexcept;

    ATLAS_HOST
    DsmcOperator(const HardSphereDsmcOperator<T>& op);

    ATLAS_HOST
    DsmcOperator(const VhsDsmcOperator<T>& op);

    ATLAS_HOST
    DsmcOperator(const VssDsmcOperator<T>& op);

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
    copy_from(const DsmcOperator& other) noexcept;
};

}

namespace atlas {

template <typename T>
using DsmcOperator = atlas::system::DsmcOperator<T>;

template <typename T>
using HardSphereDsmcOperator = atlas::system::HardSphereDsmcOperator<T>;

template <typename T>
using VhsDsmcOperator = atlas::system::VhsDsmcOperator<T>;

template <typename T>
using VssDsmcOperator = atlas::system::VssDsmcOperator<T>;

using DsmcModelType = atlas::system::DsmcModelType;

}

#include <atlas/solve/dsmc/dsmc_operator.hpp>
