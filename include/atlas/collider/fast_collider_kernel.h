#pragma once

/**
 * @file fast_collider_kernel.h
 * @brief Declares the fast post-collider placement kernel.
 */

#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/math/constants.h>
#include <atlas/unit/unit.h>

namespace atlas::system {

template <typename T>
class FastColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    FastColliderKernel() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    surface_velocity(const Unit<T>& unit,
                     const Vector3<T>& surface_point) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit<T>& unit,
                 const Vector3<T>& origin,
                 const Vector3<T>& incident,
                 T incident_speed,
                 T dt,
                 Vector3<T>& sweep_direction,
                 T& sweep_speed,
                 T& sweep_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& position,
               Vector3<T>& velocity,
               const Vector3<T>& incident,
               const Vector3<T>& hit_position,
               const Vector3<T>& hit_normal,
               T hit_distance,
               T sweep_speed,
               T dt,
               const Unit<T>& unit,
               const ColliderSurfaceInteraction<T>& interaction) const noexcept;
};

} // namespace atlas::system

#include <atlas/collider/fast_collider_kernel.hpp>
