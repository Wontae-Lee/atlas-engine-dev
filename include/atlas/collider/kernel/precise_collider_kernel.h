#pragma once

/**
 * @file precise_collider_kernel.h
 * @brief Declares the moving-surface-aware post-collider placement kernel.
 */

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>

namespace atlas {

template <typename T>
class PreciseColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PreciseColliderKernel() noexcept = default;

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
               const SurfaceInteractionKernel<T>& interaction) const noexcept;
};

} // namespace atlas

namespace atlas {
} // namespace atlas

#include <atlas/collider/kernel/precise_collider_kernel.hpp>
