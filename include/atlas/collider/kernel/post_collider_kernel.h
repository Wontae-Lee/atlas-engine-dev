#pragma once

/**
 * @file post_collider_kernel.h
 * @brief Declares tagged post-collision kernels for Collider.
 */

#include <atlas/core/detail/device_variant.h>
#include <atlas/collider/kernel/dt_remain_collider_kernel.h>
#include <atlas/collider/kernel/fast_collider_kernel.h>
#include <atlas/collider/kernel/precise_collider_kernel.h>

namespace atlas {

/**
 * @brief Selects how Collider places a particle after a surface hit.
 */
enum struct PostColliderType : int {
    /**
     * @brief Stop at the collision point and only update velocity.
     */
    fast,

    /**
     * @brief Move for the remaining time step after applying collision response.
     */
    dt_remain,

    /**
     * @brief Use moving-surface relative motion for hit timing, then move remaining time.
     */
    precise
};

/**
 * @brief Tagged runtime wrapper for post-collision particle placement.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct PostColliderKernel final {
    /**
     * @brief Type tag selecting the active post-collision behavior.
     */
    PostColliderType type = PostColliderType::fast;

    union {
        FastColliderKernel<T> fast;
        DtRemainColliderKernel<T> dt_remain;
        PreciseColliderKernel<T> precise;
    };

    /**
     * @brief Construct a default post-collision kernel.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PostColliderKernel() noexcept;

    /**
     * @brief Construct a post-collision kernel from a type tag.
     *
     * @param type Active post-collision behavior.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit
    PostColliderKernel(PostColliderType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PostColliderKernel(const PostColliderKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE PostColliderKernel&
    operator=(const PostColliderKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    ~PostColliderKernel() noexcept;

    /**
     * @brief Computes the swept motion used for hit testing.
     *
     * @param unit Collider unit being tested.
     * @param origin Particle position at the start of the step.
     * @param incident Particle velocity at the start of the step.
     * @param incident_speed Precomputed particle speed.
     * @param dt Simulation time step.
     * @param sweep_direction Output swept displacement direction.
     * @param sweep_speed Output speed used to convert hit distance to hit time.
     * @param sweep_length Output swept segment length.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit<T>& unit,
                 const Vector3<T>& origin,
                 const Vector3<T>& incident,
                 T incident_speed,
                 T dt,
                 Vector3<T>& sweep_direction,
                 T& sweep_speed,
                 T& sweep_length) const noexcept;

    /**
     * @brief Applies the active post-collision behavior in place.
     *
     * @param position Particle position to update in place.
     * @param velocity Particle velocity to update in place.
     * @param incident Particle velocity before collision.
     * @param hit_position World-space hit position.
     * @param hit_normal World-space collision normal.
     * @param hit_distance Swept distance traveled before collision.
     * @param sweep_speed Speed used to convert hit distance to hit time.
     * @param dt Simulation time step.
     * @param unit Collider unit hit by the particle.
     * @param interaction Tagged surface interaction kernel for outgoing velocity.
     */
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

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const PostColliderKernel& other) noexcept;
};

} // namespace atlas

#include <atlas/collider/kernel/post_collider_kernel.hpp>
