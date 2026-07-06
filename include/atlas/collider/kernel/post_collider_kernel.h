#pragma once

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>
#include <atlas/collider/kernel/fast_collider_kernel.h>
#include <atlas/collider/kernel/precise_collider_kernel.h>
#include <atlas/core/detail/device_variant.h>

/**
 * @file post_collider_kernel.h
 * @brief Runtime-selectable dispatcher over the three post-collision
 *        response policies (`FastColliderKernel`, `DtRemainColliderKernel`,
 *        `PreciseColliderKernel`), so a `Collider` can pick a
 *        cost/accuracy tradeoff without templating every call site.
 *
 * @details
 * ### Operating principle
 * `PostColliderKernel` follows the tagged-union `DeviceVariant` pattern
 * used throughout Atlas for device-callable polymorphism without virtual
 * dispatch (see `SurfaceInteractionKernel`, `Geometry`,
 * `Generate` for the same shape): a `PostColliderType` tag plus a
 * `union` holding exactly one of the three kernel payloads at a time.
 * `detail::PostColliderVariant::apply` switches on the tag and forwards
 * the call to the active payload's method, so `sweep_motion`/`operator()`
 * compile to a single branch instead of a virtual call — required because
 * device code cannot resolve virtual dispatch without `-rdc` (see
 * `docs/architecture/04-backend-portability.md`). See each payload
 * kernel's own file for what `fast`/`dt_remain`/`precise` actually do
 * differently.
 */

namespace atlas {

/**
 * @brief Selects which post-collision response `PostColliderKernel`
 *        applies; see `fast_collider_kernel.h`,
 *        `dt_remain_collider_kernel.h`, `precise_collider_kernel.h` for
 *        what each option models and its cost/accuracy tradeoff.
 */
enum struct PostColliderType : int {

    /** Snap to the hit point, discard the leftover timestep. Cheapest. */
    fast,

    /** Snap to the hit point, then ballistically advect through the
     *  leftover timestep at the post-collision velocity. */
    dt_remain,

    /** Sweep in the wall-relative frame (correct for moving/rotating
     *  units), then apply the `dt_remain` post-hit response. Most
     *  accurate, most expensive. */
    precise
};

/**
 * @brief Tagged-union wrapper that lets `ColliderCollisionKernel` call
 *        `sweep_motion`/`operator()` on whichever `PostColliderType` a
 *        `Collider` was configured with, without virtual dispatch.
 *
 * See this file's top-of-file documentation for the `DeviceVariant`
 * pattern this follows.
 */
struct PostColliderKernel final {

    PostColliderType type = PostColliderType::fast;

    union {
        FastColliderKernel fast;
        DtRemainColliderKernel dt_remain;
        PreciseColliderKernel precise;
    };

    ATLAS_ALL_DEVICE
    PostColliderKernel() noexcept;

    ATLAS_ALL_DEVICE explicit PostColliderKernel(PostColliderType type) noexcept;

    ATLAS_ALL_DEVICE
    PostColliderKernel(const PostColliderKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE PostColliderKernel&
    operator=(const PostColliderKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~PostColliderKernel() noexcept = default;

    /** @brief Dispatches to the active payload's `sweep_motion` (builds
     *  the ray segment used to search for a hit this timestep). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit& unit,
                 const Float3& origin,
                 const Float3& incident,
                 float incident_speed,
                 float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept;

    /** @brief Dispatches to the active payload's post-hit response
     *  (reflects/thermalizes velocity and repositions the particle). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& position,
               Float3& velocity,
               const Float3& incident,
               const Float3& hit_position,
               const Float3& hit_normal,
               float hit_distance,
               float sweep_speed,
               float dt,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept;
};

namespace detail {

    using PostColliderVariant = DeviceVariant<
        PostColliderKernel,
        PostColliderType,
        PostColliderType::fast,
        DeviceVariantCase<PostColliderType::fast, &PostColliderKernel::fast>,
        DeviceVariantCase<PostColliderType::dt_remain, &PostColliderKernel::dt_remain>,
        DeviceVariantCase<PostColliderType::precise, &PostColliderKernel::precise>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas);
    // a struct may hold reference members for the write-back outputs.
    struct PostColliderSweepMotion {
        const Unit& unit;
        const Float3& origin;
        const Float3& incident;
        float incident_speed;
        float dt;
        Float3& sweep_direction;
        float& sweep_speed;
        float& sweep_length;
        template <typename K>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        operator()(const K& kernel) const noexcept {
            kernel.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
        }
    };
    struct PostColliderApply {
        Float3& position;
        Float3& velocity;
        const Float3& incident;
        const Float3& hit_position;
        const Float3& hit_normal;
        float hit_distance;
        float sweep_speed;
        float dt;
        const Unit& unit;
        const SurfaceInteractionKernel& interaction;
        template <typename K>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        operator()(const K& kernel) const noexcept {
            kernel(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
        }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
PostColliderKernel::PostColliderKernel() noexcept {
    detail::PostColliderVariant::construct(*this, PostColliderType::fast);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
PostColliderKernel::PostColliderKernel(const PostColliderType type_) noexcept {
    detail::PostColliderVariant::construct(*this, type_);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
PostColliderKernel::sweep_motion(const Unit& unit,
                                 const Float3& origin,
                                 const Float3& incident,
                                 const float incident_speed,
                                 const float dt,
                                 Float3& sweep_direction,
                                 float& sweep_speed,
                                 float& sweep_length) const noexcept {
    detail::PostColliderVariant::apply(
        *this,
        detail::PostColliderSweepMotion {
            unit,
            origin,
            incident,
            incident_speed,
            dt,
            sweep_direction,
            sweep_speed,
            sweep_length });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
PostColliderKernel::operator()(Float3& position,
                               Float3& velocity,
                               const Float3& incident,
                               const Float3& hit_position,
                               const Float3& hit_normal,
                               const float hit_distance,
                               const float sweep_speed,
                               const float dt,
                               const Unit& unit,
                               const SurfaceInteractionKernel& interaction) const noexcept {
    detail::PostColliderVariant::apply(
        *this,
        detail::PostColliderApply {
            position,
            velocity,
            incident,
            hit_position,
            hit_normal,
            hit_distance,
            sweep_speed,
            dt,
            unit,
            interaction });
}

}
