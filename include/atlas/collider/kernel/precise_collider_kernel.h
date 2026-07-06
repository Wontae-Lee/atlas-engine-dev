#pragma once

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>

/**
 * @file precise_collider_kernel.h
 * @brief Most accurate post-collision response
 *        (`atlas::PostColliderType::precise`): finds the impact by
 *        sweeping in the *relative* frame of a possibly moving/rotating
 *        unit, then reuses `DtRemainColliderKernel`'s remaining-time
 *        correction for the response.
 *
 * @details
 * ### Background
 * `FastColliderKernel`/`DtRemainColliderKernel` both search for the hit
 * along the particle's world-space straight-line motion, only bringing
 * the wall's own velocity in afterward (for the reflected-velocity
 * calculation in `operator()`). That is exact for stationary walls but
 * approximate for moving/rotating ones: the true impact point on a
 * moving surface is where the *relative* trajectory (particle motion
 * minus local wall motion) first crosses the surface, not where the
 * particle's absolute path does. `PreciseColliderKernel` is the
 * correction: it subtracts the wall's rigid-body surface velocity at the
 * sweep's *origin* (`FastColliderKernel::surface_velocity`) from the
 * incident velocity before building the sweep segment, so
 * `ColliderCollisionKernel::closest_hit` ray-traces the relative
 * trajectory instead of the absolute one. This is a first-order
 * approximation in `dt` (the wall velocity is evaluated once at the
 * sweep origin, not re-evaluated along the sweep or at the actual impact
 * point), which is accurate as long as the wall's linear/angular velocity
 * does not change much — and the wall doesn't move far — over one
 * timestep.
 */

namespace atlas {

/**
 * @brief `PostColliderType::precise`: sweeps in the particle-relative-
 *        to-wall frame to locate the hit (correct for moving/rotating
 *        units), then delegates the post-hit remaining-time propagation
 *        to `DtRemainColliderKernel`. See this file's top-of-file
 *        documentation for why the relative-frame sweep is needed and
 *        its first-order-in-`dt` accuracy caveat.
 */
class PreciseColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PreciseColliderKernel() noexcept = default;

    /**
     * @brief Builds the sweep segment from the velocity of the particle
     *        *relative to the wall's local surface velocity at the sweep
     *        origin* (`incident - surface_velocity(unit, origin)`),
     *        rather than the particle's absolute velocity — so a
     *        particle sitting still relative to a moving wall correctly
     *        produces a zero-length (no-hit) sweep instead of a spurious
     *        hit from the wall's own motion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit& unit,
                 const Float3& origin,
                 const Float3& incident,
                 const float,
                 const float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept {
        const Float3 relative_velocity = incident - FastColliderKernel::surface_velocity(unit, origin);
        sweep_direction                = relative_velocity * dt;
        sweep_speed                    = relative_velocity.length();
        sweep_length                   = sweep_speed * dt;
    }

    /**
     * @brief Identical post-hit response to `DtRemainColliderKernel`
     *        (delegated directly): reflect, then advect through the
     *        leftover timestep at the post-collision velocity. Only the
     *        hit-finding sweep (above) differs between the two kernels.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& position,
               Float3& velocity,
               const Float3& incident,
               const Float3& hit_position,
               const Float3& hit_normal,
               const float hit_distance,
               const float sweep_speed,
               const float dt,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept {
        DtRemainColliderKernel {}(
            position,
            velocity,
            incident,
            hit_position,
            hit_normal,
            hit_distance,
            sweep_speed,
            dt,
            unit,
            interaction);
    }
};

}
