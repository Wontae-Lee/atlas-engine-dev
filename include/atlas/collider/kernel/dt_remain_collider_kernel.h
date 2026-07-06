#pragma once

#include <atlas/collider/kernel/fast_collider_kernel.h>

/**
 * @file dt_remain_collider_kernel.h
 * @brief Middle-cost post-collision response
 *        (`atlas::PostColliderType::dt_remain`): corrects
 *        `FastColliderKernel`'s bias by ballistically advancing the
 *        particle through the leftover fraction of the timestep after
 *        impact, using the *post*-collision velocity.
 *
 * @details
 * ### Background
 * `FastColliderKernel` leaves every colliding particle sitting exactly on
 * the wall at the end of the timestep, discarding whatever time remained
 * after the hit (`dt - t_hit`). For a particle that reflects near the
 * start of a large `dt`, that discarded time is a real trajectory the
 * fast kernel simply skips, biasing near-wall number density and velocity
 * statistics low relative to the true sub-stepped trajectory. This
 * kernel is the standard DSMC fix: once the reflected/thermalized
 * velocity is known, advect the particle in a straight line at that new
 * velocity for exactly the leftover time `dt - t_hit`, so a particle
 * that hits early in the step still ends the step the correct distance
 * from the wall. It still assumes the wall itself did not move during
 * the leftover interval (see `PreciseColliderKernel` for the version that
 * also accounts for wall motion during the sweep to find `t_hit`).
 *
 * ### Operating principle
 * `hit_distance / sweep_speed` recovers the time-of-impact `t_hit` within
 * the step (the sweep was parameterized by distance, not time, so this
 * converts back). `remaining = max(dt - t_hit, 0)` is the leftover
 * fraction; the particle is placed at the (tolerance-offset) hit point
 * plus `reflected_velocity * remaining` — a second, unobstructed
 * straight-line segment. If `sweep_speed` is (numerically) zero, the
 * particle is left exactly at the hit point, matching
 * `FastColliderKernel`'s behavior in that degenerate case.
 */

namespace atlas {

/**
 * @brief `PostColliderType::dt_remain`: reflect at the hit point, then
 *        ballistically advect through the remaining fraction of the
 *        timestep at the post-collision velocity. See this file's
 *        top-of-file documentation for why this correction matters and
 *        how the remaining time is recovered.
 */
class DtRemainColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DtRemainColliderKernel() noexcept = default;

    /**
     * @brief Straight-line sweep in world space, identical to
     *        `FastColliderKernel::sweep_motion` — wall motion is not
     *        accounted for while searching for the hit, only in the
     *        post-hit response below.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit&,
                 const Vector3&,
                 const Vector3& incident,
                 const float incident_speed,
                 const float dt,
                 Vector3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept {
        sweep_direction = incident * dt;
        sweep_speed     = incident_speed;
        sweep_length    = incident_speed * dt;
    }

    /**
     * @brief Reflects at `hit_position` (as `FastColliderKernel` does),
     *        then advances the particle by `reflected_velocity *
     *        remaining`, `remaining = max(dt - hit_distance/sweep_speed,
     *        0)`. Falls back to leaving the particle exactly at the hit
     *        point when `sweep_speed` is not meaningfully positive (a
     *        near-stationary sweep has no well-defined impact time).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3& position,
               Vector3& velocity,
               const Vector3& incident,
               const Vector3& hit_position,
               const Vector3& hit_normal,
               const float hit_distance,
               const float sweep_speed,
               const float dt,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept {
        const Vector3 wall_velocity     = FastColliderKernel::surface_velocity(unit, hit_position);
        const Vector3 relative_incident = incident - wall_velocity;
        const Vector3 reflected         = interaction(relative_incident, hit_normal) + wall_velocity;
        const Vector3 offset_position   = hit_position + hit_normal * atlas::tol;

        position = offset_position;
        velocity = reflected;

        if (!(sweep_speed > atlas::eps)) {
            return;
        }

        const float hit_time  = hit_distance / sweep_speed;
        const float remaining = (dt > hit_time) ? (dt - hit_time) : 0.0f;

        position = offset_position + reflected * remaining;
    }
};

}
