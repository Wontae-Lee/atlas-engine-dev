#pragma once

#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/math/math.h>
#include <atlas/unit/unit.h>

/**
 * @file fast_collider_kernel.h
 * @brief Cheapest of the three post-collision response policies
 *        (`atlas::PostColliderType::fast`): snaps the particle exactly to
 *        the hit point and reflects/thermalizes its velocity, without
 *        accounting for the remaining fraction of the timestep after
 *        impact.
 *
 * @details
 * ### Background
 * A particle that hits a wall partway through a timestep `dt` has, in
 * reality, some time left over after the collision (`dt - t_hit`) during
 * which it should keep moving with its *post*-collision velocity. Fully
 * modeling that costs an extra ray segment per hit. `FastColliderKernel`
 * takes the cheapest approximation: place the particle directly on the
 * surface (offset by `atlas::tol` along the normal to avoid immediately
 * re-intersecting it next step) and give it the reflected velocity,
 * ignoring the leftover time entirely. This trades a small
 * (`O(dt * v)`-sized) positional bias at the wall for the lowest kernel
 * cost, which is often acceptable when `dt` is small relative to the
 * mean collision time (see `atlas::PostColliderType::dt_remain` /
 * `precise` for the exact alternatives).
 *
 * `surface_velocity()` (used by every kernel in this family, including
 * `DtRemainColliderKernel` and `PreciseColliderKernel`) computes the
 * rigid-body velocity of a point on a possibly moving/rotating unit:
 * `v(p) = v_com + omega x (p - translation)`, the standard rigid-body
 * velocity field, so wall interactions can be evaluated in the wall's
 * local rest frame and the wall's own momentum added back afterward.
 */

namespace atlas {

/**
 * @brief `PostColliderType::fast`: reflect at the hit point, discard
 *        the timestep remainder. See this file's top-of-file
 *        documentation for the tradeoff this makes.
 */
class FastColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    FastColliderKernel() noexcept = default;

    /**
     * @brief Rigid-body velocity of `unit`'s surface at `surface_point`:
     *        `v_com + omega x (surface_point - translation)`. Either term
     *        is omitted (treated as zero) if the unit has no linear/
     *        angular velocity configured, so a static unit returns the
     *        zero vector cheaply.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    surface_velocity(const Unit& unit,
                     const Float3& surface_point) noexcept {
        Float3 velocity(0.0f, 0.0f, 0.0f);

        if (unit.velocity().has_value()) {
            velocity += *unit.velocity();
        }

        if (unit.angular_velocity().has_value()) {
            const Float3 radius = surface_point - unit.sync().translation;
            velocity += atlas::cross(*unit.angular_velocity(), radius);
        }

        return velocity;
    }

    /**
     * @brief Straight-line sweep in world space: ignores unit motion
     *        entirely (the wall's own velocity is only accounted for in
     *        `operator()`'s post-hit response, not in the ray used to
     *        find the hit). Cheapest of the three `sweep_motion`
     *        implementations in this kernel family.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit&,
                 const Float3&,
                 const Float3& incident,
                 const float incident_speed,
                 const float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept {
        sweep_direction = incident * dt;
        sweep_speed     = incident_speed;
        sweep_length    = incident_speed * dt;
    }

    /**
     * @brief Places the particle at `hit_position` (nudged off the
     *        surface by `atlas::tol` along `hit_normal` to avoid a
     *        self-intersection on the next sweep) and sets its velocity
     *        to the wall interaction's response, evaluated in the wall's
     *        local rest frame and shifted back by `surface_velocity`.
     *        The unused trailing `float` parameters (hit distance, sweep
     *        speed, dt) are accepted only for interface parity with the
     *        other kernels in this family, which do use them.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& position,
               Float3& velocity,
               const Float3& incident,
               const Float3& hit_position,
               const Float3& hit_normal,
               const float,
               const float,
               const float,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept {
        const Float3 wall_velocity     = surface_velocity(unit, hit_position);
        const Float3 relative_incident = incident - wall_velocity;

        position = hit_position + hit_normal * atlas::tol;
        velocity = interaction(relative_incident, hit_normal) + wall_velocity;
    }
};

}
