#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

/**
 * @file tracing_despawn.h
 * @brief Despawn rule: a particle is removed if the straight-line sweep
 *        of its motion *this timestep* (not just its current position)
 *        crosses the unit's surface — a continuous-collision despawn
 *        test, avoiding the tunneling failure mode of only testing a
 *        particle's discrete end-of-step position against thin
 *        geometry.
 *
 * A fast-moving particle can cross an entire thin surface within one
 * timestep without either endpoint of its motion landing inside/on that
 * surface, so `SurfaceDespawn`/`VolumeDespawn`'s
 * discrete point tests would miss it. `TracingDespawn` instead
 * ray-casts the particle's current position along its velocity for the
 * timestep's duration (`query.trace`, the same ray-surface intersection
 * primitive `ColliderCollisionKernel` uses — see
 * `collider_collision_kernel.h`) and despawns it if that sweep segment
 * hits the surface — the same swept-motion idea as the collider module's
 * particle sweep, applied to outflow detection instead of reflection.
 */

namespace atlas {

/**
 * @brief `despawn()` true iff the ray from `position` along `velocity`
 *        hits the unit's surface within the swept distance
 *        `|velocity| * time`. See this file's top-of-file documentation.
 */
struct TracingDespawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& position,
            const Float3& velocity,
            const float time = 0.0f) noexcept {
        const float speed = velocity.length();
        if (!(time > 0.0f) || !(speed > 0.0f)) {
            return false;
        }
        const auto hit = query.trace(atlas::Ray(position, velocity));
        return hit.is_intersecting && hit.distance >= 0.0f && hit.distance <= speed * time;
    }
};

}
