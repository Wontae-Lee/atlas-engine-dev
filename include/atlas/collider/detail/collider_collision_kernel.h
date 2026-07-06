#pragma once

#include <atlas/collider/collider_probe.h>
#include <atlas/collider/detail/hit_collider.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/core/macros.h>
#include <atlas/parallel/parallel_for.h>

#include <cstddef>
#include <cstdint>

/**
 * @file collider_collision_kernel.h
 * @brief The per-timestep particle-vs-boundary collision detection and
 *        response kernel: for every particle, finds the earliest wall
 *        intersection along its motion this step (if any) and applies the
 *        configured post-collision response.
 *
 * @details
 * ### Background
 * This is a discrete-collision (as opposed to continuous/analytic)
 * approach to particle-boundary interaction, standard in
 * particle-in-cell/DSMC codes: each particle's motion over one timestep
 * `dt` is treated as a straight (or wall-relative, see
 * `PreciseColliderKernel`) line segment ("sweep"), and finding whether/
 * where that segment first crosses a boundary is exactly a ray-vs-scene
 * intersection query, the same primitive used in ray-tracing renderers.
 * Correctness requires taking the *closest* (earliest-time) intersection
 * over *all* boundary units — a particle could geometrically cross
 * several surfaces along its sweep, but physically only interacts with
 * the first one it reaches.
 *
 * ### Operating principle
 * `resolve_particles` launches one device thread per particle
 * (`atlas::parallel_for<ExecutionPolicy::device>`) capturing the
 * `ColliderProbe` by value (see that file's documentation for why probes
 * are passed this way). Per particle:
 * 1. `make_particle_sweep` builds a `ParticleSweep` from the particle's
 *    current position/velocity and `dt` — direction = `velocity * dt`,
 *    length = `speed * dt` (a straight-line segment covering exactly this
 *    timestep's motion at constant velocity).
 * 2. If the sweep has ~zero length (`fast`/`dt_remain` post-collider
 *    types only — `precise` cannot skip this early, since a
 *    world-space-stationary particle can still have a nonzero *relative*
 *    sweep against a moving wall), skip straight to `closest_hit`.
 * 3. `closest_hit` finds the earliest intersection:
 *      - Broad-phase: if there is more than one unit and the scene bound
 *        is currently valid (`ColliderProbe::scene_bound_covers_units`),
 *        the whole sweep is rejected in one test against `scene_bound`
 *        before touching any per-unit data — see
 *        `atlas::UnitField`. Skipped for `PostColliderType::precise`,
 *        since each unit's *relative* sweep direction/length differs
 *        (`sweep_motion` per unit), so a single shared scene-level ray
 *        cannot be reused across units.
 *      - Per unit: recompute the sweep in that unit's response-policy
 *        frame (`PostColliderKernel::sweep_motion` — identity for
 *        `fast`/`dt_remain`, wall-relative for `precise`), reject against
 *        that unit's own AABB (`unit_bounds`), then transform the sweep
 *        ray into the unit's *local* space via `Unit::sync()`
 *        (handling translation/rotation/scale so each `Geometry`
 *        only ever intersects axis-local canonical shapes — see
 *        `docs/architecture/04-backend-portability.md` §4.7) and run the
 *        exact `Geometry::operator()` ray intersection.
 *      - Tracks the intersection with the smallest `hit_time =
 *        local_hit.distance / sweep_speed` seen so far (comparing *time*,
 *        not raw distance, is what makes "earliest" well-defined when
 *        different units can have different relative sweep speeds under
 *        `precise`); converts the winning hit's position/normal back to
 *        world space via the same sync operator.
 * 4. No hit: the particle simply advances by the full sweep
 *    (`position += direction`), i.e. ordinary free-streaming motion.
 * 5. Hit found: `resolve_hit` looks up the hit unit's configured surface
 *    interaction (broadcasting from a single shared interaction/flip
 *    entry when `probe.interaction_count`/`flip_count == 1`, or when the
 *    hit unit's index exceeds the array — see `ColliderProbe`), optionally
 *    flips the hit normal (`ColliderProbe::flips`, for surfaces whose
 *    authored winding faces the "wrong" way), invokes the
 *    `PostColliderKernel` to update position/velocity, then — if the
 *    fluid tracks internal energy and the particle's species has a known
 *    material — relaxes rotational/vibrational energy toward the wall via
 *    `SurfaceInteractionKernel::internal_energy`, using the particle's
 *    velocity *relative to the (possibly moving/rotating) wall surface*
 *    at the hit point (`FastColliderKernel::surface_velocity`).
 */

namespace atlas::detail {

/**
 * @brief Stateless launcher for the per-particle collision sweep; see
 *        this file's top-of-file documentation for the full algorithm.
 */
class ColliderCollisionKernel final {
private:
    /**
     * @brief One particle's straight-line motion segment for this
     *        timestep, in whatever frame `PostColliderKernel::sweep_motion`
     *        chose (world space for `fast`/`dt_remain`, wall-relative for
     *        `precise`).
     */
    struct ParticleSweep {
        /** Segment start (the particle's current position). */
        Float3 origin {};
        /** Particle velocity used to build this sweep (world-space,
         *  before any per-unit relative-frame adjustment). */
        Float3 velocity {};
        /** Segment displacement (`velocity * dt`, or its wall-relative
         *  equivalent per unit under `precise`). */
        Float3 direction {};
        /** Segment speed (`|direction| / dt`). */
        float speed {};
        /** Segment length (`speed * dt`). */
        float length {};
    };

public:
    /**
     * @brief Runs one collision-detection-and-response pass over every
     *        particle in `probe` for timestep `dt`, launching one device
     *        thread per particle. See this file's top-of-file
     *        documentation for the full per-particle algorithm.
     */
    ATLAS_HOST static void
    resolve_particles(const ColliderProbe& probe,
                      const PostColliderType post_collider_type,
                      const float dt) {
        const auto collision_probe = probe;
        const PostColliderKernel post_collider_kernel(post_collider_type);

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            collision_probe.particle_count,
            [=] ATLAS_ALL_DEVICE(const int i) {
                const ParticleSweep sweep = ColliderCollisionKernel::make_particle_sweep(
                    collision_probe,
                    i,
                    dt);

                if (sweep.length <= atlas::eps
                    && post_collider_kernel.type != PostColliderType::precise) {
                    return;
                }

                const HitCollider hit = ColliderCollisionKernel::closest_hit(
                    collision_probe,
                    post_collider_kernel,
                    sweep,
                    dt);

                if (!hit.found()) {
                    collision_probe.positions[i] = sweep.origin + sweep.direction;
                    return;
                }

                ColliderCollisionKernel::resolve_hit(
                    collision_probe,
                    post_collider_kernel,
                    i,
                    sweep,
                    hit,
                    dt);
            });
    }

private:
    /** @brief Straight-line world-space sweep for one particle over
     *  `dt`: step 1 of the algorithm in this file's top-of-file docs. */
    ATLAS_ALL_DEVICE static ParticleSweep
    make_particle_sweep(const ColliderProbe& probe,
                        const int particle_index,
                        const float dt) {
        const Float3 velocity  = probe.velocities[particle_index];
        const Float3 direction = velocity * dt;
        const float speed      = velocity.length();

        return ParticleSweep {
            probe.positions[particle_index],
            velocity,
            direction,
            speed,
            speed * dt
        };
    }

    /** @brief Broad-phase-culled, closest-in-time ray-vs-all-units
     *  intersection query: step 3 of the algorithm in this file's
     *  top-of-file docs. */
    ATLAS_ALL_DEVICE static HitCollider
    closest_hit(const ColliderProbe& probe,
                const PostColliderKernel& post_collider_kernel,
                const ParticleSweep& sweep,
                const float dt) {
        HitCollider hit {};
        float closest_time = atlas::far;

        const bool moving_surface_sweep = post_collider_kernel.type == PostColliderType::precise;
        const atlas::Ray particle_ray(sweep.origin, sweep.direction);

        if (!moving_surface_sweep && probe.unit_count > 1 && probe.scene_bound_covers_units) {
            const auto scene_hit = probe.scene_bound.trace(particle_ray);
            if (!scene_hit.is_intersecting || scene_hit.enter > sweep.length) {
                return hit;
            }
        }

        for (int unit_index = 0; unit_index < probe.unit_count; ++unit_index) {
            const Unit* unit       = nullptr;
            Float3 sweep_direction = sweep.direction;
            float sweep_speed      = sweep.speed;
            float sweep_length     = sweep.length;

            if (moving_surface_sweep) {
                unit = probe.units + unit_index;
                post_collider_kernel.sweep_motion(*unit,
                                                  sweep.origin,
                                                  sweep.velocity,
                                                  sweep.speed,
                                                  dt,
                                                  sweep_direction,
                                                  sweep_speed,
                                                  sweep_length);
            }

            if (sweep_length <= atlas::eps || sweep_speed <= atlas::eps) {
                continue;
            }

            const atlas::Ray world_ray = moving_surface_sweep
                ? atlas::Ray(sweep.origin, sweep_direction)
                : particle_ray;
            const auto& unit_bound     = probe.unit_bounds[unit_index];
            if (unit_bound.is_valid()) {
                const auto bound_hit = unit_bound.trace(world_ray);
                if (!bound_hit.is_intersecting || bound_hit.enter > sweep_length) {
                    continue;
                }
            }

            if (unit == nullptr) {
                unit = probe.units + unit_index;
            }

            const auto& sync_op        = unit->sync();
            const auto& geom_op        = unit->geometry();
            const atlas::Ray local_ray = sync_op.sync_to_local(world_ray);
            const HitSurface local_hit = geom_op(local_ray);

            if (!local_hit.is_intersecting || local_hit.distance > sweep_length) {
                continue;
            }

            const float hit_time = local_hit.distance / sweep_speed;
            if (hit_time >= closest_time) {
                continue;
            }

            closest_time   = hit_time;
            hit.distance   = local_hit.distance;
            hit.speed      = sweep_speed;
            hit.point   = sync_op.sync_to_world(local_hit.point);
            hit.normal     = sync_op.sync_dir_to_world(local_hit.normal);
            hit.unit_index = unit_index;
        }

        return hit;
    }

    /** @brief Applies the configured post-collision response and, where
     *  applicable, internal-energy relaxation for one confirmed hit:
     *  step 5 of the algorithm in this file's top-of-file docs. */
    ATLAS_ALL_DEVICE static void
    resolve_hit(const ColliderProbe& probe,
                const PostColliderKernel& post_collider_kernel,
                const int particle_index,
                const ParticleSweep& sweep,
                const HitCollider& hit,
                const float dt) {
        const int interaction_index = (probe.interaction_count == 1 || hit.unit_index >= probe.interaction_count)
            ? 0
            : hit.unit_index;
        const int flip_index        = (probe.flip_count == 1 || hit.unit_index >= probe.flip_count)
                   ? 0
                   : hit.unit_index;
        const bool flip_normal      = probe.flip_count > 0 && probe.flips[flip_index] != std::uint8_t { 0 };
        const Float3 hit_normal     = flip_normal ? -hit.normal : hit.normal;
        const auto& hit_unit        = probe.units[hit.unit_index];
        const auto& interaction     = probe.surface_interactions[interaction_index];

        post_collider_kernel(
            probe.positions[particle_index],
            probe.velocities[particle_index],
            sweep.velocity,
            hit.point,
            hit_normal,
            hit.distance,
            hit.speed,
            dt,
            hit_unit,
            interaction);

        if (probe.internal_energies == nullptr) {
            return;
        }

        const std::size_t species_index = probe.species[particle_index];
        if (species_index >= static_cast<std::size_t>(probe.material_count)) {
            return;
        }

        const Float3 wall_velocity              = FastColliderKernel::surface_velocity(hit_unit, hit.point);
        probe.internal_energies[particle_index] = interaction.internal_energy(
            probe.internal_energies[particle_index],
            sweep.velocity - wall_velocity,
            hit_normal,
            probe.materials[species_index]);
    }
};

}
