#pragma once

#include <atlas/collider/collider_probe.h>
#include <atlas/collider/hit_collider.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/core/macros.h>
#include <atlas/parallel/parallel_for.h>

#include <cstddef>
#include <cstdint>

namespace atlas::detail {

class ColliderCollisionKernel final {
private:
    struct ParticleSweep {

        Float3 origin {};

        Float3 velocity {};

        Float3 direction {};

        float speed {};

        float length {};
    };

public:
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
            hit.point      = sync_op.sync_to_world(local_hit.point);
            hit.normal     = sync_op.sync_dir_to_world(local_hit.normal);
            hit.unit_index = unit_index;
        }

        return hit;
    }

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
