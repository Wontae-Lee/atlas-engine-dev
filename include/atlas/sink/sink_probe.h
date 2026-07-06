#pragma once

#include <atlas/sink/despawn.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>

/**
 * @file sink_probe.h
 * @brief Flat, device-copyable view over everything one despawn pass
 *        needs: sink units, cached bounds, despawn rules, and the
 *        fluid's particle buffers. Same non-owning-probe pattern as
 *        `ColliderProbe` (see `collider_probe.h`).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `Sink` + its `Fluid`, passed by value
 *        into the despawn device kernel. See `detail::SinkProbeBuilder`
 *        for how it is filled and `sink.h`/`despawn.h` for how
 *        each field is used.
 */
struct SinkProbe {
    /** Sink units (geometry + sync/transform), owned by `Sink`. */
    const Unit* units {};
    /** Per-unit world-space AABB, tolerance-expanded (see
     *  `detail::SinkUnitBounds`); used as a broad-phase reject test
     *  before the exact geometry query. */
    const atlas::AABB* unit_bounds {};
    /** Despawn rule(s); indexed per unit, or broadcast from index 0
     *  when `despawn_operator_count == 1`. */
    const Despawn* despawn_operators {};
    /** Particle world-space positions, owned by `Fluid`. */
    const Vector3* positions {};
    /** Particle world-space velocities, owned by `Fluid`; `nullptr` if
     *  the fluid does not track velocity (only needed for `Tracing`
     *  despawn rules). */
    const Vector3* velocities {};
    /** Particle active flags, owned by `Fluid`; written in place — `0`
     *  marks a particle for removal by `Sink::compact_fluid_particles`. */
    int* active {};

    /** Length of `units`/`unit_bounds`. */
    int unit_count {};
    /** Length of `despawn_operators` (`1` under the broadcast
     *  convention). */
    int despawn_operator_count {};
    /** Length of `positions`/`active` (and `velocities` when non-null). */
    std::size_t particle_count {};

    /** Whether the keep/despawn decision is inverted; see `Sink`'s
     *  top-of-file documentation. */
    bool flip {};
    /** Tolerance passed to `Surface`/`Volume` despawn queries. */
    float tolerance {};
    /** Timestep passed to `Tracing` despawn queries (the sweep
     *  duration). */
    float time_step {};
};

}
