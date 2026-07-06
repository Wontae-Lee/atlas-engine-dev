#pragma once

#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

/**
 * @file collider_probe.h
 * @brief Flat, device-copyable view over everything one collision-detection
 *        pass needs: boundary geometry, cached bounds, surface interaction
 *        models, and the fluid's particle buffers.
 *
 * @details
 * ### Background
 * Kernels launched on the device cannot hold owning containers (e.g.
 * `std::vector`, `atlas::host_shared_ptr`) or call virtual methods across
 * the host/device boundary (Atlas builds without `-rdc`, so device code
 * cannot resolve virtual dispatch or cross translation units — see
 * `docs/architecture/04-backend-portability.md`). The `*Probe` pattern
 * used across Atlas (`ColliderProbe`, `SinkProbe`, `SourceProbe`, ...)
 * works around this: a probe is a small, trivially-copyable struct of raw
 * pointers and counts into buffers that are actually owned elsewhere
 * (here, by `Collider`, its `Unit` buffer, and the `Fluid` it acts on).
 * The probe is rebuilt (`ColliderProbeBuilder::make`) whenever those
 * owners' buffers might have been reallocated, then captured by value
 * into a device lambda for one collision pass.
 *
 * ### Operating principle
 * `ColliderCollisionKernel::resolve_particles` captures a `ColliderProbe`
 * by value and, per particle, sweeps `positions[i]`/`velocities[i]`
 * against every unit's geometry (using `unit_bounds`/`scene_bound` as a
 * broad-phase reject test, see `atlas::UnitField`), then mutates
 * `positions`/`velocities`/`internal_energies` in place through the
 * probe's pointers. `surface_interactions`/`flips`/`materials` are looked
 * up per hit unit/species (with a size-1 broadcast convention: if
 * `interaction_count == 1` or a unit's own index is out of range, index
 * `0` is reused — letting a single shared interaction model cover every
 * unit).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `Collider` + its `Fluid`, passed by
 *        value into the collision-detection device kernel.
 *
 * See this file's top-of-file documentation for the probe pattern and how
 * `ColliderCollisionKernel` consumes each field.
 */
struct ColliderProbe {
    /** Boundary units (geometry + sync/transform), owned by `Collider`. */
    const Unit* units {};
    /** Per-unit world-space AABB, parallel to `units` (see
     *  `atlas::UnitField`). */
    const atlas::AABB* unit_bounds {};
    /** Surface interaction model(s); indexed per hit unit, or broadcast
     *  from index 0 when `interaction_count == 1`. */
    const SurfaceInteractionKernel* surface_interactions {};
    /** Per-unit "flip the hit normal" flags; same broadcast convention as
     *  `surface_interactions` when `flip_count == 1`. */
    const std::uint8_t* flips {};
    /** Union of every unit's bound; used as a cheap scene-level reject
     *  test before the per-unit sweep loop. */
    atlas::AABB scene_bound {};

    /** Particle world-space positions, owned by `Fluid`; updated in place
     *  by the collision kernel. */
    Vector3* positions {};
    /** Particle world-space velocities, owned by `Fluid`; updated in
     *  place (reflected/thermalized on a wall hit). */
    Vector3* velocities {};
    /** Particle rotational/vibrational/translational energy, owned by
     *  `Fluid`; `nullptr` if the fluid does not track internal energy. */
    FluidInternalEnergy* internal_energies {};
    /** Particle species indices, owned by `Fluid`; used to look up
     *  `materials` for internal-energy accommodation. */
    const std::size_t* species {};
    /** Per-species material properties (rotational/vibrational dof and
     *  characteristic temperatures). */
    const MaterialProperties* materials {};

    /** Length of `units`/`unit_bounds`. */
    int unit_count {};
    /** Length of `surface_interactions` (`1` under the broadcast
     *  convention). */
    int interaction_count {};
    /** Length of `flips` (`1` under the broadcast convention; `0` if no
     *  flips are configured). */
    int flip_count {};
    /** Length of `materials`. */
    int material_count {};
    /** Length of `positions`/`velocities`/`species` (and
     *  `internal_energies` when non-null). */
    int particle_count {};
    /** Whether `scene_bound` is currently a valid superset of every
     *  `unit_bounds` entry (see `atlas::UnitField::covers_units`). */
    bool scene_bound_covers_units {};
};

}
