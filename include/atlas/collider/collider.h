#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_probe.h>
#include <atlas/collider/detail/collider_collision_kernel.h>
#include <atlas/collider/detail/collider_probe_builder.h>
#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstdint>

/**
 * @file collider.h
 * @brief Top-level owner of a scene's boundary units and the entry point
 *        for advancing them and resolving particle-boundary collisions
 *        each timestep; the object every other file in `collider/`
 *        ultimately serves.
 *
 * @details
 * ### Operating principle
 * A `Collider` owns:
 *   - `_universe`: the domain that centrally owns this collider's boundary
 *     units (walls, inlets, embedded solids — see `atlas::Unit`), reached
 *     as `_universe->collider_units()` (`atlas::UnitField`): each unit
 *     carries its own shape, transform, and optionally linear/angular
 *     velocity, alongside the broad-phase per-unit/scene AABB cache kept in
 *     sync with them. The collider borrows this field (shared handle); it
 *     does not own the units;
 *   - `_surface_interactions` / `_flips`: the wall-interaction model(s)
 *     (`SurfaceInteractionKernel`, either shared across all units or one
 *     per unit) and optional per-unit hit-normal flips;
 *   - `_fluid`: the particle population the collider acts on (not owned
 *     — a shared handle);
 *   - `_probe`: a `ColliderProbe` reused across steps and rebuilt (never
 *     reallocated in place) whenever `collide()` runs.
 *
 * `update(dt)` advances every unit's own motion (`Unit::update`, e.g.
 * integrating a scripted trajectory) and then calls `collide(dt)`.
 * `collide(dt)`:
 *   1. Refreshes `_unit_field`'s bounds from the (possibly just-moved) units.
 *   2. Rebuilds `_probe` (`Collider::make_probe`); bails out
 *      early if there is nothing to collide (empty units/interactions,
 *      no fluid, or zero particles).
 *   3. Launches `detail::ColliderCollisionKernel::resolve_particles` for
 *      the actual per-particle sweep-and-response pass (see that file for
 *      the ray-tracing algorithm).
 * Splitting `update`/`collide` lets a caller advance unit motion and
 * particle-boundary collision independently when needed (e.g. a fixed
 * scene collided against a moving fluid without re-deriving unit motion
 * each step).
 */

namespace atlas {

/**
 * @brief Owns a scene's boundary units and orchestrates per-timestep
 *        particle-boundary collision detection/response against a
 *        `Fluid`. See this file's top-of-file documentation for the
 *        `update`/`collide` pipeline.
 */
class Collider final {
public:
    class Builder;

public:
    Collider() = default;

    Collider(const Collider&) = delete;

    Collider(Collider&&) noexcept = default;

    ~Collider() = default;

    Collider&
    operator=(const Collider&)
        = delete;

    Collider&
    operator=(Collider&&) noexcept = default;

    ATLAS_HOST
    Collider(UniverseHostPtr universe,
             DeviceBuffer<SurfaceInteractionKernel> surface_interactions,
             DeviceBuffer<std::uint8_t> flips,
             PostColliderType post_collider_type,
             atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Advances every unit's own motion by `dt` (a no-op for
     *        `dt <= 0` or an empty collider), then calls `collide(dt)`.
     */
    ATLAS_HOST void
    update(float dt);

    /**
     * @brief Refreshes the bound cache, rebuilds the collision probe,
     *        and resolves particle-boundary collisions for `dt` (a no-op
     *        for `dt <= 0` or `empty()`). See this file's top-of-file
     *        documentation for the full pipeline.
     */
    ATLAS_HOST void
    collide(float dt) const;

    /** @brief Whether there is nothing to collide: no units, no surface
     *  interactions configured, or no fluid attached. */
    ATLAS_NODISCARD ATLAS_HOST bool
    empty() const noexcept;

    /**
     * @brief Rebuilds `_probe` from the collider's current buffers and
     *        `_fluid`'s current particle state (`Collider::make_probe`).
     * @return `false` if there is nothing to collide against right now
     *         (see `Collider::make_probe`); `true` if
     *         `_probe` is now valid for a collision pass.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() const noexcept;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    DeviceBuffer<SurfaceInteractionKernel> _surface_interactions;

    DeviceBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };

    mutable ColliderProbe _probe {};
};

/**
 * @brief Fluent builder for `Collider`.
 *
 * Validation (`validate()`, run by `build()`/`make_host_shared()`):
 * a non-null `_fluid` and a non-null `_universe` holding at least one
 * collider unit are required; if surface interactions or flips are
 * provided at all, their count must be either `1` (broadcast to every
 * unit, see `ColliderProbe`) or exactly the unit count. If no surface
 * interaction is provided, `build()` defaults to a
 * single default-constructed `SurfaceInteractionKernel` (isothermal,
 * fully elastic/diffuse); if no flips are provided, it defaults to a
 * single non-flipping entry.
 */
class Collider::Builder final {
public:
    Builder() = default;

    /** @brief The domain that owns this collider's boundary units
     *  (registered via `Universe::Builder::with_collider_units`); required,
     *  must be non-null and hold at least one collider unit. */
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    /** @brief The particle population this collider will act on;
     *  required, must be non-null. */
    ATLAS_HOST Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    /** @brief Sets the wall-interaction model(s) to the restitution-based
     *  `IsothermalSurfaceInteraction`, one per unit or a single shared
     *  one (see this class's validation rules). */
    ATLAS_HOST Builder&
    with_surface_interactions(const HostBuffer<IsothermalSurfaceInteraction>& surface_interactions);

    /** @brief Sets the wall-interaction model(s) to the full Maxwell-
     *  model `MaxwellianSurfaceInteraction`, one per unit or a single
     *  shared one (see this class's validation rules). */
    ATLAS_HOST Builder&
    with_surface_interactions(const HostBuffer<MaxwellianSurfaceInteraction>& surface_interactions);

    /** @brief Sets a single shared `SurfaceInteractionKernel` (already
     *  wrapping either model) for every unit. */
    ATLAS_HOST Builder&
    with_surface_interaction_kernel(const SurfaceInteractionKernel& surface_interaction);

    /** @brief Sets one already-wrapped `SurfaceInteractionKernel` per
     *  unit (or a single shared one), mixing isothermal and Maxwellian
     *  models across units if desired. */
    ATLAS_HOST Builder&
    with_surface_interaction_kernels(const HostBuffer<SurfaceInteractionKernel>& surface_interactions);

    /** @brief Sets a single shared hit-normal flip flag for every unit. */
    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

    /** @brief Sets one hit-normal flip flag per unit (or a single shared
     *  one); useful when a shape's authored winding faces away from the
     *  gas domain. */
    ATLAS_HOST Builder&
    with_flips(const HostBuffer<std::uint8_t>& flips);

    /** @brief Selects the post-collision response policy applied to
     *  every unit; see `PostColliderType`. Defaults to `fast`. */
    ATLAS_HOST Builder&
    with_post_collider_type(PostColliderType type) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Collider
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Collider>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    HostBuffer<SurfaceInteractionKernel> _surface_interactions;

    HostBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };
};

using ColliderHostPtr = atlas::host_shared_ptr<Collider>;

using ColliderDevicePtr = atlas::device_shared_ptr<Collider>;

}
