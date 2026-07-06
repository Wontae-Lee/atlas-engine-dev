#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/sink/despawn.h>
#include <atlas/sink/detail/sink_unit_bounds.h>
#include <atlas/sink/sink_probe.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstddef>

/**
 * @file sink.h
 * @brief Owns a scene's outflow/absorbing boundaries and removes fluid
 *        particles that meet the configured despawn condition each
 *        timestep, compacting the fluid's storage afterward.
 *
 * @details
 * ### Operating principle
 * A `Sink` owns:
 *   - `_universe`: the domain that centrally owns this sink's boundary
 *     units (walls/ports/regions to remove particles at), reached as
 *     `_universe->sink_units()` (`atlas::UnitField`); the sink borrows the
 *     field (shared handle) and does not own the units;
 *   - `_despawn_types`/`_despawn_operators`: which despawn rule (surface
 *     / volume / tracing, see `despawn.h`) each unit uses,
 *     either one shared operator or one per unit;
 *   - `_fluid`: the particle population this sink removes particles
 *     from (not owned — a shared handle);
 *   - `_flip`: inverts the keep/despawn decision — with `_flip == false`
 *     (the default outflow-sink behavior), a particle matching *any*
 *     unit's despawn condition is removed; with `_flip == true`, only
 *     particles matching a despawn condition are *kept* (everything
 *     else is removed) — letting a `Sink` double as a "keep only inside
 *     this region" domain-bounding filter instead of an outflow.
 *
 * `update(dt)` advances every unit's own motion then calls `sink(dt)`.
 * `sink(dt)`:
 *   1. Rebuilds `_unit_bound_cache`/`_probe` (tolerance-expanded AABBs,
 *      see `detail::SinkUnitBounds`, plus the fluid's particle buffers).
 *   2. `despawn_particles`: per particle, tests every unit's despawn
 *      condition (broad-phase AABB reject first, then the exact
 *      `Despawn::despawn` query in the unit's local frame),
 *      writing the particle's new `FluidActiveState` flag (`1` = keep,
 *      `0` = remove) per `_flip`'s inverted-or-not rule. If an
 *      `ObserverHostPtr` with a `SinkSensorMetrics` sensor is attached,
 *      also records which unit despawned each particle for
 *      per-unit removal-count metrics.
 *   3. `compact_fluid_particles()`: stream-compacts the fluid's particle
 *      buffers, physically removing every particle flagged inactive
 *      (see that method's own documentation for the compaction
 *      algorithm) — `despawn_particles` only marks particles, this step
 *      actually shrinks storage.
 */

namespace atlas {

/**
 * @brief Owns a scene's outflow/absorbing boundaries and removes
 *        matching particles from a `Fluid` each timestep. See this
 *        file's top-of-file documentation for the despawn/compaction
 *        pipeline and the `flip` inversion.
 */
class Sink final {
public:
    class Builder;

public:
    Sink() = default;

    Sink(const Sink&) = delete;

    Sink(Sink&&) noexcept = default;

    ~Sink() = default;

    Sink&
    operator=(const Sink&)
        = delete;

    Sink&
    operator=(Sink&&) noexcept = default;

    ATLAS_HOST
    Sink(UniverseHostPtr universe,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<Despawn> despawn_operators,
         atlas::host_shared_ptr<atlas::Fluid> fluid,
         bool flip                = false,
         float tolerance          = 0.0f,
         ObserverHostPtr observer = nullptr) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /** @brief Advances every unit's own motion by `dt`, then calls
     *  `sink(dt)`. No-op for an empty sink or `dt <= 0`. */
    ATLAS_HOST void
    update(float dt);

    /** @brief Marks despawned particles inactive and compacts the fluid
     *  (see this file's top-of-file documentation for the full
     *  despawn/compaction/metrics pipeline). `dt` only matters for
     *  `Tracing`-type despawn rules (the sweep duration). */
    ATLAS_HOST void
    sink(float dt = 0.0f);

    /**
     * @brief Stream-compacts every `Fluid` state buffer to remove
     *        particles flagged inactive (`FluidActiveState == 0`),
     *        shrinking `Fluid::particle_count()`.
     *
     * Standard exclusive-scan compaction: builds a `0`/`1` keep flag per
     * particle, exclusive-scans it into each surviving particle's new
     * (packed) index, scatters surviving *old* indices into
     * `_compact_indices` at their new position, then asks every
     * `FluidState` to gather itself through `_compact_indices` (each
     * state's own `compact()`, so position/velocity/species/... all move
     * together). Trailing slack past the new `particle_count` is zeroed
     * in `FluidActiveState` so stale "active" flags don't linger past
     * the buffer's logical end. Short-circuits if every particle
     * survived (nothing to compact) or none did (fills all-inactive and
     * sets `particle_count` to `0` without a gather pass).
     */
    ATLAS_HOST void
    compact_fluid_particles();

    /** @brief Refreshes unit bounds and rebuilds `_probe`
     *  (`Sink::make_probe`); `false` if there is nothing
     *  to despawn against right now. */
    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe(float dt = 0.0f) noexcept;

private:
    ATLAS_HOST void
    refresh_unit_bounds() noexcept;

public:
    /** @brief Per-particle despawn test and `FluidActiveState` update;
     *  see this file's top-of-file documentation. `removed_unit_indices`
     *  (if non-null) records which unit despawned each particle, or
     *  `-1` for particles kept, for `SinkSensorMetrics`. */
    ATLAS_HOST void
    despawn_particles(const SinkProbe& probe, int* removed_unit_indices);

private:
    UniverseHostPtr _universe;

    DeviceBuffer<atlas::AABB> _unit_bounds;

    DeviceBuffer<DespawnType> _despawn_types;

    DeviceBuffer<Despawn> _despawn_operators;

    atlas::host_shared_ptr<atlas::Fluid> _fluid;

    ObserverHostPtr _observer {};

    bool _flip = false;

    float _tolerance = 0.0f;

    DeviceBuffer<std::size_t> _keep;

    DeviceBuffer<std::size_t> _offsets;

    DeviceBuffer<std::size_t> _compact_indices;

    DeviceBuffer<int> _despawned_unit_indices;

    DeviceBuffer<std::size_t> _total_count_buffer {};

    SinkProbe _probe {};

    std::size_t _step_index = 0;

    detail::SinkUnitBounds _unit_bound_cache {};
};

/**
 * @brief Fluent builder for `Sink`. Validation requires a non-null
 *        `_fluid`, a non-null `_universe` holding at least one sink unit,
 *        and both `_despawn_types` and `_despawn_operators` non-empty with
 *        size `1` (shared) or exactly the sink-unit count.
 */
class Sink::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Sink
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Sink>
    make_host_shared();

    /** @brief The domain that owns this sink's units (registered via
     *  `Universe::Builder::with_sink_units`); required, non-null. */
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    /** @brief Optional; if set with a `SinkSensorMetrics` sensor,
     *  enables per-unit despawn-count recording (see
     *  `Sink::sink`'s top-of-file documentation). */
    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /** @brief Appends despawn-type tags (informational bookkeeping
     *  alongside `with_despawn_operator(s)`, which carries the actual
     *  callable rule); must end up size `1` or matching the unit count. */
    ATLAS_HOST Builder&
    with_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    /** @brief Appends a single shared `Despawn` for every unit. */
    ATLAS_HOST Builder&
    with_despawn_operator(const Despawn& despawn_operator) noexcept;

    /** @brief Appends one `Despawn` per unit (or a single
     *  shared one), mixing despawn rules across units if desired. */
    ATLAS_HOST Builder&
    with_despawn_operators(const HostBuffer<Despawn>& despawn_operators);

    /** @brief Tolerance passed to `Surface`/`Volume` despawn queries and
     *  used to expand the broad-phase unit bounds. */
    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    HostBuffer<DespawnType> _despawn_types;

    HostBuffer<Despawn> _despawn_operators;

    bool _flip = false;

    float _tolerance = 0.0f;
};

using SinkHostPtr = atlas::host_shared_ptr<Sink>;

using SinkDevicePtr = atlas::device_shared_ptr<Sink>;

}
