#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/source/detail/source_cache_builder.h>
#include <atlas/source/detail/source_emitter.h>
#include <atlas/source/detail/source_probe_builder.h>
#include <atlas/source/detail/source_species_shuffler.h>
#include <atlas/source/source_probe.h>
#include <atlas/source/spawn.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <cstdint>

/**
 * @file source.h
 * @brief Owns a scene's inflow boundaries and injects new particles into
 *        a `Fluid` each timestep from a cached, per-unit lattice of
 *        valid spawn positions.
 *
 * @details
 * ### Operating principle
 * A `Source` owns:
 *   - `_universe`: the domain that centrally owns this source's inflow
 *     boundaries, reached as `_universe->source_units()`
 *     (`atlas::UnitField`); the source borrows the field (shared handle)
 *     and does not own the units;
 *   - `_spawn_types`/`_spawn_operators`: which acceptance rule (surface
 *     / volume, see `spawn.h`) each unit uses for candidate
 *     spawn positions;
 *   - `_fluid`: the particle population this source injects into;
 *   - `_flip`: inverts candidate acceptance (same convention as `Sink`'s
 *     `flip` — see `sink.h`);
 *   - `_spacing`: the candidate lattice spacing (see
 *     `detail::SourceCacheBuilder`);
 *   - `_temperature`: the emission temperature passed to each species'
 *     velocity generator.
 *
 * `update(dt)` advances every unit's own motion then calls `emit()`.
 * `emit()`:
 *   1. `rebuild_cache()`: regenerates the flattened candidate spawn
 *      lattice if `_is_invalidated_cache` is set (see
 *      `detail::SourceCacheBuilder`) — a no-op most calls, since the
 *      cache only needs rebuilding when unit geometry changes.
 *   2. Computes how many particles can actually be emitted this step:
 *      `min(cached candidate count, remaining fluid buffer capacity)`
 *      — logs a warning and emits nothing if the fluid buffer is full.
 *   3. `shuffle_species(emit_count)`: randomizes which cached candidate
 *      gets which species for this call (see
 *      `detail::SourceSpeciesShuffler`).
 *   4. `make_probe()` + `detail::SourceEmitter::emit`: writes the new
 *      particles into the fluid's tail slots (positions from the cache,
 *      velocities sampled per species) and grows
 *      `Fluid::particle_count()` by `emit_count`.
 * If an `ObserverHostPtr` with a `SourceSensorMetrics` sensor is
 * attached, per-unit emission counts are recorded each step.
 */

namespace atlas {

/**
 * @brief Owns a scene's inflow boundaries and injects matching new
 *        particles into a `Fluid` each timestep. See this file's
 *        top-of-file documentation for the cache/emission pipeline.
 */
class Source final {
public:
    class Builder;

public:
    Source() = default;

    Source(const Source&) = delete;

    Source(Source&&) noexcept = default;

    ~Source() = default;

    Source&
    operator=(const Source&)
        = delete;

    Source&
    operator=(Source&&) noexcept = default;

    ATLAS_HOST
    Source(UniverseHostPtr universe,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<Spawn> spawn_operators,
           FluidHostPtr fluid,
           bool flip                = false,
           float spacing            = 0.1f,
           float tolerance          = 0.0f,
           float temperature        = 273.15f,
           ObserverHostPtr observer = nullptr) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /** @brief Advances every unit's own motion by `dt`, then calls
     *  `emit()`. No-op for an empty source or `dt <= 0`. */
    ATLAS_HOST void
    update(float dt);

    /** @brief Full per-step emission pipeline (see this file's
     *  top-of-file documentation): rebuild cache, size the emission
     *  batch to available buffer capacity, shuffle species, write new
     *  particles. */
    ATLAS_HOST void
    emit();

    /** @brief Regenerates the flattened candidate spawn lattice
     *  (`detail::SourceCacheBuilder::rebuild`) if the cache is
     *  currently invalidated; no-op otherwise. */
    ATLAS_HOST void
    rebuild_cache() noexcept;

    /** @brief Randomizes species assignment for the first `count`
     *  cached candidates (`detail::SourceSpeciesShuffler::shuffle`). */
    ATLAS_HOST void
    shuffle_species(std::size_t count);

    /** @brief Rebuilds `_probe` from the current cache/fluid state
     *  (`Source::make_probe`). */
    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

private:
    UniverseHostPtr _universe;

    DeviceBuffer<SpawnType> _spawn_types;

    DeviceBuffer<Spawn> _spawn_operators;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    bool _flip = false;

    float _spacing = 0.1f;

    float _tolerance = 0.0f;

    float _temperature { 273.15f };

    HostBuffer<int> _local_unit_counts;

    DeviceBuffer<Float3> _flat_local_positions;

    DeviceBuffer<int> _flat_unit_indices;

    std::size_t _local_particle_count = 0;

    DeviceBuffer<std::size_t> _species_cache;

    DeviceBuffer<std::size_t> _shuffled_species;

    DeviceBuffer<std::uint64_t> _shuffle_keys;

    SourceProbe _probe {};

    std::uint64_t _shuffle_seed = 0;

    bool _is_invalidated_cache = true;

    std::size_t _step_index = 0;

    detail::SourceCacheBuilder _cache_builder {};

    detail::SourceSpeciesShuffler _species_shuffler {};

    detail::SourceEmitter _emitter {};
};

/**
 * @brief Fluent builder for `Source`. Validation requires non-empty
 *        `_units`, a non-null `_fluid`, non-empty `_spawn_types`/
 *        `_spawn_operators` each sized `1` or matching the unit count,
 *        and finite `_spacing` (positive), `_tolerance`, `_temperature`.
 */
class Source::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Source
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Source>
    make_host_shared();

    /** @brief The domain that owns this source's units (registered via
     *  `Universe::Builder::with_source_units`); required, non-null. */
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    ATLAS_HOST Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    ATLAS_HOST Builder&
    with_spawn_operator(const Spawn& spawn_operator) noexcept;

    ATLAS_HOST Builder&
    with_spawn_operators(const HostBuffer<Spawn>& spawn_operators);

    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST Builder&
    with_spacing(float spacing) noexcept;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    HostBuffer<SpawnType> _spawn_types;

    HostBuffer<Spawn> _spawn_operators;

    bool _flip = false;

    float _spacing = 0.1f;

    float _tolerance = 0.0f;

    float _temperature { 273.15f };
};

using SourceHostPtr = atlas::host_shared_ptr<Source>;

using SourceDevicePtr = atlas::device_shared_ptr<Source>;

}
