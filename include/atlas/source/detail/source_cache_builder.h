#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/source/spawn.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

/**
 * @file source_cache_builder.h
 * @brief Host-side generator of the candidate spawn-position lattice
 *        `Source` emits particles from, cached until explicitly
 *        invalidated so the (relatively expensive, host-loop-based)
 *        lattice generation does not re-run every emission step.
 *
 * @details
 * ### Operating principle
 * For each unit, `rebuild()`:
 * 1. Computes a regular Cartesian lattice of candidate points spanning
 *    the unit's local-space bounding box, spaced `spacing` apart along
 *    each axis (`atlas::sample_axis_count` gives the point count per
 *    axis for a given span/spacing).
 * 2. Tests every lattice point against the unit's `Spawn`
 *    (surface or volume containment, see `spawn.h`), keeping
 *    only the accepted ones (inverted if `flip` is set — same
 *    keep/reject inversion convention as `Sink`'s `flip`). This is
 *    plain grid sampling, not Poisson-disk or other blue-noise sampling
 *    — points are exactly `spacing` apart along axes, so density is
 *    uniform but not randomized.
 * 3. Flattens every unit's accepted candidates into one contiguous
 *    `flat_local_positions`/`flat_unit_indices` pair (parallel arrays:
 *    position in the owning unit's local frame, and which unit that
 *    is), and assigns each candidate a species round-robin
 *    (`i % species_count`) so accepted candidates are spread evenly
 *    across species before any shuffling happens (see
 *    `SourceSpeciesShuffler`, which randomizes *which* candidate gets
 *    *which* species assignment per emission, without changing this
 *    even round-robin balance in aggregate).
 * This entire computation runs once per cache rebuild (host-orchestrated,
 * `Source::rebuild_cache()` skips it while `_is_invalidated_cache` is
 * `false`) rather than every emission step, since a unit's geometry
 * rarely changes between emissions — `emit()` then just draws from the
 * cached candidate pool each call.
 */

namespace atlas::detail {

/**
 * @brief Regenerates the flattened candidate spawn-position lattice for
 *        every unit. See this file's top-of-file documentation for the
 *        lattice-generation algorithm and why it is cached.
 */
class SourceCacheBuilder final {
public:
    /**
     * @brief Rebuilds every out-parameter from `units`/`spawn_types`/
     *        `spawn_operators`/`fluid`'s generators/`flip`/`spacing`/
     *        `tolerance`. Clears everything (via `clear()`) if inputs
     *        are missing (empty units/spawn config, null fluid, or a
     *        fluid with no configured generators).
     */
    ATLAS_HOST void
    rebuild(const DeviceBuffer<Unit>& units,
            const DeviceBuffer<SpawnType>& spawn_types,
            const DeviceBuffer<Spawn>& spawn_operators,
            const FluidHostPtr& fluid,
            bool flip,
            float spacing,
            float tolerance,
            HostBuffer<int>& local_unit_counts,
            DeviceBuffer<Vector3>& flat_local_positions,
            DeviceBuffer<int>& flat_unit_indices,
            std::size_t& local_particle_count,
            DeviceBuffer<std::size_t>& species_cache,
            DeviceBuffer<std::size_t>& shuffled_species,
            DeviceBuffer<std::uint64_t>& shuffle_keys,
            std::uint64_t& shuffle_seed) const noexcept;

private:
    ATLAS_HOST void
    clear(HostBuffer<int>& local_unit_counts,
          DeviceBuffer<Vector3>& flat_local_positions,
          DeviceBuffer<int>& flat_unit_indices,
          std::size_t& local_particle_count,
          DeviceBuffer<std::size_t>& species_cache,
          DeviceBuffer<std::size_t>& shuffled_species,
          DeviceBuffer<std::uint64_t>& shuffle_keys,
          std::uint64_t& shuffle_seed) const noexcept;
};

}
