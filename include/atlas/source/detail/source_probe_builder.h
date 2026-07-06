#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/source/source_probe.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

/**
 * @file source_probe_builder.h
 * @brief Host-only factory that assembles a `SourceProbe` from a
 *        `Source`'s cached candidate lattice and its target `Fluid`.
 */

namespace atlas::detail {

/**
 * @brief Populates a `SourceProbe`'s raw-pointer/count fields from live
 *        buffers.
 */
class SourceProbeBuilder final {
public:
    /**
     * @brief Fills `probe` in place from `units`/`shuffled_species`/
     *        `flat_local_positions`/`flat_unit_indices`/`temperature`/
     *        `emission_seed` and the position/velocity/species/active
     *        buffers owned by `fluid`.
     * @return `false` if `fluid` is null or lacks the required particle
     *         states, so callers can skip emission entirely; `true`
     *         otherwise.
     */
    ATLAS_NODISCARD ATLAS_HOST static bool
    make(const FluidHostPtr& fluid,
         const DeviceBuffer<Unit>& units,
         const DeviceBuffer<std::size_t>& shuffled_species,
         const DeviceBuffer<Float3>& flat_local_positions,
         const DeviceBuffer<int>& flat_unit_indices,
         float temperature,
         std::uint64_t emission_seed,
         SourceProbe& probe) noexcept;
};

}
