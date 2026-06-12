#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/source/source_probe.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas::detail {

/**
 * @brief Builds device-readable source emission probes.
 */
template <typename T>
class SourceProbeBuilder final {
public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    build(const FluidHostPtr<T>& fluid,
          const DeviceBuffer<Unit<T>>& units,
          const DeviceBuffer<std::size_t>& shuffled_species,
          const DeviceBuffer<Vector3<T>>& flat_local_positions,
          const DeviceBuffer<int>& flat_unit_indices,
          T temperature,
          std::uint64_t emission_seed,
          SourceProbe<T>& probe) const noexcept;
};

} // namespace atlas::detail

#include <atlas/source/detail/source_probe_builder.hpp>
