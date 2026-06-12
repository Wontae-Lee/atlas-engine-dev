#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <cstddef>

namespace atlas::detail {

/**
 * @brief Compacts fluid state storage using the active mask.
 */
template <typename T>
class SinkFluidCompactor final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compact(const FluidHostPtr<T>& fluid,
            DeviceBuffer<std::size_t>& keep,
            DeviceBuffer<std::size_t>& offsets,
            DeviceBuffer<std::size_t>& compact_indices,
            DeviceBuffer<std::size_t>& total_count_buffer) const;
};

} // namespace atlas::detail

#include <atlas/sink/detail/sink_fluid_compactor.hpp>
