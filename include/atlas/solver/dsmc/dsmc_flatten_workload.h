#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>

namespace atlas {

class DsmcFlattenWorkload final {
public:
    atlas::DeviceBuffer<int> collision_offsets {};

    atlas::DeviceBuffer<int> collision_cells {};

    atlas::DeviceBuffer<int> filtered_collision_counts {};

    atlas::DeviceBuffer<int> total_count_buffer {};

    int flattened_collision_count {};

    ATLAS_NODISCARD ATLAS_HOST const atlas::DeviceBuffer<int>&
    offsets() const noexcept;

    ATLAS_HOST void
    clear();

    ATLAS_HOST bool
    build(int* collision_count_ptr,
          int cell_count,
          const int* allocated_solver_ptr = nullptr,
          int index                       = 0);
};

}
