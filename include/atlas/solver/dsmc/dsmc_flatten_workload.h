#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>

namespace atlas::system {

template <typename T>
class DsmcFlattenWorkload final {
public:
    atlas::DeviceBuffer<int> collision_offsets {};
    atlas::DeviceBuffer<int> collision_cells {};
    atlas::DeviceBuffer<int> filtered_collision_counts {};
    int flattened_collision_count {};

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::DeviceBuffer<int>&
    offsets() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build(int* collision_count_ptr,
          int num_of_cells,
          const int* allocated_solver_ptr = nullptr,
          int index = 0);
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcFlattenWorkload = atlas::system::DsmcFlattenWorkload<T>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_flatten_workload.hpp>
