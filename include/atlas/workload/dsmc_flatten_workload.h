#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/workload/dsmc_collision_workload.h>

namespace atlas::workload {

template <typename T>
class DsmcFlattenWorkload final : public DsmcCollisionWorkload<T> {
public:
    using Probe = atlas::system::DsmcProbe<T>;

    atlas::DeviceBuffer<int> collision_offsets {};
    atlas::DeviceBuffer<int> collision_cells {};
    int flattened_collision_count {};

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::DeviceBuffer<int>&
    offsets() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build(int* collision_count_ptr, int num_of_cells);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    schedule(const Probe& probe, const atlas::DeviceBuffer<int>* allocated_solver, int index) override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    execute_collision_trial(const Probe& probe,
                            int cell,
                            int local_collision,
                            int begin,
                            int end,
                            int count,
                            T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    select_pair_offsets(const Probe& probe,
                        int& lhs_local,
                        int& rhs_local,
                        int cell,
                        int local_collision,
                        int count) noexcept;
};

}

namespace atlas {

template <typename T>
using DsmcFlattenWorkload = atlas::workload::DsmcFlattenWorkload<T>;

}

#include <atlas/workload/dsmc_flatten_workload.hpp>
