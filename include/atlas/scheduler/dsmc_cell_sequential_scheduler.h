#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/scheduler/dsmc_collision_scheduler.h>

namespace atlas::scheduler {

template <typename T>
class DsmcCellSequentialScheduler final : public DsmcCollisionScheduler<T> {
public:
    using Probe = atlas::system::DsmcProbe<T>;

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
using DsmcCellSequentialScheduler = atlas::scheduler::DsmcCellSequentialScheduler<T>;

}

#include <atlas/scheduler/dsmc_cell_sequential_scheduler.hpp>
