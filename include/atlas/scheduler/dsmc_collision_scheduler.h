#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>

namespace atlas::scheduler {

template <typename T>
class DsmcCollisionScheduler {
public:
    using Probe = atlas::system::DsmcProbe<T>;

    DsmcCollisionScheduler() = default;

    virtual ~DsmcCollisionScheduler() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    schedule(const Probe& probe, const atlas::DeviceBuffer<int>* allocated_solver, int index) = 0;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    limits_collision_count() const noexcept { return false; }

protected:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    execute_collision_pair(const Probe& probe,
                           int cell,
                           int local_collision,
                           int begin,
                           int end,
                           int lhs_local,
                           int rhs_local,
                           T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    particle_at(int nth,
                int begin,
                int end,
                int particle_count,
                const int* indices_ptr) noexcept;
};

}

namespace atlas {

template <typename T>
using DsmcCollisionScheduler = atlas::scheduler::DsmcCollisionScheduler<T>;

}

#include <atlas/scheduler/dsmc_collision_scheduler.hpp>
