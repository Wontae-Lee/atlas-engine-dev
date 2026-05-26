#pragma once

#include <atlas/core/macros.h>
#include <atlas/scheduler/dsmc_collision_scheduler.h>

#include <cstdint>

namespace atlas::scheduler {

template <typename T>
class DsmcPiclasScheduler final : public DsmcCollisionScheduler<T> {
public:
    using Probe = atlas::system::DsmcProbe<T>;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    schedule(const Probe& probe, const atlas::DeviceBuffer<int>* allocated_solver, int index) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    limits_collision_count() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    select_pair_offsets(int& lhs_local,
                        int& rhs_local,
                        int local_collision,
                        int count,
                        int selector,
                        std::uint64_t seed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    execute_collision_trial(const Probe& probe,
                            int cell,
                            int local_collision,
                            int begin,
                            int end,
                            int count,
                            T max_sigma_g) noexcept;
};

}

namespace atlas {

template <typename T>
using DsmcPiclasScheduler = atlas::scheduler::DsmcPiclasScheduler<T>;

}

#include <atlas/scheduler/dsmc_piclas_scheduler.hpp>
