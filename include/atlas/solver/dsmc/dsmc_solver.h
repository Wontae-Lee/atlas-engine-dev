#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas {

enum struct DsmcCollisionWorkloadType : int {
    cell,
    flatten
};

template <typename T>
class DsmcSolver : public Solver<T> {
public:
    using Probe = atlas::DsmcProbe<T>;

    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type              = DsmcKernelType::hard_sphere,
               DsmcCollisionWorkloadType workload_type = DsmcCollisionWorkloadType::cell) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcCollisionWorkloadType
    workload_type() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset_states();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         T max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    sample_distinct_pair(int& lhs_local,
                         int& rhs_local,
                         int cell,
                         int count,
                         std::uint64_t seed,
                         std::uint64_t stream) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    particle_at(int nth,
                int begin,
                int end,
                int particle_count,
                const int* indices_ptr) noexcept;

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_flattened_collision(const DeviceBuffer<int>* allocated_solver, int index);

    DsmcProbe<T> _probe {};

    DsmcKernel<T> _kernel {};

    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };

    DsmcFlattenWorkload<T> _flatten_workload {};

    DsmcStatistics<T> _statistics {};

    std::uint64_t _collision_seed = 0;
};

}

namespace atlas {

template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSolver<T>>;

template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSolver<T>>;
}

#include <atlas/solver/dsmc/dsmc_solver.hpp>