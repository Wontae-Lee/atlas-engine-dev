#pragma once

#include <atlas/core/macros.h>
#include <atlas/workload/dsmc_collision_workload.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

template <typename T>
class DsmcSolver : public Solver<T> {
public:
    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type                                                 = DsmcKernelType::hard_sphere,
               atlas::host_shared_ptr<atlas::workload::DsmcCollisionWorkload<T>> workload = {}) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::host_shared_ptr<atlas::workload::DsmcCollisionWorkload<T>>&
    workload() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset_states();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt);

protected:
    DsmcProbe<T> _probe {};
    DsmcKernel<T> _kernel {};
    atlas::host_shared_ptr<atlas::workload::DsmcCollisionWorkload<T>> _workload {};
    std::uint64_t _collision_seed = 0;
};

}

namespace atlas {

template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_solver.hpp>
