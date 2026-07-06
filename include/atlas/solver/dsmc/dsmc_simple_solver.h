#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>

namespace atlas {

class DsmcSimpleSolver final : public DsmcSolver {
public:
    using Base = DsmcSolver;
    using Base::Base;

    class Builder;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

private:
    DsmcSimpleStatistics _simple_statistics {};
};

class DsmcSimpleSolver::Builder final {
public:
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST Builder&
    with_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST void
    validate() const;

    ATLAS_NODISCARD ATLAS_HOST DsmcSimpleSolver
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<DsmcSimpleSolver>
    make_host_shared() const;

private:
    UniverseHostPtr _universe {};
    FluidHostPtr _fluid {};
    SearcherHostPtr _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };
};

using DsmcSimpleSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSimpleSolver>;

using DsmcSimpleSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSimpleSolver>;

}
