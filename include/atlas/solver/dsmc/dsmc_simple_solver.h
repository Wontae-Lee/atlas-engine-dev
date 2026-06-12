#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>

namespace atlas {

template <typename T>
class DsmcSimpleSolver final : public DsmcSolver<T> {
public:
    using Base = DsmcSolver<T>;
    using Base::Base;

    class Builder;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

private:
    DsmcSimpleStatistics<T> _simple_statistics {};
};

template <typename T>
class DsmcSimpleSolver<T>::Builder final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcSimpleSolver<T>
    build() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcSimpleSolver<T>>
    make_host_shared() const;

private:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SearcherHostPtr<T> _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };
};

}

namespace atlas {
template <typename T>
using DsmcSimpleSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSimpleSolver<T>>;

template <typename T>
using DsmcSimpleSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSimpleSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_simple_solver.hpp>