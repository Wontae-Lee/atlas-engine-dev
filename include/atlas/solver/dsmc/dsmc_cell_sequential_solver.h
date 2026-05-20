#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcCellSequentialSolver final : public DsmcSolver<T> {
public:
    class Builder;
    using Probe = typename DsmcSolver<T>::DsmcSolverProbe;

public:
    DsmcCellSequentialSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcCellSequentialSolver(UniverseHostPtr<T> universe,
                             FluidHostPtr<T> fluid,
                             SpatialHashingSearcherHostPtr<T> searcher,
                             DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    ~DsmcCellSequentialSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

template <typename T>
class DsmcCellSequentialSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcCellSequentialSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcCellSequentialSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SpatialHashingSearcherHostPtr<T> _searcher {};
    DsmcKernel<T> _kernel {};
};

}

namespace atlas {

template <typename T>
using DsmcCellSequentialSolver = atlas::system::DsmcCellSequentialSolver<T>;

template <typename T>
using DsmcCellSequentialSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcCellSequentialSolver<T>>;

template <typename T>
using DsmcCellSequentialSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcCellSequentialSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_cell_sequential_solver.hpp>
