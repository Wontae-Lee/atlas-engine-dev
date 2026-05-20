#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcFlattenSolver : public DsmcSolver<T> {
public:
    class Builder;
    using Probe = typename DsmcSolver<T>::DsmcSolverProbe;

public:
    DsmcFlattenSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcFlattenSolver(UniverseHostPtr<T> universe,
                      FluidHostPtr<T> fluid,
                      SpatialHashingSearcherHostPtr<T> searcher,
                      DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    ~DsmcFlattenSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data() override;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const Probe& probe, int index, T dt);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    cell_from_collision_index(int work_index,
                              int num_of_cells,
                              const int* collision_offsets_ptr,
                              const int* collision_count_ptr) noexcept;

private:
    DeviceBuffer<int> _collision_offsets {};
    int _flattened_collision_count {};
};

template <typename T>
class DsmcFlattenSolver<T>::Builder final {
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

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcFlattenSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcFlattenSolver<T>>
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
using DsmcFlattenSolver = atlas::system::DsmcFlattenSolver<T>;

template <typename T>
using DsmcFlattenSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

template <typename T>
using DsmcFlattenSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_flatten_solver.hpp>
