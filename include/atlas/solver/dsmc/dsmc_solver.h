#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the common DSMC solver base class shared by DSMC variants.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

template <typename T>
class DsmcSolver : public Solver<T> {
public:
    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type = DsmcKernelType::hard_sphere,
               T collision_rate_scale = T(1)) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) final;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) final;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_rate_scale() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    flattened_collision_cells() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_collision_workload(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_collision_context() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload() noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    nth_valid_particle(int nth,
                       int begin,
                       int end,
                       int particle_count,
                       const int* indices_ptr) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    pair_ordinal_to_rhs(int count, int ordinal, int& lhs_local) noexcept;

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt)
        = 0;

protected:
    DsmcKernel<T> _kernel {};

    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    DeviceBuffer<int> _collision_offsets {};

    DeviceBuffer<int> _flattened_collision_cells {};

    T _collision_rate_scale { T(1) };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_solver.hpp>
