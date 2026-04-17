#pragma once

/**
 * @file dsmc_disjoint_pair_solver.h
 * @brief Declares a DSMC solver variant that only processes disjoint local pairs per cell.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcDisjointPairSolver final : public DsmcSolver<T> {
public:
    class Builder;

public:
    DsmcDisjointPairSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcDisjointPairSolver(UniverseHostPtr<T> universe,
                           FluidHostPtr<T> fluid,
                           SpatialHashingSearcherHostPtr<T> searcher,
                           DsmcKernelType kernel_type = DsmcKernelType::hard_sphere,
                           T collision_rate_scale = T(1)) noexcept;

    ~DsmcDisjointPairSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

template <typename T>
class DsmcDisjointPairSolver<T>::Builder final {
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

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collision_rate_scale(T collision_rate_scale) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcDisjointPairSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    T _collision_rate_scale { T(1) };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcDisjointPairSolver = atlas::system::DsmcDisjointPairSolver<T>;

template <typename T>
using DsmcDisjointPairSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

template <typename T>
using DsmcDisjointPairSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.hpp>
