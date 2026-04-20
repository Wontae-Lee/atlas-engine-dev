#pragma once

/**
 * @file dsmc_ntc_solver.h
 * @brief Declares the classic NTC DSMC solver.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcNtcSolver final : public DsmcSolver<T> {
public:
    class Builder;

public:
    DsmcNtcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcNtcSolver(UniverseHostPtr<T> universe,
                  FluidHostPtr<T> fluid,
                  SpatialHashingSearcherHostPtr<T> searcher,
                  DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    ~DsmcNtcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

template <typename T>
class DsmcNtcSolver<T>::Builder final {
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

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcNtcSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcNtcSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

};

} // namespace atlas::system

namespace atlas {

template <typename T>
using DsmcNtcSolver = atlas::system::DsmcNtcSolver<T>;

template <typename T>
using DsmcNtcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

template <typename T>
using DsmcNtcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_ntc_solver.hpp>
