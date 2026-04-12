#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcDisjointPairSolver final : public DsmcSolver<T> {
public:
    class Builder;

public:
    DsmcDisjointPairSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcDisjointPairSolver(FluidHostPtr<T> fluid,
                           DsmcKernel<T> op = DsmcKernel<T> {});

    ~DsmcDisjointPairSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) override;

private:
    DeviceBuffer<int> _cell_pair_counts {};
    DeviceBuffer<int> _pair_offsets {};
};

template <typename T>
class DsmcDisjointPairSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_operator(const DsmcKernel<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcDisjointPairSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    FluidHostPtr<T> _fluid {};
    DsmcKernel<T> _operator {};
};

}

namespace atlas {

template <typename T>
using DsmcDisjointPairSolver = atlas::system::DsmcDisjointPairSolver<T>;

template <typename T>
using DsmcDisjointPair = atlas::system::DsmcDisjointPairSolver<T>;

template <typename T>
using DsmcDisjointPairHostPtr = atlas::host_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

template <typename T>
using DsmcDisjointPairDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.hpp>
