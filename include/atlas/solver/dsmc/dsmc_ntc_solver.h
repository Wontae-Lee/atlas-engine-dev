#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

template <typename T>
class DsmcNtcSolver final : public DsmcSolver<T> {
public:
    class Builder;

public:
    DsmcNtcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcNtcSolver(FluidHostPtr<T> fluid,
                  DsmcKernel<T> op = DsmcKernel<T> {});

    ~DsmcNtcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T>& domain,
          SpatialHashingProbe<T>& searcher,
          FluidDeviceProbe<T>& particle,
          CodecDeviceProbe<T>& codec) override;

private:
    DeviceBuffer<int> _cell_trial_counts {};
    DeviceBuffer<int> _trial_offsets {};
};

template <typename T>
class DsmcNtcSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_operator(const DsmcKernel<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcNtcSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcNtcSolver<T>>
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
using DsmcNtcSolver = atlas::system::DsmcNtcSolver<T>;

template <typename T>
using DsmcNtc = atlas::system::DsmcNtcSolver<T>;

template <typename T>
using DsmcNtcHostPtr = atlas::host_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

template <typename T>
using DsmcNtcDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_ntc_solver.hpp>
