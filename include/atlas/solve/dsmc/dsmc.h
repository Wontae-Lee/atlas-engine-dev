#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/indexer/device_pair_indexer.h>
#include <atlas/logging/logging.h>
#include <atlas/solve/solve.h>
#include <atlas/solve/dsmc/dsmc_operator.h>

#include <type_traits>

namespace atlas::system {

enum class DsmcSolveMode : int {
    disjoint_pair,
    ntc
};

template <typename T>
class Dsmc final : public Solve<T> {
    static_assert(std::is_floating_point_v<T>, "Dsmc requires a floating-point T");

public:
    class Builder;

public:
    Dsmc() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Dsmc(FluidHostPtr<T> fluid,
         DsmcOperator<T> op = DsmcOperator<T> {},
         DsmcSolveMode mode = DsmcSolveMode::disjoint_pair);

    ~Dsmc() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operator(DsmcOperator<T> op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_solve_mode(DsmcSolveMode mode) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DsmcOperator<T>&
    collision_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcOperator<T>&
    collision_operator() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcSolveMode
    solve_mode() const noexcept;


private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(DomainDeviceProbe<T> domain,
                      FluidDeviceProbe<T> particle) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve_disjoint_pair(DomainDeviceProbe<T> domain,
                        SpatialHashingProbe<T> searcher,
                        FluidDeviceProbe<T> particle) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve_ntc(DomainDeviceProbe<T> domain,
              SpatialHashingProbe<T> searcher,
              FluidDeviceProbe<T> particle) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_pair_tables();

private:
    FluidHostPtr<T> _fluid {};
    DsmcOperator<T> _operator {};
    DsmcSolveMode _solve_mode = DsmcSolveMode::disjoint_pair;
    DeviceBuffer<int> _cell_pair_counts {};
    DeviceBuffer<int> _pair_offsets {};
    DeviceBuffer<int> _cell_trial_counts {};
    DeviceBuffer<int> _trial_offsets {};
    DeviceBuffer<T> _effective_collision_diameters {};
    DeviceBuffer<T> _effective_viscosity_indices {};
    DeviceBuffer<T> _effective_scattering_parameters {};
    DeviceBuffer<T> _reduced_masses {};
    atlas::DevicePairIndexer<int> _pair_indexer {};
    int _species_count = 0;
};

template <typename T>
class Dsmc<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_operator(const DsmcOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solve_mode(DsmcSolveMode mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Dsmc<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Dsmc<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    FluidHostPtr<T> _fluid {};
    DsmcOperator<T> _operator {};
    DsmcSolveMode _solve_mode = DsmcSolveMode::disjoint_pair;
};

}

namespace atlas {

template <typename T>
using Dsmc = atlas::system::Dsmc<T>;

template <typename T>
using DsmcHostPtr = atlas::host_shared_ptr<atlas::system::Dsmc<T>>;

template <typename T>
using DsmcDevicePtr = atlas::device_shared_ptr<atlas::system::Dsmc<T>>;

using DsmcSolveMode = atlas::system::DsmcSolveMode;

}

#include <atlas/solve/dsmc/dsmc.hpp>
