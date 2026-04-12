#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/indexer/device_pair_indexer.h>
#include <atlas/logging/logging.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class DsmcSolver : public Solver<T> {
    static_assert(std::is_floating_point_v<T>, "DsmcSolver requires a floating-point T");

public:
    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(FluidHostPtr<T> fluid,
               DsmcKernel<T> op = DsmcKernel<T> {});

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) override = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operator(DsmcKernel<T> op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DsmcKernel<T>&
    collision_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernel<T>&
    collision_operator() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(DomainDeviceProbe<T> domain,
                      FluidDeviceProbe<T> particle) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_pair_tables();

protected:
    FluidHostPtr<T> _fluid {};
    DsmcKernel<T> _operator {};
    DeviceBuffer<T> _effective_collision_diameters {};
    DeviceBuffer<T> _effective_viscosity_indices {};
    DeviceBuffer<T> _effective_scattering_parameters {};
    DeviceBuffer<T> _reduced_masses {};
    atlas::DevicePairIndexer<int> _pair_indexer {};
    int _species_count = 0;
};

}

namespace atlas {

template <typename T>
using DsmcHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

template <typename T>
using DsmcDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_solver.hpp>
