#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/indexer/device_pair_indexer.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class SphSolver final : public Solver<T> {
    static_assert(std::is_floating_point_v<T>, "SphSolver requires a floating-point T");

public:
    class Builder;

public:
    SphSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SphSolver(FluidHostPtr<T> fluid,
        SphKernel<T> op = SphKernel<T> {});

    ~SphSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operator(SphKernel<T> op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SphKernel<T>&
    interaction_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernel<T>&
    interaction_operator() noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_pair_tables();

private:
    FluidHostPtr<T> _fluid {};
    SphKernel<T> _operator {};
    DeviceBuffer<T> _rest_densities {};
    DeviceBuffer<T> _pressure_coefficients {};
    DeviceBuffer<T> _dynamic_viscosities {};
    atlas::DevicePairIndexer<int> _pair_indexer {};
    int _species_count = 0;
};

template <typename T>
class SphSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_operator(const SphKernel<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SphSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    FluidHostPtr<T> _fluid {};
    SphKernel<T> _operator {};
};

}

namespace atlas {

template <typename T>
using Sph = atlas::system::SphSolver<T>;

template <typename T>
using SphHostPtr = atlas::host_shared_ptr<atlas::system::SphSolver<T>>;

template <typename T>
using SphDevicePtr = atlas::device_shared_ptr<atlas::system::SphSolver<T>>;

}

#include <atlas/solver/sph/sph_solver.hpp>
