#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/indexer/device_pair_indexer.h>
#include <atlas/logging/logging.h>
#include <atlas/solve/solve.h>
#include <atlas/solve/sph/sph_operator.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Sph final : public Solve<T> {
    static_assert(std::is_floating_point_v<T>, "Sph requires a floating-point T");

public:
    class Builder;

public:
    Sph() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Sph(FluidHostPtr<T> fluid,
        SphOperator<T> op = SphOperator<T> {});

    ~Sph() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operator(SphOperator<T> op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SphOperator<T>&
    interaction_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphOperator<T>&
    interaction_operator() noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_pair_tables();

private:
    FluidHostPtr<T> _fluid {};
    SphOperator<T> _operator {};
    DeviceBuffer<int> _cell_pair_counts {};
    DeviceBuffer<int> _pair_offsets {};
    DeviceBuffer<T> _rest_densities {};
    DeviceBuffer<T> _pressure_coefficients {};
    DeviceBuffer<T> _dynamic_viscosities {};
    DeviceBuffer<T> _smoothing_lengths {};
    atlas::DevicePairIndexer<int> _pair_indexer {};
    int _species_count = 0;
};

template <typename T>
class Sph<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_operator(const SphOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Sph<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sph<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    FluidHostPtr<T> _fluid {};
    SphOperator<T> _operator {};
};

}

namespace atlas {

template <typename T>
using Sph = atlas::system::Sph<T>;

template <typename T>
using SphHostPtr = atlas::host_shared_ptr<atlas::system::Sph<T>>;

template <typename T>
using SphDevicePtr = atlas::device_shared_ptr<atlas::system::Sph<T>>;

}

#include <atlas/solve/sph/sph.hpp>
