#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/universe/universe.h>

namespace atlas {

template <typename T>
class KnudsenCodec final : public Codec<T> {
public:
    class Builder;

    KnudsenCodec() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    KnudsenCodec(UniverseHostPtr<T> domain,
                 FluidHostPtr<T> fluid,
                 SpatialHashingSearcherHostPtr<T> searcher,
                 T characteristic_length,
                 T representative_collision_cross_sectional_area = T(1));

    ~KnudsenCodec() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    encode() override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    decode() override;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    fixed_cell(const CodecProbe<T>& probe, int cell) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    knudsen_number(T particle_count,
                   T statistical_weight,
                   T cell_volume,
                   T characteristic_length,
                   T representative_collision_cross_sectional_area) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    solver_index(T kn, const T* splits, int split_count) noexcept;

private:
    T _characteristic_length = T(1);

    T _representative_collision_cross_sectional_area = T(1);

    DeviceBuffer<T> d_kn_split {};
};

template <typename T>
class KnudsenCodec<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(UniverseHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_length(T characteristic_length) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_representative_collision_cross_sectional_area(
        T representative_collision_cross_sectional_area) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fixed_region(DeviceBuffer<int> fixed_region) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE KnudsenCodec<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<KnudsenCodec<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    UniverseHostPtr<T> _domain {};

    FluidHostPtr<T> _fluid {};

    SpatialHashingSearcherHostPtr<T> _searcher {};

    T _characteristic_length = T(1);

    T _representative_collision_cross_sectional_area = T(1);

    DeviceBuffer<int> _fixed_solver {};

    DeviceBuffer<int> _fixed_region {};
};

}

namespace atlas {

template <typename T>
using KnudsenCodecHostPtr = atlas::host_shared_ptr<KnudsenCodec<T>>;

template <typename T>
using KnudsenCodecDevicePtr = atlas::device_shared_ptr<KnudsenCodec<T>>;

}

#include <atlas/codec/knudsen_codec.hpp>