#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/universe/universe.h>

namespace atlas::system {

template <typename T>
class SpatialHashingSearcher final {
public:
    class Builder;

    SpatialHashingSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit SpatialHashingSearcher(
        UniverseHostPtr<T> universe,
        FluidHostPtr<T> fluid);

    ~SpatialHashingSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_buffers(int alive);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int n_active);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_keys(int alive, const Vector3<T>* pos);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int active);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    indices() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    cell_start() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const int*
    cell_end() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    DeviceBuffer<std::uint32_t> d_keys;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<int> d_cell_start;

    DeviceBuffer<int> d_cell_end;
};

template <typename T>
class SpatialHashingSearcher<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingSearcher<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpatialHashingSearcher<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};
};

}

namespace atlas {

template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;

template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;

template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;

}

#include <atlas/searcher/spatial_hashing_searcher.hpp>
