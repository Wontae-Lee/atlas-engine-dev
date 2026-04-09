#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/domain/domain.h>

namespace atlas::system {

enum class NeighborSearchRange : int {
    single,
    multiple
};

template <typename T>
struct SpatialHashingProbe {

    Vector3<T> lower_corner {};

    Vector3<int> grid_size { 0, 0, 0 };

    T inv_h = T(1);

    T cell_size = T(1);

    NeighborSearchRange range = NeighborSearchRange::single;

    const int* indices {};

    const int* cell_start {};

    const int* cell_end {};

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid_cell(int ix, int iy, int iz) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    cell_index(int ix, int iy, int iz) const noexcept;

    template <typename Func>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    for_each_neighbor(int p, const Vector3<T>* pos, Func&& func) const;
};

template <typename T>
class SpatialHashingSearcher final {
public:
    class Builder;

    SpatialHashingSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit SpatialHashingSearcher(
        DomainHostPtr<T> domain,
        NeighborSearchRange range = NeighborSearchRange::single);

    ~SpatialHashingSearcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const system::FluidDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingProbe<T>
    make_device_probe() noexcept;

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
    sort_by_key(int active) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

private:
    std::uint64_t _probe_count = 0;

    DomainHostPtr<T> _domain {};

    NeighborSearchRange _range = NeighborSearchRange::single;

    DeviceBuffer<std::uint32_t> d_keys;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<int> d_cell_start;

    DeviceBuffer<int> d_cell_end;

    std::uint32_t* d_keys_ptr = nullptr;

    int* d_indices_ptr = nullptr;

    int* d_cell_start_ptr = nullptr;

    int* d_cell_end_ptr = nullptr;
};

template <typename T>
class SpatialHashingSearcher<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_range(NeighborSearchRange range) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingSearcher<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpatialHashingSearcher<T>>
    make_host_shared() const;

private:
    void
    validate() const;

private:
    DomainHostPtr<T> _domain {};

    NeighborSearchRange _range = NeighborSearchRange::single;
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
