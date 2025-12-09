#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/box.h>
#include <atlas/math/math.h>
#include <atlas/searcher/searcher.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <cstdint>

namespace atlas {
namespace system {
    template <typename T>
    struct SpatialHashProbe {
        Vector3<T> lower_corner;
        Vector3<int> grid_size;
        T inv_h;
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
    class SpatialHashingSearcher final : public Searcher<T> {
    public:
        SpatialHashingSearcher() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE
        SpatialHashingSearcher(Vector3<T> lower_corner_,
                               Vector3<T> upper_corner_,
                               T cell_size_,
                               NeighborSearchRange range_ = NeighborSearchRange::single);
        ATLAS_HOST ATLAS_FORCE_INLINE
        SpatialHashingSearcher(const Box<T>& box,
                               T cell_size_,
                               NeighborSearchRange range_ = NeighborSearchRange::single);
        ATLAS_HOST ATLAS_FORCE_INLINE
        SpatialHashingSearcher(const AABB<T>& aabb,
                               T cell_size_,
                               NeighborSearchRange range_ = NeighborSearchRange::single);
        ~SpatialHashingSearcher() override = default;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        build(const system::ParticleDeviceProbe<T>& data, int& active) override;
        ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashProbe<T>
        make_device_probe() const noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        reset() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        readjust(const system::ParticleDeviceProbe<T>& data,
                 const int& active) noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::size_t
        n_cells() const noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE T
        cell_size() const noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE T
        cell_volume() const noexcept;

    private:
        Vector3<T> _lower_corner {};
        Vector3<T> _upper_corner {};
        T _cell_size               = T(1);
        T _cell_volume             = T(1);
        NeighborSearchRange _range = NeighborSearchRange::single;
        NeighborSearchMode _mode   = NeighborSearchMode::active;
        size_t _n_cells             = 1;
        Vector3<int> _grid_size { 0, 0, 0 };
        DeviceBuffer<std::uint32_t> d_keys;
        DeviceBuffer<int> d_indices;
        DeviceBuffer<int> d_cell_start;
        DeviceBuffer<int> d_cell_end;
        std::uint32_t* d_keys_ptr = nullptr;
        int* d_indices_ptr        = nullptr;
        int* d_cell_start_ptr     = nullptr;
        int* d_cell_end_ptr       = nullptr;
    };
}

template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;
template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;
template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;
}

#include <atlas/searcher/spatial_hashing_searcher.hpp>