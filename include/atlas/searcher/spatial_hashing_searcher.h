#ifndef ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_H
#define ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_H

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>
#include <atlas/searcher/searcher.h>

#include <cstdint>

namespace atlas {
namespace system {

    template <typename T>
    struct ParticleDeviceProbe;

    enum class NeighborSearchRange : int {
        SingleCell   = 0,
        MultipleCell = 1
    };

    template <typename T>
    class SpatialHashingSearcher final : public Searcher<T> {
    public:
        SpatialHashingSearcher() noexcept  = default;
        ~SpatialHashingSearcher() override = default;

        void
        build(const system::ParticleDeviceProbe<T>& data, int& active) override;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_cell_size(T h) noexcept {
            cell_size_ = h;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        cell_size() const noexcept {
            return cell_size_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<T>&
        min_corner() const noexcept {
            return min_corner_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<T>&
        max_corner() const noexcept {
            return max_corner_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<int>&
        grid_size() const noexcept {
            return grid_size_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::size_t
        num_cells() const noexcept {
            return static_cast<std::size_t>(grid_size_.x) *
                   static_cast<std::size_t>(grid_size_.y) *
                   static_cast<std::size_t>(grid_size_.z);
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
        device_sorted_indices() const noexcept {
            return d_indices_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
        device_cell_start() const noexcept {
            return d_cell_start_;
        }

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
        device_cell_end() const noexcept {
            return d_cell_end_;
        }

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::uint32_t
        cell_index(const Vector3<int>& c) const noexcept {
            return static_cast<std::uint32_t>(
                c.x +
                grid_size_.x * (c.y + grid_size_.y * c.z));
        }

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<int>
        position_to_cell(const Vector3<T>& p) const noexcept;

        template <typename Func>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        for_each_neighbor(const Vector3<T>& position,
                          T radius,
                          const Vector3<T>* positions,
                          NeighborSearchRange range,
                          Func&& func) const noexcept;

    private:
        T           cell_size_  = T(1);
        Vector3<T>  min_corner_ {};
        Vector3<T>  max_corner_ {};
        Vector3<int> grid_size_ { 0, 0, 0 };

        DeviceBuffer<std::uint32_t> d_keys_;
        DeviceBuffer<int>           d_indices_;
        DeviceBuffer<int>           d_cell_start_;
        DeviceBuffer<int>           d_cell_end_;

        std::uint32_t* d_keys_ptr_       = nullptr;
        int*           d_indices_ptr_    = nullptr;
        int*           d_cell_start_ptr_ = nullptr;
        int*           d_cell_end_ptr_   = nullptr;
    };

} // namespace system

template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;

template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;

template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;

} // namespace atlas

#include <atlas/searcher/spatial_hashing_searcher.hpp>

#endif // ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_H
