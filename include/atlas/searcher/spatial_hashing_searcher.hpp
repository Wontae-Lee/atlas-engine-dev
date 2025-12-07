#ifndef ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_HPP
#define ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <atlas/logging/logging.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>

#include <cuda_runtime.h>

#include <atlas/parallel/parallel_for.h> // ★ atlas parallel_for

namespace atlas::system {


// ---------------------------------------------------------------------
// AABB reduction helpers
// ---------------------------------------------------------------------
template <typename T>
struct DeviceBBox {
    Vector3<T> min_p;
    Vector3<T> max_p;
};

template <typename T>
struct BBoxUnaryOp {
    __host__ __device__
    DeviceBBox<T>
    operator()(const Vector3<T>& p) const noexcept {
        DeviceBBox<T> box;
        box.min_p = p;
        box.max_p = p;
        return box;
    }
};

template <typename T>
struct BBoxBinaryOp {
    __host__ __device__
    DeviceBBox<T>
    operator()(const DeviceBBox<T>& a,
               const DeviceBBox<T>& b) const noexcept {
        DeviceBBox<T> out;
        out.min_p = Vector3<T> {
            a.min_p.x < b.min_p.x ? a.min_p.x : b.min_p.x,
            a.min_p.y < b.min_p.y ? a.min_p.y : b.min_p.y,
            a.min_p.z < b.min_p.z ? a.min_p.z : b.min_p.z
        };
        out.max_p = Vector3<T> {
            a.max_p.x > b.max_p.x ? a.max_p.x : b.max_p.x,
            a.max_p.y > b.max_p.y ? a.max_p.y : b.max_p.y,
            a.max_p.z > b.max_p.z ? a.max_p.z : b.max_p.z
        };
        return out;
    }
};

// ---------------------------------------------------------------------
// position_to_cell
// ---------------------------------------------------------------------
template <typename T>
Vector3<int>
SpatialHashingSearcher<T>::position_to_cell(const Vector3<T>& p) const noexcept {

    if (grid_size_.x == 0 || grid_size_.y == 0 || grid_size_.z == 0) {
        return Vector3<int> { 0, 0, 0 };
    }

    const T inv_h  = T(1) / cell_size_;
    Vector3<T> rel = (p - min_corner_) * inv_h;

    int ix = static_cast<int>(std::floor(rel.x));
    int iy = static_cast<int>(std::floor(rel.y));
    int iz = static_cast<int>(std::floor(rel.z));

    const int max_x = grid_size_.x - 1;
    const int max_y = grid_size_.y - 1;
    const int max_z = grid_size_.z - 1;

    if (ix < 0) ix = 0;
    else if (ix > max_x)
        ix = max_x;

    if (iy < 0) iy = 0;
    else if (iy > max_y)
        iy = max_y;

    if (iz < 0) iz = 0;
    else if (iz > max_z)
        iz = max_z;

    return Vector3<int> { ix, iy, iz };
}

// ---------------------------------------------------------------------
// Key computation helpers
// ---------------------------------------------------------------------
template <typename T>
struct ComputeKeyFunctor {
    Vector3<T>   min_corner;
    T            inv_h;
    Vector3<int> grid_size;

    __host__ __device__
    std::uint32_t
    operator()(int i,
               const Vector3<T>* positions) const noexcept {
        const Vector3<T>& p = positions[i];
        Vector3<T> rel      = (p - min_corner) * inv_h;

        int ix = static_cast<int>(std::floor(rel.x));
        int iy = static_cast<int>(std::floor(rel.y));
        int iz = static_cast<int>(std::floor(rel.z));

        const int max_x = grid_size.x - 1;
        const int max_y = grid_size.y - 1;
        const int max_z = grid_size.z - 1;

        if (ix < 0) ix = 0;
        else if (ix > max_x)
            ix = max_x;

        if (iy < 0) iy = 0;
        else if (iy > max_y)
            iy = max_y;

        if (iz < 0) iz = 0;
        else if (iz > max_z)
            iz = max_z;

        return static_cast<std::uint32_t>(
            ix + grid_size.x * (iy + grid_size.y * iz));
    }
};

template <typename T>
struct ComputeKeyWrapper {
    const Vector3<T>* positions;
    ComputeKeyFunctor<T> core;

    __host__ __device__
    std::uint32_t
    operator()(int i) const noexcept {
        return core(i, positions);
    }
};

// ---------------------------------------------------------------------
// build (Thrust-only, GPU + atlas::parallel_for)
// ---------------------------------------------------------------------
template <typename T>
void
SpatialHashingSearcher<T>::build(const system::ParticleDeviceProbe<T>& data,
                                 int& active) {

    ATLAS_INFO << "SpatialHashingSearcher: Building spatial hash grid with "
               << active << " active particles." ;
    int n_active      = active;

    if (n_active == 0) {
        grid_size_  = Vector3<int> { 0, 0, 0 };
        min_corner_ = Vector3<T> {};
        max_corner_ = Vector3<T> {};

        d_keys_.resize(0);
        d_indices_.resize(0);
        d_cell_start_.resize(0);
        d_cell_end_.resize(0);

        d_keys_ptr_       = nullptr;
        d_indices_ptr_    = nullptr;
        d_cell_start_ptr_ = nullptr;
        d_cell_end_ptr_   = nullptr;

        active = 0;
        return;
    }

    const Vector3<T>* d_pos_raw = data.pos;

    // 1) Bounding box on device
    {
        thrust::device_ptr<const Vector3<T>> pos_begin(d_pos_raw);
        thrust::device_ptr<const Vector3<T>> pos_end = pos_begin + n_active;

        const T inf = std::numeric_limits<T>::infinity();
        DeviceBBox<T> init;
        init.min_p = Vector3<T> { inf, inf, inf };
        init.max_p = Vector3<T> { -inf, -inf, -inf };

        DeviceBBox<T> result = thrust::transform_reduce(
            pos_begin,
            pos_end,
            BBoxUnaryOp<T> {},
            init,
            BBoxBinaryOp<T> {});

        min_corner_ = result.min_p;
        max_corner_ = result.max_p;

        const T eps = std::numeric_limits<T>::epsilon() * T(4);
        const Vector3<T> veps { eps, eps, eps };
        min_corner_ -= veps;
        max_corner_ += veps;
    }

    // 2) Grid resolution
    {
        const Vector3<T> extent = max_corner_ - min_corner_;
        const T          inv_h  = T(1) / cell_size_;

        int nx = static_cast<int>(std::floor(extent.x * inv_h)) + 1;
        int ny = static_cast<int>(std::floor(extent.y * inv_h)) + 1;
        int nz = static_cast<int>(std::floor(extent.z * inv_h)) + 1;

        if (nx < 1) nx = 1;
        if (ny < 1) ny = 1;
        if (nz < 1) nz = 1;

        grid_size_ = Vector3<int> { nx, ny, nz };
    }

    const std::size_t n_cells = num_cells();

    // 3) Allocate device buffers
    d_keys_.resize(static_cast<std::size_t>(n_active));
    d_indices_.resize(static_cast<std::size_t>(n_active));
    d_cell_start_.resize(n_cells);
    d_cell_end_.resize(n_cells);

    d_keys_ptr_       = d_keys_.data().get();
    d_indices_ptr_    = d_indices_.data().get();
    d_cell_start_ptr_ = d_cell_start_.data().get();
    d_cell_end_ptr_   = d_cell_end_.data().get();

    // 4) Fill particle indices: 0..n_active-1
    {
        thrust::device_ptr<int> idx_begin(d_indices_ptr_);
        thrust::device_ptr<int> idx_end = idx_begin + n_active;
        thrust::sequence(idx_begin, idx_end, 0);
    }

    // 5) Compute cell keys on device
    {
        thrust::device_ptr<std::uint32_t> keys_begin(d_keys_ptr_);

        ComputeKeyWrapper<T> wrapper;
        wrapper.positions       = d_pos_raw;
        wrapper.core.min_corner = min_corner_;
        wrapper.core.inv_h      = T(1) / cell_size_;
        wrapper.core.grid_size  = grid_size_;

        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last(n_active);

        thrust::transform(first, last, keys_begin, wrapper);
    }

    // 6) Sort (key, index) by key
    {
        thrust::device_ptr<std::uint32_t> keys_begin(d_keys_ptr_);
        thrust::device_ptr<std::uint32_t> keys_end = keys_begin + n_active;
        thrust::device_ptr<int>           idx_begin(d_indices_ptr_);

        thrust::sort_by_key(keys_begin, keys_end, idx_begin);
    }

    // 7) Build cell_start_/cell_end_ with atlas::parallel_for
    {
        thrust::device_ptr<int> cs_begin(d_cell_start_ptr_);
        thrust::device_ptr<int> cs_end   = cs_begin + static_cast<std::ptrdiff_t>(n_cells);
        thrust::device_ptr<int> ce_begin(d_cell_end_ptr_);
        thrust::device_ptr<int> ce_end   = ce_begin + static_cast<std::ptrdiff_t>(n_cells);

        // Initialize to "empty"
        thrust::fill(cs_begin, cs_end, -1);
        thrust::fill(ce_begin, ce_end, -1);

        const std::uint32_t* keys       = d_keys_ptr_;
        int*                 cell_start = d_cell_start_ptr_;
        int*                 cell_end   = d_cell_end_ptr_;
        const int            count      = n_active;

        ::atlas::parallel_for<::atlas::ExecutionPolicy::device>(
            0, n_active,
            [keys, cell_start, cell_end, count] __host__ __device__ (int i) {
                const std::uint32_t key = keys[i];

                if (i == 0 || key != keys[i - 1]) {
                    cell_start[key] = i;
                }

                if (i == count - 1 || key != keys[i + 1]) {
                    cell_end[key] = i + 1;
                }
            });
    }

    active = n_active;
}

// ---------------------------------------------------------------------
// for_each_neighbor (device-side, NeighborSearchRange 지원)
// ---------------------------------------------------------------------
template <typename T>
template <typename Func>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
SpatialHashingSearcher<T>::for_each_neighbor(const Vector3<T>& position,
                                             T radius,
                                             const Vector3<T>* positions,
                                             NeighborSearchRange range,
                                             Func&& func) const noexcept {

#ifndef __CUDA_ARCH__
    // Host with device pointers: no-op (undefined otherwise).
    (void)position;
    (void)radius;
    (void)positions;
    (void)range;
    (void)func;
    return;
#else
    if (positions == nullptr) return;
    if (grid_size_.x == 0 || grid_size_.y == 0 || grid_size_.z == 0) return;

    const T r2    = radius * radius;
    const T inv_h = T(1) / cell_size_;

    const Vector3<int> c0 = position_to_cell(position);

    int cx0, cy0, cz0, cx1, cy1, cz1;

    if (range == NeighborSearchRange::SingleCell) {

        cx0 = cx1 = c0.x;
        cy0 = cy1 = c0.y;
        cz0 = cz1 = c0.z;
    } else {

        int r_cell = static_cast<int>(::ceil(static_cast<double>(radius * inv_h)));
        if (r_cell < 0) r_cell = 0;

        const int min_x = 0;
        const int min_y = 0;
        const int min_z = 0;
        const int max_x = grid_size_.x - 1;
        const int max_y = grid_size_.y - 1;
        const int max_z = grid_size_.z - 1;

        cx0 = c0.x - r_cell;
        cy0 = c0.y - r_cell;
        cz0 = c0.z - r_cell;
        cx1 = c0.x + r_cell;
        cy1 = c0.y + r_cell;
        cz1 = c0.z + r_cell;

        if (cx0 < min_x) cx0 = min_x;
        if (cy0 < min_y) cy0 = min_y;
        if (cz0 < min_z) cz0 = min_z;
        if (cx1 > max_x) cx1 = max_x;
        if (cy1 > max_y) cy1 = max_y;
        if (cz1 > max_z) cz1 = max_z;
    }

    const int*           cell_start = d_cell_start_ptr_;
    const int*           cell_end   = d_cell_end_ptr_;
    const int*           sorted_idx = d_indices_ptr_;
    const Vector3<T>*    pos_ptr    = positions;

    for (int iz = cz0; iz <= cz1; ++iz) {
        for (int iy = cy0; iy <= cy1; ++iy) {
            for (int ix = cx0; ix <= cx1; ++ix) {
                const std::uint32_t cell_id = static_cast<std::uint32_t>(
                    ix +
                    grid_size_.x * (iy + grid_size_.y * iz));

                const int s = cell_start[cell_id];
                if (s < 0) continue;

                const int e = cell_end[cell_id];
                for (int k = s; k < e; ++k) {
                    const int idx       = sorted_idx[k];
                    const Vector3<T>& q = pos_ptr[idx];

                    const Vector3<T> d = position - q;
                    const T          dist2 = d.length_squared();
                    if (dist2 <= r2) {
                        func(idx);
                    }
                }
            }
        }
    }
#endif
}

} // namespace atlas::system

#endif // ATLAS_ENGINE_DEV_SPATIAL_HASHING_SEARCHER_HPP
