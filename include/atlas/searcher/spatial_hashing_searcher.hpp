#pragma once
#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/transform/transform_reduce.h>
#include <cmath>

namespace atlas::system {
template <typename T>
bool
SpatialHashProbe<T>::is_valid_cell(int ix, int iy, int iz) const noexcept {
    return (ix >= 0 && ix < grid_size.x && iy >= 0 && iy < grid_size.y && iz >= 0 && iz < grid_size.z);
}

template <typename T>
int
SpatialHashProbe<T>::cell_index(int ix, int iy, int iz) const noexcept {
    return ix + iy * grid_size.x + iz * grid_size.x * grid_size.y;
}

template <typename T>
template <typename Func>
void
SpatialHashProbe<T>::for_each_neighbor(int p,
                                       const Vector3<T>* pos,
                                       Func&& func) const {
    const Vector3<T> xp = pos[p];
    const T h           = T(1) / inv_h;
    if (range == NeighborSearchRange::single) {
        T rx   = (xp.x - lower_corner.x) * inv_h;
        T ry   = (xp.y - lower_corner.y) * inv_h;
        T rz   = (xp.z - lower_corner.z) * inv_h;
        int ix = static_cast<int>(::floor(rx));
        int iy = static_cast<int>(::floor(ry));
        int iz = static_cast<int>(::floor(rz));
        if (ix < 0) ix = 0;
        else if (ix >= grid_size.x) ix = grid_size.x - 1;
        if (iy < 0) iy = 0;
        else if (iy >= grid_size.y) iy = grid_size.y - 1;
        if (iz < 0) iz = 0;
        else if (iz >= grid_size.z) iz = grid_size.z - 1;
        const int cell  = cell_index(ix, iy, iz);
        const int begin = cell_start[cell];
        if (begin < 0) {
            return;
        }
        const int end = cell_end[cell];
        for (int k = begin; k < end; ++k) {
            const int q = indices[k];
            if (q == p) {
                continue;
            }
            if (func(q)) {
                return;
            }
        }
    } else {
        const T r           = T(0.5) * h;
        const T r2          = r * r;
        const Vector3<T> lc = lower_corner;
        int ix_min          = static_cast<int>(::floor((xp.x - r - lc.x) * inv_h));
        int ix_max          = static_cast<int>(::floor((xp.x + r - lc.x) * inv_h));
        int iy_min          = static_cast<int>(::floor((xp.y - r - lc.y) * inv_h));
        int iy_max          = static_cast<int>(::floor((xp.y + r - lc.y) * inv_h));
        int iz_min          = static_cast<int>(::floor((xp.z - r - lc.z) * inv_h));
        int iz_max          = static_cast<int>(::floor((xp.z + r - lc.z) * inv_h));
        if (ix_min < 0) ix_min = 0;
        if (iy_min < 0) iy_min = 0;
        if (iz_min < 0) iz_min = 0;
        if (ix_max >= grid_size.x) ix_max = grid_size.x - 1;
        if (iy_max >= grid_size.y) iy_max = grid_size.y - 1;
        if (iz_max >= grid_size.z) iz_max = grid_size.z - 1;
        for (int iz = iz_min; iz <= iz_max; ++iz) {
            for (int iy = iy_min; iy <= iy_max; ++iy) {
                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    const int cell  = cell_index(ix, iy, iz);
                    const int begin = cell_start[cell];
                    if (begin < 0) {
                        continue;
                    }
                    const int end = cell_end[cell];
                    for (int k = begin; k < end; ++k) {
                        const int q = indices[k];
                        if (q == p) {
                            continue;
                        }
                        const Vector3<T> dx = pos[q] - xp;
                        const T dist2       = dx.x * dx.x
                            + dx.y * dx.y
                            + dx.z * dx.z;
                        if (dist2 <= r2) {
                            if (func(q)) {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(Vector3<T> lower_corner_,
                                                  Vector3<T> upper_corner_,
                                                  T cell_size_,
                                                  NeighborSearchRange range_)
    : _lower_corner(lower_corner_)
      , _upper_corner(upper_corner_)
      , _cell_size(cell_size_)
      , _range(range_) {
    _cell_volume            = _cell_size * _cell_size * _cell_size;
    const Vector3<T> extent = _upper_corner - _lower_corner;
    const T inv_h           = T(1) / _cell_size;
    int nx                  = static_cast<int>(std::floor(extent.x * inv_h)) + 1;
    int ny                  = static_cast<int>(std::floor(extent.y * inv_h)) + 1;
    int nz                  = static_cast<int>(std::floor(extent.z * inv_h)) + 1;
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;
    if (nz < 1) nz = 1;
    _grid_size = Vector3<int>{ nx, ny, nz };
    _n_cells   = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
        * static_cast<std::size_t>(nz);
    _mode = NeighborSearchMode::passive;
}

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(const Box<T>& box,
                                                  T cell_size_,
                                                  NeighborSearchRange range_)
    : SpatialHashingSearcher(box.lower_corner,
                             box.upper_corner,
                             cell_size_,
                             range_) {}

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(const AABB<T>& aabb,
                                                  T cell_size_,
                                                  NeighborSearchRange range_)
    : SpatialHashingSearcher(aabb.lower_corner,
                             aabb.upper_corner,
                             cell_size_,
                             range_) {}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {
    if (_mode == NeighborSearchMode::active) {
        _grid_size    = Vector3<int>{ 0, 0, 0 };
        _lower_corner = Vector3<T>{ T(inf), T(inf), T(inf) };
        _upper_corner = Vector3<T>{ -T(inf), -T(inf), -T(inf) };
    }
    d_keys.resize(0);
    d_indices.resize(0);
    d_cell_start.resize(_n_cells);
    d_cell_end.resize(_n_cells);
    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
std::size_t
SpatialHashingSearcher<T>::n_cells() const noexcept {
    return _n_cells;
}

template <typename T>
T
SpatialHashingSearcher<T>::cell_size() const noexcept {
    return _cell_size;
}

template <typename T>
T
SpatialHashingSearcher<T>::cell_volume() const noexcept {
    return _cell_volume;
}

template <typename T>
void
SpatialHashingSearcher<T>::readjust(const system::ParticleDeviceProbe<T>& data,
                                    const int& active) noexcept {
    const Vector3<T>* d_pos_raw = data.pos;
    int n_active                = active;
    ATLAS_INFO << "SpatialHashingSearcher: Readjusting spatial hash grid with "
        << n_active << " active particles.";
    atlas::device_ptr<const Vector3<T>> pos_begin(d_pos_raw);
    atlas::device_ptr<const Vector3<T>> pos_end = pos_begin + n_active;
    AABB<T> aabb                                =
        atlas::transform_reduce<ExecutionPolicy::device>(
            pos_begin,
            pos_end,
            aabb,
            [] ATLAS_ALL_DEVICE(const Vector3<T>& p) {
                return spatial::make_aabb<T>(p);
            },
            []
        ATLAS_ALL_DEVICE(const spatial::AxisAlignedBoundingBox<T>& a,

                         const spatial::AxisAlignedBoundingBox<T>& b
            ) {
                return spatial::merge_aabb<T>(a, b);
            }
            );
    const Vector3<T> extent = aabb.extents();

    T inv_h = T(1) / T(_cell_size);
    int nx  = static_cast<int>(std::floor(extent.x * inv_h)) + 1;
    int ny  = static_cast<int>(std::floor(extent.y * inv_h)) + 1;
    int nz  = static_cast<int>(std::floor(extent.z * inv_h)) + 1;
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;
    if (nz < 1) nz = 1;
    _grid_size = Vector3<int>{ nx, ny, nz };
    _n_cells   = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
        * static_cast<std::size_t>(nz);
}

template <typename T>
void
SpatialHashingSearcher<T>::build(const system::ParticleDeviceProbe<T>& data) {
    int n_active = data.alive;
    if (n_active <= 0) {
        this->reset();
        return;
    }
    if (_mode == NeighborSearchMode::active) {
        this->readjust(data, n_active);
    }
    const auto n_cells = static_cast<std::size_t>(_n_cells);
    d_keys.resize(static_cast<std::size_t>(n_active));
    d_indices.resize(static_cast<std::size_t>(n_active));
    d_cell_start.resize(n_cells);
    d_cell_end.resize(n_cells);
    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
    atlas::parallel_fill<ExecutionPolicy::device>(d_indices_ptr, d_indices_ptr + n_active, 0);
    {
        atlas::device_ptr<std::uint32_t> keys_ptr(d_keys_ptr);
        const Vector3<T> lower_corner = _lower_corner;
        const T inv_h                 = T(1) / T(_cell_size);
        const Vector3<int> grid_size  = _grid_size;
        auto pos                      = data.pos;
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            n_active,
            [=] ATLAS_ALL_DEVICE(int i) {
                const Vector3<T>& p = pos[i];
                Vector3<T> rel      = (p - lower_corner) * inv_h;
                int ix              = static_cast<int>(floor(rel.x));
                int iy              = static_cast<int>(floor(rel.y));
                int iz              = static_cast<int>(floor(rel.z));
                const int max_x     = grid_size.x - 1;
                const int max_y     = grid_size.y - 1;
                const int max_z     = grid_size.z - 1;
                if (ix < 0) ix = 0;
                else if (ix > max_x) ix = max_x;
                if (iy < 0) iy = 0;
                else if (iy > max_y) iy = max_y;
                if (iz < 0) iz = 0;
                else if (iz > max_z) iz = max_z;
                keys_ptr[i] = static_cast<std::uint32_t>(
                    ix + iy * grid_size.x + iz * grid_size.x * grid_size.y);
            }
            );
    }
    {
        const atlas::device_ptr<std::uint32_t> keys_begin(d_keys_ptr);
        const atlas::device_ptr<std::uint32_t> keys_end = keys_begin + n_active;
        const atlas::device_ptr<int> idx_begin(d_indices_ptr);
        atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
    }
    {
        const atlas::device_ptr<int> cs_begin(d_cell_start_ptr);
        const atlas::device_ptr<int> cs_end = cs_begin + static_cast<std::ptrdiff_t>(n_cells);
        atlas::device_ptr<int> ce_begin(d_cell_end_ptr);
        const atlas::device_ptr<int> ce_end = ce_begin + static_cast<std::ptrdiff_t>(n_cells);
        atlas::parallel_fill<ExecutionPolicy::device>(cs_begin, cs_end, -1);
        atlas::parallel_fill<ExecutionPolicy::device>(ce_begin, ce_end, -1);
        const std::uint32_t* keys = atlas::raw_pointer_cast<std::uint32_t>(d_keys_ptr);
        int* cell_start           = atlas::raw_pointer_cast(d_cell_start_ptr);
        int* cell_end             = atlas::raw_pointer_cast(d_cell_end_ptr);
        const int count           = n_active;
        atlas::parallel_for<::atlas::ExecutionPolicy::device>(
            0,
            n_active,
            [keys, cell_start, cell_end, count] ATLAS_ALL_DEVICE(int i) {
                const std::uint32_t key = keys[i];
                if (i == 0 || key != keys[i - 1]) {
                    cell_start[key] = i;
                }
                if (i == count - 1 || key != keys[i + 1]) {
                    cell_end[key] = i + 1;
                }
            }
            );
    }
}

template <typename T>
SpatialHashProbe<T>
SpatialHashingSearcher<T>::make_device_probe() const noexcept {
    SpatialHashProbe<T> probe;
    probe.lower_corner = _lower_corner;
    probe.grid_size    = _grid_size;
    probe.inv_h        = T(1) / _cell_size;
    probe.indices      = atlas::raw_pointer_cast(d_indices_ptr);
    probe.cell_start   = atlas::raw_pointer_cast(d_cell_start_ptr);
    probe.cell_end     = atlas::raw_pointer_cast(d_cell_end_ptr);
    probe.range        = _range;
    return probe;
}
}