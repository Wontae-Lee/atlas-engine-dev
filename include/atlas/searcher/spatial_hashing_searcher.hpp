#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

#include <cmath>

namespace atlas::system {

template <typename T>
bool
SpatialHashingProbe<T>::is_valid_cell(const int ix, const int iy, const int iz) const noexcept {

    return (ix >= 0 && ix < grid_size.x && iy >= 0 && iy < grid_size.y && iz >= 0 && iz < grid_size.z);
}

template <typename T>
int
SpatialHashingProbe<T>::cell_index(const int ix, const int iy, const int iz) const noexcept {

    return ix + iy * grid_size.x + iz * grid_size.x * grid_size.y;
}

template <typename T>
template <typename Func>
void
SpatialHashingProbe<T>::for_each_neighbor(int p, const Vector3<T>* pos, Func&& func) const {

    const Vector3<T> xp = pos[p];

    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };

    if (range == NeighborSearchRange::single) {

        const Vector3<T> rel = (xp - lower_corner) * inv_h;

        Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();
        ijk              = atlas::math::clamp(ijk, lo, hi);

        const int cell  = cell_index(ijk.x, ijk.y, ijk.z);
        const int begin = cell_start[cell];
        if (begin < 0) return;

        const int end = cell_end[cell];
        for (int k = begin; k < end; ++k) {
            const int q = indices[k];
            if (q == p) continue;
            if (func(q)) return;
        }
        return;
    }

    const T r  = T(0.5) * cell_size;
    const T r2 = r * r;

    const Vector3<T> rel_min = (xp - Vector3<T> { r, r, r } - lower_corner) * inv_h;
    const Vector3<T> rel_max = (xp + Vector3<T> { r, r, r } - lower_corner) * inv_h;

    Vector3<int> ijk_min = atlas::math::floor(rel_min).template cast_to<int>();
    Vector3<int> ijk_max = atlas::math::floor(rel_max).template cast_to<int>();

    ijk_min = atlas::math::clamp(ijk_min, lo, hi);
    ijk_max = atlas::math::clamp(ijk_max, lo, hi);

    for (int iz = ijk_min.z; iz <= ijk_max.z; ++iz) {
        for (int iy = ijk_min.y; iy <= ijk_max.y; ++iy) {
            for (int ix = ijk_min.x; ix <= ijk_max.x; ++ix) {
                const int cell  = cell_index(ix, iy, iz);
                const int begin = cell_start[cell];
                if (begin < 0) continue;

                const int end = cell_end[cell];
                for (int k = begin; k < end; ++k) {
                    const int q = indices[k];
                    if (q == p) continue;

                    const Vector3<T> dx = pos[q] - xp;
                    const T dist2       = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

                    if (dist2 <= r2) {
                        if (func(q)) return;
                    }
                }
            }
        }
    }
}

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(DomainHostPtr<T> domain,
                                                  const NeighborSearchRange range)
    : _domain(std::move(domain))
    , _range(range) {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher: domain must not be null.";
    reset();
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder
SpatialHashingSearcher<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {

    d_keys.resize(0);
    d_indices.resize(0);

    const auto num_of_cells = _domain->number_of_cells();
    d_cell_start.resize(num_of_cells);
    d_cell_end.resize(num_of_cells);

    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {

    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {

    const auto num_of_cells = _domain->number_of_cells();

    d_keys.resize(static_cast<std::size_t>(alive));
    d_indices.resize(static_cast<std::size_t>(alive));
    d_cell_start.resize(num_of_cells);
    d_cell_end.resize(num_of_cells);

    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(int n_active) {

    int* indices_ptr = d_indices_ptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_active,
        [=](const int i) {
            indices_ptr[i] = i;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(int alive, const Vector3<T>* pos) {

    const atlas::device_ptr<std::uint32_t> keys_ptr(d_keys_ptr);

    const Vector3<T> lc   = _domain->lower_corner();
    const T inv_h         = _domain->inverse_cell_size();
    const Vector3<int> gs = _domain->grid_size();

    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=](int i) {
            const Vector3<T> rel = (pos[i] - lc) * inv_h;

            Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();
            ijk              = atlas::math::clamp(ijk, lo, hi);

            keys_ptr[i] = linear_key(ijk.x, ijk.y, ijk.z, gs);
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) const {

    const atlas::device_ptr<std::uint32_t> keys_begin(d_keys_ptr);
    const atlas::device_ptr<std::uint32_t> keys_end = keys_begin + active;
    const atlas::device_ptr<int> idx_begin(d_indices_ptr);

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(int alive) {

    const auto num_of_cells = _domain->number_of_cells();

    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_start_ptr,
        d_cell_start_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_end_ptr,
        d_cell_end_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    const std::uint32_t* keys = atlas::raw_pointer_cast<std::uint32_t>(d_keys_ptr);
    int* cell_start           = atlas::raw_pointer_cast(d_cell_start_ptr);
    int* cell_end             = atlas::raw_pointer_cast(d_cell_end_ptr);
    const int count           = alive;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=](const int i) {
            const std::uint32_t key = keys[i];
            if (i == 0 || key != keys[i - 1]) cell_start[key] = i;
            if (i == count - 1 || key != keys[i + 1]) cell_end[key] = i + 1;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build(const system::FluidDeviceProbe<T>& particle_probe) {

    const int alive = particle_probe.particle_count;

    if (alive <= 0) {
        reset();
        return;
    }

    atlas::logger::info() << "\n"
                          << "Building SpatialHashingSearcher for " << alive << " particles.";

    prepare_buffers(alive);

    init_indices_iota(alive);

    compute_keys(alive, particle_probe.pos);

    sort_by_key(alive);

    build_cell_ranges(alive);
}

template <typename T>
SpatialHashingProbe<T>
SpatialHashingSearcher<T>::make_device_probe() noexcept {

    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    SpatialHashingProbe<T> probe {};
    if (!_domain) return probe;

    probe.lower_corner = _domain->lower_corner();
    probe.grid_size    = _domain->grid_size();
    probe.inv_h        = _domain->inverse_cell_size();
    probe.cell_size    = _domain->cell_size();

    probe.range = _range;

    probe.indices    = atlas::raw_pointer_cast(d_indices_ptr);
    probe.cell_start = atlas::raw_pointer_cast(d_cell_start_ptr);
    probe.cell_end   = atlas::raw_pointer_cast(d_cell_end_ptr);

    return probe;
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {

    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_range(const NeighborSearchRange range) noexcept {

    _range = range;
    return *this;
}

template <typename T>
void
SpatialHashingSearcher<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher::Builder: domain must not be null.";
}

template <typename T>
SpatialHashingSearcher<T>
SpatialHashingSearcher<T>::Builder::build() const {

    validate();
    return SpatialHashingSearcher<T>(_domain, _range);
}

template <typename T>
atlas::host_shared_ptr<SpatialHashingSearcher<T>>
SpatialHashingSearcher<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain, _range);
}

}
