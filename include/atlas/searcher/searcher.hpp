#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/scan/exclusive_scan.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas {

namespace detail {

    template <typename T, typename CandidateFilter>
    struct SearcherNeighborCount {
        int* counts {};
        const int* indices {};
        const int* start {};
        const int* end {};
        const Vector3<T>* positions {};
        CandidateFilter filter {};
        Vector3<T> lower_corner {};
        T inverse_cell_size {};
        T radius_squared {};
        Vector3<int> grid_size {};

        ATLAS_DEVICE void
        operator()(const int i) const {
            const Vector3<T> pi     = positions[i];
            const Vector3<int> cell = Searcher<T>::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

            int count = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Vector3<int> neighbor_cell = cell + Vector3<int>(dx, dy, dz);
                        if (!Searcher<T>::contains_cell(neighbor_cell, grid_size)) continue;

                        const std::uint32_t key = Searcher<T>::linear_key(neighbor_cell, grid_size);
                        const int first         = start[key];
                        if (first < 0) continue;

                        const int last = end[key];
                        for (int cursor = first; cursor < last; ++cursor) {
                            const int j = indices[cursor];
                            if (j == i) continue;

                            const Vector3<T> pj = positions[j];
                            if (!filter(i, j, pi, pj)) continue;

                            const Vector3<T> delta = pj - pi;
                            if (delta.length_squared() <= radius_squared) {
                                ++count;
                            }
                        }
                    }
                }
            }

            counts[i] = count;
        }
    };

    template <typename T, typename CandidateFilter>
    struct SearcherNeighborWrite {
        int* neighbors {};
        const int* offsets {};
        const int* indices {};
        const int* start {};
        const int* end {};
        const Vector3<T>* positions {};
        CandidateFilter filter {};
        Vector3<T> lower_corner {};
        T inverse_cell_size {};
        T radius_squared {};
        Vector3<int> grid_size {};

        ATLAS_DEVICE void
        operator()(const int i) const {
            const Vector3<T> pi     = positions[i];
            const Vector3<int> cell = Searcher<T>::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

            int write = offsets[i];
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Vector3<int> neighbor_cell = cell + Vector3<int>(dx, dy, dz);
                        if (!Searcher<T>::contains_cell(neighbor_cell, grid_size)) continue;

                        const std::uint32_t key = Searcher<T>::linear_key(neighbor_cell, grid_size);
                        const int first         = start[key];
                        if (first < 0) continue;

                        const int last = end[key];
                        for (int cursor = first; cursor < last; ++cursor) {
                            const int j = indices[cursor];
                            if (j == i) continue;

                            const Vector3<T> pj = positions[j];
                            if (!filter(i, j, pi, pj)) continue;

                            const Vector3<T> delta = pj - pi;
                            if (delta.length_squared() <= radius_squared) {
                                neighbors[write++] = j;
                            }
                        }
                    }
                }
            }
        }
    };

    struct SearcherNeighborTotal {
        int* total {};
        int* offsets {};
        const int* counts {};
        int alive {};
        int last {};

        ATLAS_DEVICE void
        operator()(int) const {
            total[0]       = offsets[last] + counts[last];
            offsets[alive] = total[0];
        }
    };

}

template <typename T>
Searcher<T>::Searcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid)
    : _universe(std::move(universe))
    , _fluid(std::move(fluid)) {
    validate_dependencies("Searcher");
    reset();
}

template <typename T>
void
Searcher<T>::validate_dependencies(const char* owner) const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << owner << ": universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << owner << ": fluid must not be null.";
}

template <typename T>
void
Searcher<T>::invalidate() noexcept {
    _is_invalidated = true;
}

template <typename T>
void
Searcher<T>::reset() noexcept {
    _keys.resize(0);
    _indices.resize(0);
    clear_neighbors();

    const auto num_of_cells = _universe ? _universe->number_of_cells() : std::size_t(0);
    _cell_start.resize(num_of_cells);
    _cell_end.resize(num_of_cells);
    _is_invalidated = true;
}

template <typename T>
Vector3<T>
Searcher<T>::lower_corner() const noexcept {
    return _universe ? _universe->lower_corner() : Vector3<T> {};
}

template <typename T>
Vector3<int>
Searcher<T>::grid_size() const noexcept {
    return _universe ? _universe->grid_size() : Vector3<int> { 0, 0, 0 };
}

template <typename T>
T
Searcher<T>::inverse_cell_size() const noexcept {
    return _universe ? _universe->inverse_cell_size() : T(1);
}

template <typename T>
T
Searcher<T>::cell_size() const noexcept {
    return _universe ? _universe->cell_size() : T(1);
}

template <typename T>
const int*
Searcher<T>::indices() const noexcept {
    return atlas::raw_pointer_cast(_indices.data());
}

template <typename T>
const int*
Searcher<T>::cell_start() const noexcept {
    return atlas::raw_pointer_cast(_cell_start.data());
}

template <typename T>
const int*
Searcher<T>::cell_end() const noexcept {
    return atlas::raw_pointer_cast(_cell_end.data());
}

template <typename T>
const int*
Searcher<T>::neighbor_offsets() const noexcept {
    return atlas::raw_pointer_cast(_neighbor_offsets.data());
}

template <typename T>
const int*
Searcher<T>::neighbor_indices() const noexcept {
    return atlas::raw_pointer_cast(_neighbor_indices.data());
}

template <typename T>
int
Searcher<T>::neighbor_count() const noexcept {
    return _neighbor_count;
}

template <typename T>
std::uint32_t
Searcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {
    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
std::uint32_t
Searcher<T>::linear_key(const Vector3<int>& cell, const Vector3<int>& gs) noexcept {
    return linear_key(cell.x, cell.y, cell.z, gs);
}

template <typename T>
Vector3<int>
Searcher<T>::cell_for(const Vector3<T>& position,
                      const Vector3<T>& lower_corner,
                      const T inverse_cell_size,
                      const Vector3<int>& grid_size) noexcept {
    auto cell = atlas::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();
    return atlas::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
Searcher<T>::contains_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept {
    return atlas::all(cell >= Vector3<int>(0, 0, 0))
        && atlas::all(cell < grid_size);
}

template <typename T>
int
Searcher<T>::search_radius_for(const T length, const T cell_size) noexcept {
    return static_cast<int>(std::ceil(length / cell_size));
}

template <typename T>
const Vector3<T>*
Searcher<T>::position_ptr() const noexcept {
    if (!_fluid) {
        return nullptr;
    }

    const auto* position_state = _fluid->template state<atlas::FluidPositionState<T>>();
    if (!position_state || position_state->data().empty()) {
        return nullptr;
    }

    return atlas::raw_pointer_cast(position_state->data().data());
}

template <typename T>
int
Searcher<T>::active_count() const noexcept {
    return _fluid ? static_cast<int>(_fluid->particle_count()) : 0;
}

template <typename T>
void
Searcher<T>::prepare_grid_buffers(const int alive) {
    const auto num_of_cells = _universe->number_of_cells();
    _keys.resize(static_cast<std::size_t>(alive));
    _indices.resize(static_cast<std::size_t>(alive));
    _cell_start.resize(num_of_cells);
    _cell_end.resize(num_of_cells);
}

template <typename T>
void
Searcher<T>::init_indices_iota(const int alive) {
    auto* indices_ptr = atlas::raw_pointer_cast(_indices.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            indices_ptr[i] = i;
        });
}

template <typename T>
void
Searcher<T>::compute_grid_keys(const int alive, const Vector3<T>* positions) {
    auto* keys_ptr        = atlas::raw_pointer_cast(_keys.data());
    auto* indices_ptr     = atlas::raw_pointer_cast(_indices.data());
    const Vector3<T> lc   = _universe->lower_corner();
    const T inv_h         = _universe->inverse_cell_size();
    const Vector3<int> gs = _universe->grid_size();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<int> ijk = Searcher<T>::cell_for(positions[i], lc, inv_h, gs);
            keys_ptr[i]            = Searcher<T>::linear_key(ijk.x, ijk.y, ijk.z, gs);
            indices_ptr[i]         = i;
        });
}

template <typename T>
void
Searcher<T>::sort_by_key(const int alive) {
    auto* keys_begin = atlas::raw_pointer_cast(_keys.data());
    auto* keys_end   = keys_begin + alive;
    auto* idx_begin  = atlas::raw_pointer_cast(_indices.data());
    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
Searcher<T>::build_cell_ranges(const int alive) {
    const auto num_of_cells = _universe->number_of_cells();
    auto* start             = atlas::raw_pointer_cast(_cell_start.data());
    auto* end               = atlas::raw_pointer_cast(_cell_end.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            start[cell] = -1;
            end[cell]   = -1;
        });

    const auto* keys = atlas::raw_pointer_cast(_keys.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const std::uint32_t key = keys[i];

            if (i == 0 || key != keys[i - 1]) start[key] = i;
            if (i == alive - 1 || key != keys[i + 1]) end[key] = i + 1;
        });
}

template <typename T>
template <typename CandidateFilter>
void
Searcher<T>::build_cell_neighbors(const int alive,
                                  const Vector3<T>* positions,
                                  CandidateFilter filter) {
    _neighbor_offsets.resize(static_cast<std::size_t>(alive + 1));
    _neighbor_counts.resize(static_cast<std::size_t>(alive));

    auto* counts          = atlas::raw_pointer_cast(_neighbor_counts.data());
    const auto* indices   = atlas::raw_pointer_cast(_indices.data());
    const auto* start     = atlas::raw_pointer_cast(_cell_start.data());
    const auto* end       = atlas::raw_pointer_cast(_cell_end.data());
    const Vector3<T> lc   = _universe->lower_corner();
    const T inv_h         = _universe->inverse_cell_size();
    const T radius2       = _universe->cell_size() * _universe->cell_size();
    const Vector3<int> gs = _universe->grid_size();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        detail::SearcherNeighborCount<T, CandidateFilter> {
            counts,
            indices,
            start,
            end,
            positions,
            filter,
            lc,
            inv_h,
            radius2,
            gs });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        _neighbor_counts.begin(),
        _neighbor_counts.begin() + static_cast<std::ptrdiff_t>(alive),
        _neighbor_offsets.begin(),
        0);

    const int total = finalize_neighbor_offsets(alive);
    _neighbor_count = total;
    _neighbor_indices.resize(static_cast<std::size_t>(total));
    if (total == 0) {
        return;
    }

    auto* neighbors     = atlas::raw_pointer_cast(_neighbor_indices.data());
    const auto* offsets = atlas::raw_pointer_cast(_neighbor_offsets.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        detail::SearcherNeighborWrite<T, CandidateFilter> {
            neighbors,
            offsets,
            indices,
            start,
            end,
            positions,
            filter,
            lc,
            inv_h,
            radius2,
            gs });
}

template <typename T>
int
Searcher<T>::finalize_neighbor_offsets(const int alive) {
    if (_neighbor_total_count.size() < 1) {
        _neighbor_total_count.resize(1);
    }

    auto* total        = atlas::raw_pointer_cast(_neighbor_total_count.data());
    auto* offsets      = atlas::raw_pointer_cast(_neighbor_offsets.data());
    const auto* counts = atlas::raw_pointer_cast(_neighbor_counts.data());
    const int last     = alive - 1;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        detail::SearcherNeighborTotal {
            total,
            offsets,
            counts,
            alive,
            last });

    int host_total = 0;
    atlas::copy_device_to_host(total, &host_total, 1);
    return host_total;
}

template <typename T>
void
Searcher<T>::clear_neighbors() {
    _neighbor_offsets.resize(0);
    _neighbor_indices.resize(0);
    _neighbor_counts.resize(0);
    _neighbor_count = 0;
}

}