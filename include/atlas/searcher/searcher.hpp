#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

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
const Vector3<T>*
Searcher<T>::position_ptr() const noexcept {
    if (!_fluid) {
        return nullptr;
    }

    const auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
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
    auto* keys_ptr       = atlas::raw_pointer_cast(_keys.data());
    const Vector3<T> lc  = _universe->lower_corner();
    const T inv_h        = _universe->inverse_cell_size();
    const Vector3<int> gs = _universe->grid_size();
    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<T> rel = (positions[i] - lc) * inv_h;
            Vector3<int> ijk     = atlas::math::floor(rel).template cast_to<int>();
            ijk                  = atlas::math::clamp(ijk, lo, hi);
            keys_ptr[i]          = Searcher<T>::linear_key(ijk.x, ijk.y, ijk.z, gs);
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

    // Empty cells are represented by -1 so consumers can test both range ends directly.
    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(_cell_start.data()),
        atlas::raw_pointer_cast(_cell_start.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);
    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(_cell_end.data()),
        atlas::raw_pointer_cast(_cell_end.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    const auto* keys = atlas::raw_pointer_cast(_keys.data());
    auto* start      = atlas::raw_pointer_cast(_cell_start.data());
    auto* end        = atlas::raw_pointer_cast(_cell_end.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const std::uint32_t key = keys[i];

            // Sorted keys make every occupied cell a contiguous range.
            if (i == 0 || key != keys[i - 1]) start[key] = i;
            if (i == alive - 1 || key != keys[i + 1]) end[key] = i + 1;
        });
}

template <typename T>
void
Searcher<T>::clear_neighbors() {
    _neighbor_offsets.resize(0);
    _neighbor_indices.resize(0);
    _neighbor_count = 0;
}

} // namespace atlas::system
