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
SpatialHashingSearcher<T>::SpatialHashingSearcher(UniverseHostPtr<T> universe,
                                                  FluidHostPtr<T> fluid)
    : _universe(std::move(universe))
    , _fluid(std::move(fluid)) {

    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "SpatialHashingSearcher: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "SpatialHashingSearcher: fluid must not be null.";

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

    const auto num_of_cells = _universe->number_of_cells();

    d_cell_start.resize(num_of_cells);

    d_cell_end.resize(num_of_cells);
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {

    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {

    const auto num_of_cells = _universe->number_of_cells();

    d_keys.resize(static_cast<std::size_t>(alive));

    d_indices.resize(static_cast<std::size_t>(alive));

    d_cell_start.resize(num_of_cells);

    d_cell_end.resize(num_of_cells);
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(int n_active) {

    auto* indices_ptr = atlas::raw_pointer_cast(this->d_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_active,
        [=] ATLAS_DEVICE(const int i) {
            indices_ptr[i] = i;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(int alive, const Vector3<T>* pos) {

    const atlas::device_ptr<std::uint32_t> keys_ptr(atlas::raw_pointer_cast(d_keys.data()));

    const Vector3<T> lc = _universe->lower_corner();

    const T inv_h = _universe->inverse_cell_size();

    const Vector3<int> gs = _universe->grid_size();

    const Vector3<int> lo { 0, 0, 0 };

    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(int i) {
            const Vector3<T> rel = (pos[i] - lc) * inv_h;

            Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();

            ijk = atlas::math::clamp(ijk, lo, hi);

            keys_ptr[i] = linear_key(ijk.x, ijk.y, ijk.z, gs);
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) const {

    const unsigned* keys_begin(atlas::raw_pointer_cast(d_keys.data()));
    const unsigned* keys_end = keys_begin + active;
    const int* idx_begin(atlas::raw_pointer_cast(d_indices.data()));

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(int alive) {

    const auto num_of_cells = _universe->number_of_cells();

    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(d_cell_start.data()),
        atlas::raw_pointer_cast(d_cell_start.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(d_cell_end.data()),
        atlas::raw_pointer_cast(d_cell_end.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    const auto* keys = atlas::raw_pointer_cast(this->d_keys.data());
    auto* cell_start = atlas::raw_pointer_cast(this->d_cell_start.data());
    auto* cell_end = atlas::raw_pointer_cast(this->d_cell_end.data());

    const int count = alive;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const std::uint32_t key = keys[i];

            if (i == 0 || key != keys[i - 1]) cell_start[key] = i;

            if (i == count - 1 || key != keys[i + 1]) cell_end[key] = i + 1;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build() {

    if (!_fluid) {

        reset();
        return;
    }

    const auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();

    if (!position_state) {

        reset();
        return;
    }

    const int alive = static_cast<int>(_fluid->particle_count());

    if (alive <= 0) {

        reset();
        return;
    }

    atlas::logger::info() << "\n"
                          << "Building SpatialHashingSearcher for " << alive << " particles.";

    prepare_buffers(alive);

    init_indices_iota(alive);

    compute_keys(alive, atlas::raw_pointer_cast(position_state->data().data()));

    sort_by_key(alive);

    build_cell_ranges(alive);
}

template <typename T>
Vector3<T>
SpatialHashingSearcher<T>::lower_corner() const noexcept {

    return _universe ? _universe->lower_corner() : Vector3<T> {};
}

template <typename T>
Vector3<int>
SpatialHashingSearcher<T>::grid_size() const noexcept {

    return _universe ? _universe->grid_size() : Vector3<int> { 0, 0, 0 };
}

template <typename T>
T
SpatialHashingSearcher<T>::inverse_cell_size() const noexcept {

    return _universe ? _universe->inverse_cell_size() : T(1);
}

template <typename T>
T
SpatialHashingSearcher<T>::cell_size() const noexcept {

    return _universe ? _universe->cell_size() : T(1);
}

template <typename T>
const int*
SpatialHashingSearcher<T>::indices() const noexcept {

    return atlas::raw_pointer_cast(d_indices.data());
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_start() const noexcept {

    return atlas::raw_pointer_cast(d_cell_start.data());
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_end() const noexcept {

    return atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {

    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
void
SpatialHashingSearcher<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "SpatialHashingSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "SpatialHashingSearcher::Builder: fluid must not be null.";
}

template <typename T>
SpatialHashingSearcher<T>
SpatialHashingSearcher<T>::Builder::build() const {

    validate();
    return SpatialHashingSearcher<T>(_universe, _fluid);
}

template <typename T>
atlas::host_shared_ptr<SpatialHashingSearcher<T>>
SpatialHashingSearcher<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<SpatialHashingSearcher<T>>(_universe, _fluid);
}

}
