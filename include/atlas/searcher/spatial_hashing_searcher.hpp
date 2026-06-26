#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(UniverseHostPtr<T> universe,
                                                  FluidHostPtr<T> fluid)
    : Searcher<T>(std::move(universe), std::move(fluid)) {
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder
SpatialHashingSearcher<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {
    Searcher<T>::reset();
}

template <typename T>
void
SpatialHashingSearcher<T>::invalidate() noexcept {
    Searcher<T>::invalidate();
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {
    return Searcher<T>::linear_key(ix, iy, iz, gs);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {
    this->prepare_grid_buffers(alive);
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(const int alive) {
    Searcher<T>::init_indices_iota(alive);
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(const int alive, const Vector3<T>* pos) {
    this->compute_grid_keys(alive, pos);
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) {
    Searcher<T>::sort_by_key(active);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(const int alive) {
    Searcher<T>::build_cell_ranges(alive);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_neighbors(const int alive, const Vector3<T>* pos) {
    this->build_cell_neighbors(
        alive,
        pos,
        [] ATLAS_DEVICE(const int,
                        const int,
                        const Vector3<T>&,
                        const Vector3<T>&) {
            return true;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build() {
    if (!this->_is_invalidated) {
        return;
    }

    const Vector3<T>* positions = this->position_ptr();
    const int alive             = this->active_count();

    if (!positions || alive <= 0) {
        reset();
        return;
    }

    prepare_buffers(alive);
    compute_keys(alive, positions);
    sort_by_key(alive);
    build_cell_ranges(alive);
    build_neighbors(alive, positions);
    this->_is_invalidated = false;
}

template <typename T>
Vector3<T>
SpatialHashingSearcher<T>::lower_corner() const noexcept {
    return Searcher<T>::lower_corner();
}

template <typename T>
Vector3<int>
SpatialHashingSearcher<T>::grid_size() const noexcept {
    return Searcher<T>::grid_size();
}

template <typename T>
T
SpatialHashingSearcher<T>::inverse_cell_size() const noexcept {
    return Searcher<T>::inverse_cell_size();
}

template <typename T>
T
SpatialHashingSearcher<T>::cell_size() const noexcept {
    return Searcher<T>::cell_size();
}

template <typename T>
const int*
SpatialHashingSearcher<T>::indices() const noexcept {
    return Searcher<T>::indices();
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_start() const noexcept {
    return Searcher<T>::cell_start();
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_end() const noexcept {
    return Searcher<T>::cell_end();
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