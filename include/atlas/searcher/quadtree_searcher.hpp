#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
QuadtreeSearcher<T>::QuadtreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid)
    : Searcher<T>(std::move(universe), std::move(fluid)) {
}

template <typename T>
void
QuadtreeSearcher<T>::build_neighbors(const int alive, const Vector3<T>* pos) {
    const Vector3<T> center = (this->lower_corner() + this->_universe->upper_corner()) * T(0.5);

    this->build_cell_neighbors(
        alive,
        pos,
        [=] ATLAS_DEVICE(const int,
                         const int,
                         const Vector3<T>& pi,
                         const Vector3<T>& pj) {
            const int ix = pi.x >= center.x ? 1 : 0;
            const int iy = pi.y >= center.y ? 1 : 0;
            const int jx = pj.x >= center.x ? 1 : 0;
            const int jy = pj.y >= center.y ? 1 : 0;
            return ix == jx && iy == jy;
        });
}

template <typename T>
void
QuadtreeSearcher<T>::build() {
    if (!this->_is_invalidated) return;

    const Vector3<T>* positions = this->position_ptr();
    const int alive = this->active_count();

    if (!positions || alive <= 0) {
        this->reset();
        return;
    }

    this->prepare_grid_buffers(alive);
    this->compute_grid_keys(alive, positions);
    this->sort_by_key(alive);
    this->build_cell_ranges(alive);
    build_neighbors(alive, positions);
    this->_is_invalidated = false;
}

template <typename T>
typename QuadtreeSearcher<T>::Builder
QuadtreeSearcher<T>::builder() noexcept { return Builder {}; }

template <typename T>
typename QuadtreeSearcher<T>::Builder&
QuadtreeSearcher<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename QuadtreeSearcher<T>::Builder&
QuadtreeSearcher<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
void
QuadtreeSearcher<T>::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "QuadtreeSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "QuadtreeSearcher::Builder: fluid must not be null.";
}

template <typename T>
QuadtreeSearcher<T>
QuadtreeSearcher<T>::Builder::build() const {
    validate();
    return QuadtreeSearcher<T>(_universe, _fluid);
}

template <typename T>
atlas::host_shared_ptr<QuadtreeSearcher<T>>
QuadtreeSearcher<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<QuadtreeSearcher<T>>(_universe, _fluid);
}

} // namespace atlas
