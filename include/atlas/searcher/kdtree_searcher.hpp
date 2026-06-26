#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
KdTreeSearcher<T>::KdTreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid)
    : Searcher<T>(std::move(universe), std::move(fluid)) {
}

template <typename T>
void
KdTreeSearcher<T>::build_neighbors(const int alive, const Vector3<T>* pos) {
    const T radius = this->cell_size();

    this->build_cell_neighbors(
        alive,
        pos,
        [=] ATLAS_DEVICE(const int,
                         const int,
                         const Vector3<T>& pi,
                         const Vector3<T>& pj) {
            const T dx = pj.x - pi.x;
            return dx >= -radius && dx <= radius;
        });
}

template <typename T>
void
KdTreeSearcher<T>::build() {
    if (!this->_is_invalidated) {
        return;
    }

    const Vector3<T>* positions = this->position_ptr();
    const int alive             = this->active_count();

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
typename KdTreeSearcher<T>::Builder
KdTreeSearcher<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
typename KdTreeSearcher<T>::Builder&
KdTreeSearcher<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename KdTreeSearcher<T>::Builder&
KdTreeSearcher<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
void
KdTreeSearcher<T>::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "KdTreeSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KdTreeSearcher::Builder: fluid must not be null.";
}

template <typename T>
KdTreeSearcher<T>
KdTreeSearcher<T>::Builder::build() const {
    validate();
    return KdTreeSearcher<T>(_universe, _fluid);
}

template <typename T>
atlas::host_shared_ptr<KdTreeSearcher<T>>
KdTreeSearcher<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<KdTreeSearcher<T>>(_universe, _fluid);
}

}