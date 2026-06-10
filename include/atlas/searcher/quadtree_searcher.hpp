#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
QuadtreeSearcher<T>::QuadtreeSearcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid)
    : Searcher<T>(std::move(universe), std::move(fluid)) {
}

template <typename T>
void
QuadtreeSearcher<T>::build_neighbors(const int alive, const Vector3<T>* pos) {
    // Fixed-width slots keep the device write pattern deterministic and allocation-free.
    this->_neighbor_offsets.resize(static_cast<std::size_t>(alive + 1));
    this->_neighbor_indices.resize(static_cast<std::size_t>(alive * alive));

    auto* offsets = atlas::raw_pointer_cast(this->_neighbor_offsets.data());
    auto* neighbors = atlas::raw_pointer_cast(this->_neighbor_indices.data());
    const T radius = this->cell_size();
    const T radius2 = radius * radius;
    const Vector3<T> center = (this->lower_corner() + this->_universe->upper_corner()) * T(0.5);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const int base = i * alive;
            const int ix = pos[i].x >= center.x ? 1 : 0;
            const int iy = pos[i].y >= center.y ? 1 : 0;
            offsets[i] = base;

            for (int j = 0; j < alive; ++j) {
                if (i == j) {
                    neighbors[base + j] = -1;
                    continue;
                }

                const int jx = pos[j].x >= center.x ? 1 : 0;
                const int jy = pos[j].y >= center.y ? 1 : 0;

                // The XY quadrant test is the quadtree candidate filter before exact distance.
                if (ix != jx || iy != jy) {
                    neighbors[base + j] = -1;
                    continue;
                }

                const Vector3<T> delta = pos[j] - pos[i];
                neighbors[base + j] = delta.length_squared() <= radius2 ? j : -1;
            }
        });

    this->_neighbor_offsets[static_cast<std::size_t>(alive)] = alive * alive;
    this->_neighbor_count = alive * alive;
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
    this->init_indices_iota(alive);
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

} // namespace atlas::system
