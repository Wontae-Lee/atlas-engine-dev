#include <atlas/searcher/kdtree_searcher.h>

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

KdTreeSearcher::KdTreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid)
    : Searcher(std::move(universe), std::move(fluid)) {
}

void
KdTreeSearcher::build_neighbors(const int alive, const Vector3* pos) {
    const float radius = cell_size();

    // Not an actual k-d tree traversal — this filter just adds a
    // single-axis (x) band restriction on top of SpatialHashingSearcher's
    // ordinary 3x3x3-cell neighbor search; see kdtree_searcher.h's
    // top-of-file documentation for why.
    build_cell_neighbors(
        alive,
        pos,
        [=] ATLAS_ALL_DEVICE(const int,
                         const int,
                         const Vector3& pi,
                         const Vector3& pj) {
            const float dx = pj.x - pi.x;
            return dx >= -radius && dx <= radius;
        });
}

void
KdTreeSearcher::build() {
    if (!_is_invalidated) {
        return;
    }

    const Vector3* positions = position_ptr();
    const int alive          = active_count();

    if (!positions || alive <= 0) {
        reset();
        return;
    }

    prepare_grid_buffers(alive);
    compute_grid_keys(alive, positions);
    sort_by_key(alive);
    build_cell_ranges(alive);
    build_neighbors(alive, positions);
    _is_invalidated = false;
}

KdTreeSearcher::Builder
KdTreeSearcher::builder() noexcept {
    return Builder {};
}

KdTreeSearcher::Builder&
KdTreeSearcher::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

KdTreeSearcher::Builder&
KdTreeSearcher::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

void
KdTreeSearcher::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "KdTreeSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "KdTreeSearcher::Builder: fluid must not be null.";
}

KdTreeSearcher
KdTreeSearcher::Builder::build() const {
    validate();
    return KdTreeSearcher(_universe, _fluid);
}

atlas::host_shared_ptr<KdTreeSearcher>
KdTreeSearcher::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<KdTreeSearcher>(_universe, _fluid);
}

}
