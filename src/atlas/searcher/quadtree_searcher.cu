#include <atlas/searcher/quadtree_searcher.h>

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

QuadtreeSearcher::QuadtreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid)
    : Searcher(std::move(universe), std::move(fluid)) {
}

void
QuadtreeSearcher::build_neighbors(const int alive, const Vector3* pos) {
    // Not an actual recursive quadtree — the 2D analog of OctreeSearcher's
    // single-level octant filter (x/y only, z ignored); see
    // quadtree_searcher.h's top-of-file documentation.
    const Vector3 center = (lower_corner() + _universe->upper_corner()) * 0.5f;

    build_cell_neighbors(
        alive,
        pos,
        [=] ATLAS_ALL_DEVICE(const int,
                         const int,
                         const Vector3& pi,
                         const Vector3& pj) {
            const int ix = pi.x >= center.x ? 1 : 0;
            const int iy = pi.y >= center.y ? 1 : 0;
            const int jx = pj.x >= center.x ? 1 : 0;
            const int jy = pj.y >= center.y ? 1 : 0;
            return ix == jx && iy == jy;
        });
}

void
QuadtreeSearcher::build() {
    if (!_is_invalidated) return;

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

QuadtreeSearcher::Builder
QuadtreeSearcher::builder() noexcept {
    return Builder {};
}

QuadtreeSearcher::Builder&
QuadtreeSearcher::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

QuadtreeSearcher::Builder&
QuadtreeSearcher::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

void
QuadtreeSearcher::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "QuadtreeSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "QuadtreeSearcher::Builder: fluid must not be null.";
}

QuadtreeSearcher
QuadtreeSearcher::Builder::build() const {
    validate();
    return QuadtreeSearcher(_universe, _fluid);
}

atlas::host_shared_ptr<QuadtreeSearcher>
QuadtreeSearcher::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<QuadtreeSearcher>(_universe, _fluid);
}

}
