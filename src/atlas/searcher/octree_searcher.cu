#include <atlas/searcher/octree_searcher.h>

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

OctreeSearcher::OctreeSearcher(UniverseHostPtr universe, FluidHostPtr fluid)
    : Searcher(std::move(universe), std::move(fluid)) {
}

void
OctreeSearcher::build_neighbors(const int alive, const Vector3* pos) {
    // Not an actual recursive octree — this filter restricts
    // SpatialHashingSearcher's ordinary neighbor search to candidates
    // sharing the same octant (one bit per axis, comparing each of x/y/z
    // against the domain's midpoint independently) as the query particle;
    // see octree_searcher.h's top-of-file documentation for why this is
    // only a single level of partitioning, not a real tree.
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
            const int iz = pi.z >= center.z ? 1 : 0;
            const int jx = pj.x >= center.x ? 1 : 0;
            const int jy = pj.y >= center.y ? 1 : 0;
            const int jz = pj.z >= center.z ? 1 : 0;
            return ix == jx && iy == jy && iz == jz;
        });
}

void
OctreeSearcher::build() {
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

OctreeSearcher::Builder
OctreeSearcher::builder() noexcept {
    return Builder {};
}

OctreeSearcher::Builder&
OctreeSearcher::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

OctreeSearcher::Builder&
OctreeSearcher::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

void
OctreeSearcher::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "OctreeSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "OctreeSearcher::Builder: fluid must not be null.";
}

OctreeSearcher
OctreeSearcher::Builder::build() const {
    validate();
    return OctreeSearcher(_universe, _fluid);
}

atlas::host_shared_ptr<OctreeSearcher>
OctreeSearcher::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<OctreeSearcher>(_universe, _fluid);
}

}
