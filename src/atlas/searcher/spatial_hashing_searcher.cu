#include <atlas/searcher/spatial_hashing_searcher.h>

#include <atlas/logging/logging.h>

#include <stdexcept>
#include <utility>

namespace atlas {

SpatialHashingSearcher::SpatialHashingSearcher(UniverseHostPtr universe, FluidHostPtr fluid)
    : Searcher(std::move(universe), std::move(fluid)) {
}

SpatialHashingSearcher::Builder
SpatialHashingSearcher::builder() noexcept {
    return Builder {};
}

void
SpatialHashingSearcher::reset() noexcept {
    Searcher::reset();
}

void
SpatialHashingSearcher::invalidate() noexcept {
    Searcher::invalidate();
}

void
SpatialHashingSearcher::prepare_buffers(const int alive) {
    prepare_grid_buffers(alive);
}

void
SpatialHashingSearcher::init_indices_iota(const int alive) {
    Searcher::init_indices_iota(alive);
}

void
SpatialHashingSearcher::compute_keys(const int alive, const Float3* pos) {
    compute_grid_keys(alive, pos);
}

void
SpatialHashingSearcher::sort_by_key(const int active) {
    Searcher::sort_by_key(active);
}

void
SpatialHashingSearcher::build_cell_ranges(const int alive) {
    Searcher::build_cell_ranges(alive);
}

void
SpatialHashingSearcher::build_neighbors(const int alive, const Float3* pos) {
    // Always-accepting filter: this is the unfiltered baseline
    // KdTreeSearcher/OctreeSearcher/QuadtreeSearcher each restrict further
    // with their own geometric predicate (see those files).
    build_cell_neighbors(
        alive,
        pos,
        [] ATLAS_ALL_DEVICE(const int,
                            const int,
                            const Float3&,
                            const Float3&) {
            return true;
        });
}

void
SpatialHashingSearcher::build() {
    if (!_is_invalidated) {
        return;
    }

    const Float3* positions = position_ptr();
    const int alive         = active_count();

    if (!positions || alive <= 0) {
        reset();
        return;
    }

    prepare_buffers(alive);
    compute_keys(alive, positions);
    sort_by_key(alive);
    build_cell_ranges(alive);
    build_neighbors(alive, positions);
    _is_invalidated = false;
}

Float3
SpatialHashingSearcher::lower_corner() const noexcept {
    return Searcher::lower_corner();
}

Int3
SpatialHashingSearcher::grid_size() const noexcept {
    return Searcher::grid_size();
}

float
SpatialHashingSearcher::inverse_cell_size() const noexcept {
    return Searcher::inverse_cell_size();
}

float
SpatialHashingSearcher::cell_size() const noexcept {
    return Searcher::cell_size();
}

const int*
SpatialHashingSearcher::indices() const noexcept {
    return Searcher::indices();
}

const int*
SpatialHashingSearcher::cell_start() const noexcept {
    return Searcher::cell_start();
}

const int*
SpatialHashingSearcher::cell_end() const noexcept {
    return Searcher::cell_end();
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

void
SpatialHashingSearcher::Builder::validate() const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "SpatialHashingSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "SpatialHashingSearcher::Builder: fluid must not be null.";
}

SpatialHashingSearcher
SpatialHashingSearcher::Builder::build() const {
    validate();
    return SpatialHashingSearcher(_universe, _fluid);
}

atlas::host_shared_ptr<SpatialHashingSearcher>
SpatialHashingSearcher::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<SpatialHashingSearcher>(_universe, _fluid);
}

}
