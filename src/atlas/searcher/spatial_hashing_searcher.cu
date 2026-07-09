#include <atlas/searcher/spatial_hashing_searcher.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <stdexcept>

namespace atlas {

SpatialHashingSearcher::SpatialHashingSearcher(const Float3& lower_corner,
                                               const float cell_size,
                                               const Int3& grid_size)
    : _cell_count(grid_size.x * grid_size.y * grid_size.z)
    , _cell_size(cell_size)
    , _inverse_cell_size(1.0f / cell_size)
    , _lower_corner(lower_corner)
    , _grid_size(grid_size) {
    reset();
}

SpatialHashingSearcher::Builder
SpatialHashingSearcher::builder() noexcept {
    return Builder {};
}

void
SpatialHashingSearcher::classify(const FluidPositionState* positions,
                                 UniverseNumberParticleState* number_particle,
                                 const int particle_count) {
    if (positions == nullptr
        || particle_count <= 0
        || static_cast<std::size_t>(particle_count) > positions->data().size()) {
        reset();
        write_cell_counts(number_particle);
        return;
    }

    const auto* position_ptr = atlas::raw_pointer_cast(positions->data().data());

    _keys.resize(static_cast<std::size_t>(particle_count));
    _indices.resize(static_cast<std::size_t>(particle_count));

    compute_keys(particle_count, position_ptr);
    sort_keys(particle_count);
    build_cell_ranges(particle_count);
    write_cell_counts(number_particle);
}

void
SpatialHashingSearcher::reset() {
    _keys.resize(0);
    _indices.resize(0);
    _cell_start.assign(static_cast<std::size_t>(_cell_count), -1);
    _cell_end.assign(static_cast<std::size_t>(_cell_count), -1);
}

const int*
SpatialHashingSearcher::indices() const noexcept {
    return atlas::raw_pointer_cast(_indices.data());
}

const int*
SpatialHashingSearcher::cell_start() const noexcept {
    return atlas::raw_pointer_cast(_cell_start.data());
}

const int*
SpatialHashingSearcher::cell_end() const noexcept {
    return atlas::raw_pointer_cast(_cell_end.data());
}

void
SpatialHashingSearcher::compute_keys(const int alive, const Float3* positions) {
    auto* keys_ptr    = atlas::raw_pointer_cast(_keys.data());
    auto* indices_ptr = atlas::raw_pointer_cast(_indices.data());
    const Float3 lc   = _lower_corner;
    const float inv_h = _inverse_cell_size;
    const Int3 gs     = _grid_size;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const Int3 cell = SpatialHashingSearcher::cell_for(positions[i], lc, inv_h, gs);
            keys_ptr[i]     = SpatialHashingSearcher::linear_key(cell, gs);
            indices_ptr[i]  = i;
        });
}

void
SpatialHashingSearcher::sort_keys(const int alive) {
    // Sorting _indices by cell key groups particle indices contiguously by
    // cell (a counting-sort-equivalent outcome via key sort); _keys itself
    // ends up sorted too, which build_cell_ranges below relies on to find
    // each cell's contiguous run via a single linear pass.
    auto* keys_begin = atlas::raw_pointer_cast(_keys.data());
    auto* keys_end   = keys_begin + alive;
    auto* idx_begin  = atlas::raw_pointer_cast(_indices.data());
    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

void
SpatialHashingSearcher::build_cell_ranges(const int alive) {
    auto* start = atlas::raw_pointer_cast(_cell_start.data());
    auto* end   = atlas::raw_pointer_cast(_cell_end.data());

    // -1 sentinel marks a cell with no particles at all, which distinguishes
    // "empty cell" from "the range [0, 0) of cell 0".
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        _cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            start[cell] = -1;
            end[cell]   = -1;
        });

    const auto* keys = atlas::raw_pointer_cast(_keys.data());

    // One thread per sorted particle detects run boundaries in the
    // sorted-by-key array: a particle starts a new cell's run whenever its
    // key differs from the previous sorted particle's (or it's the very
    // first), and ends one whenever the next particle's key differs (or
    // it's the last) — the standard "find contiguous run boundaries in a
    // sorted array" pattern, parallelized by letting every element check
    // its own neighbors rather than a sequential scan.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const std::uint32_t key = keys[i];

            if (i == 0 || key != keys[i - 1]) start[key] = i;
            if (i == alive - 1 || key != keys[i + 1]) end[key] = i + 1;
        });
}

void
SpatialHashingSearcher::write_cell_counts(UniverseNumberParticleState* number_particle) const {
    if (number_particle == nullptr) {
        atlas::warn()
            << "SpatialHashingSearcher::classify: no UniverseNumberParticleState; "
            << "per-cell particle counts are not recorded. "
            << "The DSMC solver and the Knudsen codec need them.";
        return;
    }

    if (number_particle->data().size() != static_cast<std::size_t>(_cell_count)) {
        atlas::warn()
            << "SpatialHashingSearcher::classify: UniverseNumberParticleState holds "
            << number_particle->data().size() << " cells but the grid has " << _cell_count
            << "; per-cell particle counts are not recorded.";
        return;
    }

    auto* counts      = atlas::raw_pointer_cast(number_particle->data().data());
    const auto* start = atlas::raw_pointer_cast(_cell_start.data());
    const auto* end   = atlas::raw_pointer_cast(_cell_end.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        _cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            counts[cell] = start[cell] < 0
                ? 0.0f
                : static_cast<float>(end[cell] - start[cell]);
        });
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_universe(const Universe& universe) {
    _lower_corner = universe.lower_corner();
    _cell_size    = universe.cell_size();
    _grid_size    = universe.grid_size();
    return *this;
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_lower_corner(const Float3& lower_corner) noexcept {
    _lower_corner = lower_corner;
    return *this;
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_cell_size(const float cell_size) noexcept {
    _cell_size = cell_size;
    return *this;
}

SpatialHashingSearcher::Builder&
SpatialHashingSearcher::Builder::with_grid_size(const Int3& grid_size) noexcept {
    _grid_size = grid_size;
    return *this;
}

void
SpatialHashingSearcher::Builder::validate() const {
    atlas::check<std::invalid_argument>(_cell_size > 0.0f)
        << "SpatialHashingSearcher::Builder: cell_size must be positive. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(atlas::all(_grid_size >= Int3(1, 1, 1)))
        << "SpatialHashingSearcher::Builder: grid_size must be >= 1 on all axes. "
        << "grid_size=(" << _grid_size.x << "," << _grid_size.y << "," << _grid_size.z << ")";
}

SpatialHashingSearcher
SpatialHashingSearcher::Builder::build() const {
    validate();

    return SpatialHashingSearcher(_lower_corner, _cell_size, _grid_size);
}

atlas::host_shared_ptr<SpatialHashingSearcher>
SpatialHashingSearcher::Builder::make_host_shared() const {
    return atlas::make_host_shared<SpatialHashingSearcher>(build());
}

}
