#include <atlas/searcher/searcher.h>

#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_sort.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Searcher::Searcher(UniverseHostPtr universe, FluidHostPtr fluid)
    : _universe(std::move(universe))
    , _fluid(std::move(fluid)) {
    validate_dependencies("Searcher");
    reset();
}

void
Searcher::validate_dependencies(const char* owner) const {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << owner << ": universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << owner << ": fluid must not be null.";
}

void
Searcher::invalidate() noexcept {
    _is_invalidated = true;
}

void
Searcher::reset() noexcept {
    _keys.resize(0);
    _indices.resize(0);
    clear_neighbors();

    const int cell_count = _universe ? _universe->cell_count() : 0;
    _cell_start.resize(static_cast<std::size_t>(cell_count));
    _cell_end.resize(static_cast<std::size_t>(cell_count));
    _is_invalidated = true;
}

Float3
Searcher::lower_corner() const noexcept {
    return _universe ? _universe->lower_corner() : Float3(0.0f, 0.0f, 0.0f);
}

Int3
Searcher::grid_size() const noexcept {
    return _universe ? _universe->grid_size() : Int3(0, 0, 0);
}

float
Searcher::inverse_cell_size() const noexcept {
    return _universe ? _universe->inverse_cell_size() : 1.0f;
}

float
Searcher::cell_size() const noexcept {
    return _universe ? _universe->cell_size() : 1.0f;
}

const int*
Searcher::indices() const noexcept {
    return atlas::raw_pointer_cast(_indices.data());
}

const int*
Searcher::cell_start() const noexcept {
    return atlas::raw_pointer_cast(_cell_start.data());
}

const int*
Searcher::cell_end() const noexcept {
    return atlas::raw_pointer_cast(_cell_end.data());
}

const int*
Searcher::neighbor_offsets() const noexcept {
    return atlas::raw_pointer_cast(_neighbor_offsets.data());
}

const int*
Searcher::neighbor_indices() const noexcept {
    return atlas::raw_pointer_cast(_neighbor_indices.data());
}

int
Searcher::neighbor_count() const noexcept {
    return _neighbor_count;
}

const Float3*
Searcher::position_ptr() const noexcept {
    if (!_fluid) {
        return nullptr;
    }

    const auto* position_state = _fluid->state<atlas::FluidPositionState>();
    if (!position_state || position_state->data().empty()) {
        return nullptr;
    }

    return atlas::raw_pointer_cast(position_state->data().data());
}

int
Searcher::active_count() const noexcept {
    return _fluid ? static_cast<int>(_fluid->particle_count()) : 0;
}

void
Searcher::prepare_grid_buffers(const int alive) {
    const int cell_count = _universe->cell_count();
    _keys.resize(static_cast<std::size_t>(alive));
    _indices.resize(static_cast<std::size_t>(alive));
    _cell_start.resize(static_cast<std::size_t>(cell_count));
    _cell_end.resize(static_cast<std::size_t>(cell_count));
}

void
Searcher::init_indices_iota(const int alive) {
    auto* indices_ptr = atlas::raw_pointer_cast(_indices.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(const int i) {
            indices_ptr[i] = i;
        });
}

void
Searcher::compute_grid_keys(const int alive, const Float3* positions) {
    auto* keys_ptr    = atlas::raw_pointer_cast(_keys.data());
    auto* indices_ptr = atlas::raw_pointer_cast(_indices.data());
    const Float3 lc   = _universe->lower_corner();
    const float inv_h = _universe->inverse_cell_size();
    const Int3 gs     = _universe->grid_size();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const Int3 ijk = Searcher::cell_for(positions[i], lc, inv_h, gs);
            keys_ptr[i]    = Searcher::linear_key(ijk.x, ijk.y, ijk.z, gs);
            indices_ptr[i] = i;
        });
}

void
Searcher::sort_by_key(const int alive) {
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
Searcher::build_cell_ranges(const int alive) {
    const int cell_count = _universe->cell_count();
    auto* start          = atlas::raw_pointer_cast(_cell_start.data());
    auto* end            = atlas::raw_pointer_cast(_cell_end.data());

    // -1 sentinel marks a cell with no particles at all (distinguishes
    // "empty cell" from "cell 0..0", which build_cell_neighbors and
    // similar consumers check for via `begin < 0`).
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        cell_count,
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

int
Searcher::finalize_neighbor_offsets(const int alive) {
    if (_neighbor_total_count.size() < 1) {
        _neighbor_total_count.resize(1);
    }

    auto* total        = atlas::raw_pointer_cast(_neighbor_total_count.data());
    auto* offsets      = atlas::raw_pointer_cast(_neighbor_offsets.data());
    const auto* counts = atlas::raw_pointer_cast(_neighbor_counts.data());
    const int last     = alive - 1;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        SearcherNeighborTotal {
            total,
            offsets,
            counts,
            alive,
            last });

    int host_total = 0;
    atlas::copy_device_to_host(total, &host_total, 1);
    return host_total;
}

void
Searcher::clear_neighbors() {
    _neighbor_offsets.resize(0);
    _neighbor_indices.resize(0);
    _neighbor_counts.resize(0);
    _neighbor_count = 0;
}

}
