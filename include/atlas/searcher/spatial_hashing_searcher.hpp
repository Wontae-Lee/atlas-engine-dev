#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

#include <cmath>

namespace atlas::system {

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(UniverseHostPtr<T> universe,
                                                  FluidHostPtr<T> fluid)
    : _universe(std::move(universe))
    , _fluid(std::move(fluid)) {

    // Validate required external dependencies immediately.
    //
    // This searcher cannot function without:
    // 1. a valid universe object, which defines the simulation domain and grid layout
    // 2. a valid fluid object, which provides the particle data to be indexed
    //
    // Failing early here keeps the object from entering a partially valid state.
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "SpatialHashingSearcher: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "SpatialHashingSearcher: fluid must not be null.";

    // Initialize internal buffers into a consistent empty state.
    //
    // Even though the searcher has valid dependencies, its device-side indexing
    // structures are empty until build() is called. reset() ensures that:
    // - particle-dependent buffers start with zero size
    // - cell lookup buffers are resized to the current number of grid cells
    reset();
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder
SpatialHashingSearcher<T>::builder() noexcept {

    // Return a fresh builder instance for chained configuration.
    return Builder {};
}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {

    // Clear particle-dependent buffers because no valid search structure exists yet.
    //
    // d_keys:
    //   Stores one spatial hash key per active particle.
    //
    // d_indices:
    //   Stores particle indices, typically reordered to follow sorted hash keys.
    //
    // When the searcher is reset, there is no active particle indexing state,
    // so both buffers are resized to zero.
    d_keys.resize(0);

    d_indices.resize(0);

    // Keep the cell range buffers sized to the current number of grid cells.
    //
    // Even in a reset state, having cell_start/cell_end sized correctly makes it
    // easier to rebuild later without relying on stale capacity assumptions.
    //
    // Each cell will eventually store:
    // - d_cell_start[cell] = first sorted particle index in that cell
    // - d_cell_end[cell]   = one-past-last sorted particle index in that cell
    //
    // If a cell contains no particles, both values are typically left as -1.
    const auto num_of_cells = _universe->number_of_cells();

    d_cell_start.resize(num_of_cells);

    d_cell_end.resize(num_of_cells);
    _is_invalidated = true;
}

template <typename T>
void
SpatialHashingSearcher<T>::invalidate() noexcept {
    _is_invalidated = true;
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {

    // Flatten a 3D integer cell coordinate (ix, iy, iz) into a single linear key.
    //
    // Layout convention:
    //   x changes fastest,
    //   then y,
    //   then z.
    //
    // Formula:
    //   key = ix + iy * gs.x + iz * gs.x * gs.y
    //
    // This assumes:
    // - 0 <= ix < gs.x
    // - 0 <= iy < gs.y
    // - 0 <= iz < gs.z
    //
    // The caller is responsible for clamping or validating coordinates before
    // invoking this function.
    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {

    // Determine the total number of hash cells defined by the universe grid.
    const auto num_of_cells = _universe->number_of_cells();

    // Allocate or resize per-particle buffers to match the number of active particles.
    //
    // d_keys[i]:
    //   spatial hash key for particle i before sorting
    //
    // d_indices[i]:
    //   original particle index, later reordered together with d_keys
    //
    // After sorting by key, particles that belong to the same cell become contiguous
    // in the sorted arrays.
    d_keys.resize(static_cast<std::size_t>(alive));

    d_indices.resize(static_cast<std::size_t>(alive));

    // Allocate or resize per-cell lookup buffers.
    //
    // These arrays are indexed by linear cell key and later populated with
    // the [start, end) range into the sorted particle index list.
    d_cell_start.resize(num_of_cells);

    d_cell_end.resize(num_of_cells);
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(int n_active) {

    // Obtain a raw device pointer to the index buffer.
    auto* indices_ptr = atlas::raw_pointer_cast(this->d_indices.data());

    // Initialize d_indices with the sequence:
    //   0, 1, 2, ..., n_active - 1
    //
    // This is a standard preparation step before sort-by-key:
    // - d_keys holds the cell key for each particle
    // - d_indices holds the original particle id
    //
    // After sorting by key, d_indices becomes a permutation array that maps
    // sorted entries back to the original particle ordering.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_active,
        [=] ATLAS_DEVICE(const int i) {
            indices_ptr[i] = i;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(int alive, const Vector3<T>* pos) {

    // Obtain a raw device pointer to the key buffer.
    const atlas::device_ptr<std::uint32_t> keys_ptr(atlas::raw_pointer_cast(d_keys.data()));

    // Cache universe parameters locally so they can be captured efficiently by value
    // into the device lambda below.
    //
    // lc:
    //   lower corner of the spatial hashing domain
    //
    // inv_h:
    //   inverse cell size, used to convert world-space distances into grid-space units
    //
    // gs:
    //   grid resolution along x, y, z
    const Vector3<T> lc = _universe->lower_corner();

    const T inv_h = _universe->inverse_cell_size();

    const Vector3<int> gs = _universe->grid_size();

    // Define valid integer index bounds for grid coordinates.
    //
    // lo = minimum legal cell coordinate
    // hi = maximum legal cell coordinate
    //
    // Positions outside the universe bounds are clamped into this valid range,
    // preventing illegal cell access and guaranteeing that linear_key() receives
    // in-range indices.
    const Vector3<int> lo { 0, 0, 0 };

    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(int i) {
            // Convert the particle position from world space into grid-relative coordinates.
            //
            // rel = (pos - lower_corner) / cell_size
            //
            // Because inv_h = 1 / cell_size, the division is implemented as multiplication.
            const Vector3<T> rel = (pos[i] - lc) * inv_h;

            // Compute the integer cell coordinate by flooring the continuous grid coordinate.
            //
            // Example:
            //   rel = (2.7, 5.1, 3.9) -> ijk = (2, 5, 3)
            Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();

            // Clamp the cell coordinate to the valid grid range.
            //
            // This makes the indexing robust when particles lie slightly outside the
            // nominal domain due to numerical drift, boundary handling, or integration noise.
            ijk = atlas::math::clamp(ijk, lo, hi);

            // Convert the 3D cell coordinate into a single linear key.
            //
            // This key is the quantity used for sorting and later for building
            // contiguous cell ranges.
            keys_ptr[i] = linear_key(ijk.x, ijk.y, ijk.z, gs);
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) {

    // Raw pointers to the device arrays used by the key-value sort.
    auto* keys_begin = atlas::raw_pointer_cast(d_keys.data());
    auto* keys_end   = keys_begin + active;
    auto* idx_begin  = atlas::raw_pointer_cast(d_indices.data());

    // Sort particle entries by spatial hash key.
    //
    // Conceptually, this sorts pairs:
    //   (key[i], index[i])
    //
    // After sorting:
    // - equal keys form contiguous segments
    // - d_indices contains the corresponding original particle ids
    //
    // This contiguous layout is what allows the later construction of fast
    // per-cell [start, end) ranges.
    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(int alive) {

    // Determine the number of cells in the spatial hash grid.
    const auto num_of_cells = _universe->number_of_cells();

    // Initialize all cell starts to -1.
    //
    // Interpretation:
    //   -1 means "this cell currently has no particles assigned".
    //
    // Only cells that actually appear in the sorted key stream will later receive
    // valid start/end entries.
    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(d_cell_start.data()),
        atlas::raw_pointer_cast(d_cell_start.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    // Initialize all cell ends to -1 for the same reason.
    atlas::parallel_fill<ExecutionPolicy::device>(
        atlas::raw_pointer_cast(d_cell_end.data()),
        atlas::raw_pointer_cast(d_cell_end.data()) + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    // Raw pointers to the sorted key array and the output cell range buffers.
    const auto* keys = atlas::raw_pointer_cast(this->d_keys.data());
    auto* cell_start = atlas::raw_pointer_cast(this->d_cell_start.data());
    auto* cell_end   = atlas::raw_pointer_cast(this->d_cell_end.data());

    // Cache the active particle count so the device lambda can use it directly.
    const int count = alive;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const std::uint32_t key = keys[i];

            // Detect the first occurrence of a cell key in the sorted key array.
            //
            // Because the keys are sorted, the first time a new key appears marks
            // the inclusive begin position of that cell's particle range.
            if (i == 0 || key != keys[i - 1]) cell_start[key] = i;

            // Detect the last occurrence of a cell key in the sorted key array.
            //
            // The stored end index is exclusive, so we write i + 1.
            //
            // This means each populated cell can later be iterated as:
            //   for (int k = cell_start[cell]; k < cell_end[cell]; ++k) { ... }
            if (i == count - 1 || key != keys[i + 1]) cell_end[key] = i + 1;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build() {

    if (!_is_invalidated) {
        return;
    }

    // If the fluid object is missing, no search structure can be built.
    //
    // Reset the internal buffers so the searcher remains in a safe, empty state.
    if (!_fluid) {

        reset();
        return;
    }

    // Retrieve the particle position state required for spatial indexing.
    //
    // The spatial hash is fundamentally built from particle positions, so if that
    // state is unavailable there is nothing meaningful to index.
    const auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();

    if (!position_state) {

        reset();
        return;
    }

    // Determine the number of currently alive / active particles.
    //
    // Only active particles participate in the search structure.
    const int alive = static_cast<int>(_fluid->particle_count());

    // If there are no active particles, keep the searcher empty.
    if (alive <= 0) {

        reset();
        return;
    }

    // Step 1:
    // Ensure all required device buffers are sized appropriately.
    prepare_buffers(alive);

    // Step 2:
    // Initialize the index array with the identity permutation.
    //
    // This array will be reordered together with the keys during sort_by_key().
    init_indices_iota(alive);

    // Step 3:
    // Compute one spatial hash key per particle from its world-space position.
    compute_keys(alive, atlas::raw_pointer_cast(position_state->data().data()));

    // Step 4:
    // Sort particle entries by spatial key so particles in the same cell become contiguous.
    sort_by_key(alive);

    // Step 5:
    // Build per-cell [start, end) ranges over the sorted particle list.
    build_cell_ranges(alive);
    _is_invalidated = false;
}

template <typename T>
Vector3<T>
SpatialHashingSearcher<T>::lower_corner() const noexcept {

    // Return the universe lower corner when available.
    // Otherwise, return a default zero vector as a safe fallback.
    return _universe ? _universe->lower_corner() : Vector3<T> {};
}

template <typename T>
Vector3<int>
SpatialHashingSearcher<T>::grid_size() const noexcept {

    // Return the universe grid resolution when available.
    // Otherwise, return a zero-sized grid.
    return _universe ? _universe->grid_size() : Vector3<int> { 0, 0, 0 };
}

template <typename T>
T
SpatialHashingSearcher<T>::inverse_cell_size() const noexcept {

    // Return the inverse cell size when the universe exists.
    // Use 1 as a conservative fallback to avoid division-related surprises
    // in code paths that may inspect the value defensively.
    return _universe ? _universe->inverse_cell_size() : T(1);
}

template <typename T>
T
SpatialHashingSearcher<T>::cell_size() const noexcept {

    // Return the physical cell size when the universe exists.
    // Use 1 as a fallback for the same defensive reason as inverse_cell_size().
    return _universe ? _universe->cell_size() : T(1);
}

template <typename T>
const int*
SpatialHashingSearcher<T>::indices() const noexcept {

    // Expose the raw device pointer to the sorted particle index array.
    //
    // Consumers interpret this together with d_cell_start/d_cell_end to traverse
    // particles belonging to a specific cell.
    return atlas::raw_pointer_cast(d_indices.data());
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_start() const noexcept {

    // Expose the raw device pointer to the inclusive cell-begin array.
    return atlas::raw_pointer_cast(d_cell_start.data());
}

template <typename T>
const int*
SpatialHashingSearcher<T>::cell_end() const noexcept {

    // Expose the raw device pointer to the exclusive cell-end array.
    return atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {

    // Store the universe dependency for later validation and construction.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the fluid dependency for later validation and construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
void
SpatialHashingSearcher<T>::Builder::validate() const {

    // Validate that all mandatory dependencies are present before object creation.
    //
    // Keeping this logic centralized avoids duplicated checks in build() and
    // make_host_shared(), and makes builder misuse fail consistently.
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "SpatialHashingSearcher::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "SpatialHashingSearcher::Builder: fluid must not be null.";
}

template <typename T>
SpatialHashingSearcher<T>
SpatialHashingSearcher<T>::Builder::build() const {

    // Refuse construction if any required dependency is missing.
    validate();

    // Construct the final searcher from the validated dependencies.
    return SpatialHashingSearcher<T>(_universe, _fluid);
}

template <typename T>
atlas::host_shared_ptr<SpatialHashingSearcher<T>>
SpatialHashingSearcher<T>::Builder::make_host_shared() const {

    // Refuse shared-object construction if any required dependency is missing.
    validate();

    // Create the object directly inside a host-managed shared pointer.
    return atlas::make_host_shared<SpatialHashingSearcher<T>>(_universe, _fluid);
}

}
