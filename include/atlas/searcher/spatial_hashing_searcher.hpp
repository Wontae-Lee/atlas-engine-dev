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
SpatialHashingSearcher<T>::SpatialHashingSearcher(DomainHostPtr<T> domain)
    : _domain(std::move(domain)) {
    /**
     * @brief Construct a spatial hashing searcher for a domain.
     *
     * @param domain Domain defining the spatial grid geometry.
     *
     * @details
     * The domain is mandatory because the searcher derives:
     * - lower corner,
     * - grid size,
     * - inverse cell size,
     * - total number of cells
     *
     * from the domain configuration.
     */
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher: domain must not be null.";

    /**
     * @brief Initialize internal buffers into a consistent empty state.
     */
    reset();
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder
SpatialHashingSearcher<T>::builder() noexcept {
    /**
     * @brief Return a fresh builder object for staged construction.
     */
    return Builder {};
}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {
    /**
     * @brief Reset all internal buffers to an empty search state.
     *
     * @details
     * This method:
     * - clears key and index arrays,
     * - resizes cell-range arrays to the domain cell count,
     * - refreshes cached raw pointers into the device buffers.
     *
     * It does not remove the associated domain.
     */
    d_keys.resize(0);
    ///< Clear the per-particle cell-key array.

    d_indices.resize(0);
    ///< Clear the per-particle sorted-index array.

    const auto num_of_cells = _domain->number_of_cells();
    ///< Total number of hash cells in the domain grid.

    d_cell_start.resize(num_of_cells);
    ///< Allocate per-cell begin offsets.

    d_cell_end.resize(num_of_cells);
    ///< Allocate per-cell end offsets.

    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
    ///< Refresh cached raw pointers after buffer resize operations.
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {
    /**
     * @brief Compute a flattened 32-bit cell key from a 3D grid coordinate.
     *
     * @param ix Cell x-index.
     * @param iy Cell y-index.
     * @param iz Cell z-index.
     * @param gs Grid dimensions.
     * @return Flattened cell key.
     *
     * @details
     * The key is laid out in row-major order with x as the fastest-varying axis.
     * This key is later used for:
     * - sorting particles by cell,
     * - building cell start/end ranges.
     */
    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {
    /**
     * @brief Resize all working buffers for a given number of active particles.
     *
     * @param alive Number of active particles that will participate in hashing.
     *
     * @details
     * This method prepares:
     * - one hash key per active particle,
     * - one sorted particle index per active particle,
     * - one cell-start and one cell-end entry per domain cell.
     *
     * After resizing, cached raw pointers are refreshed so subsequent device
     * kernels can access the storage directly.
     */
    const auto num_of_cells = _domain->number_of_cells();
    ///< Total number of domain cells.

    d_keys.resize(static_cast<std::size_t>(alive));
    ///< Allocate per-particle cell keys.

    d_indices.resize(static_cast<std::size_t>(alive));
    ///< Allocate per-particle sorted indices.

    d_cell_start.resize(num_of_cells);
    ///< Allocate cell begin offsets.

    d_cell_end.resize(num_of_cells);
    ///< Allocate cell end offsets.

    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
    ///< Refresh raw pointer cache after resize.
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(int n_active) {
    /**
     * @brief Initialize the particle index array with `0, 1, ..., n_active-1`.
     *
     * @param n_active Number of active particles.
     *
     * @details
     * This index array is later permuted alongside the hash-key array during
     * sorting so that sorted cell membership can still be mapped back to the
     * original particle indices.
     */
    auto* indices_ptr = atlas::raw_pointer_cast(this->d_indices.data());
    ///< Raw pointer to the per-particle index buffer.

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_active,
        [=] ATLAS_DEVICE(const int i) {
            indices_ptr[i] = i;
            ///< Assign identity mapping before key-based sorting.
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(int alive, const Vector3<T>* pos) {
    /**
     * @brief Compute one spatial hash key per active particle.
     *
     * @param alive Number of active particles.
     * @param pos Pointer to particle positions.
     *
     * @details
     * Each particle is mapped into the domain grid by:
     * 1. transforming world position into cell-space coordinates,
     * 2. applying `floor()` to obtain the containing integer cell,
     * 3. clamping the cell coordinate into valid grid bounds,
     * 4. flattening the 3D cell coordinate into one linear key.
     */
    const atlas::device_ptr<std::uint32_t> keys_ptr(d_keys_ptr);
    ///< Device pointer wrapper around the key buffer.

    const Vector3<T> lc = _domain->lower_corner();
    ///< Domain lower corner.

    const T inv_h = _domain->inverse_cell_size();
    ///< Reciprocal of the grid cell size.

    const Vector3<int> gs = _domain->grid_size();
    ///< Grid resolution in x/y/z.

    const Vector3<int> lo { 0, 0, 0 };
    ///< Minimum valid cell index on each axis.

    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };
    ///< Maximum valid cell index on each axis.

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(int i) {
            const Vector3<T> rel = (pos[i] - lc) * inv_h;
            ///< Particle position expressed in cell-space coordinates.

            Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();
            ///< Integer cell coordinate before clamping.

            ijk = atlas::math::clamp(ijk, lo, hi);
            ///< Clamp into valid grid range to keep every particle in-bounds.

            keys_ptr[i] = linear_key(ijk.x, ijk.y, ijk.z, gs);
            ///< Store the flattened cell key.
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) const {
    /**
     * @brief Sort particle indices by their computed cell keys.
     *
     * @param active Number of active particles.
     *
     * @details
     * After sorting:
     * - `d_keys` becomes grouped by cell,
     * - `d_indices` becomes the permutation that maps sorted slots back to
     *   original particle indices.
     *
     * This grouped representation is then used to build per-cell start/end ranges.
     */
    const atlas::device_ptr<std::uint32_t> keys_begin(d_keys_ptr);
    ///< Beginning of the key range.

    const atlas::device_ptr<std::uint32_t> keys_end = keys_begin + active;
    ///< End of the key range.

    const atlas::device_ptr<int> idx_begin(d_indices_ptr);
    ///< Beginning of the associated particle-index range.

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(int alive) {
    /**
     * @brief Build per-cell begin/end offsets from sorted particle keys.
     *
     * @param alive Number of active particles.
     *
     * @details
     * The method first initializes:
     * - `cell_start[cell] = -1`
     * - `cell_end[cell]   = -1`
     *
     * for every cell.
     *
     * It then scans the sorted key array and marks:
     * - the first occurrence of each key as the cell start,
     * - one-past-the-last occurrence of each key as the cell end.
     *
     * Empty cells remain marked with `-1`.
     */
    const auto num_of_cells = _domain->number_of_cells();
    ///< Total number of cells in the domain grid.

    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_start_ptr,
        d_cell_start_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);
    ///< Initialize every cell start entry to "empty".

    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_end_ptr,
        d_cell_end_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);
    ///< Initialize every cell end entry to "empty".

    const auto* keys = atlas::raw_pointer_cast<std::uint32_t>(this->d_keys_ptr);
    ///< Raw pointer to the sorted cell-key array.

    auto* cell_start = atlas::raw_pointer_cast(this->d_cell_start_ptr);
    ///< Raw pointer to the cell-start array.

    auto* cell_end = atlas::raw_pointer_cast(this->d_cell_end_ptr);
    ///< Raw pointer to the cell-end array.

    const int count = alive;
    ///< Cached active-particle count for device lambda capture.

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_DEVICE(const int i) {
            const std::uint32_t key = keys[i];
            ///< Sorted key at slot `i`.

            if (i == 0 || key != keys[i - 1]) cell_start[key] = i;
            ///< Mark the first slot belonging to this cell.

            if (i == count - 1 || key != keys[i + 1]) cell_end[key] = i + 1;
            ///< Mark one-past-the-last slot belonging to this cell.
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build(const system::FluidDeviceProbe<T>& particle_probe) {
    /**
     * @brief Build the spatial hash structure from a fluid particle probe.
     *
     * @param particle_probe Device-visible particle data.
     *
     * @details
     * The build pipeline is:
     * 1. determine the active particle count,
     * 2. reset to empty if no active particles exist,
     * 3. allocate working buffers,
     * 4. initialize particle indices with an identity permutation,
     * 5. compute one cell key per particle,
     * 6. sort particles by key,
     * 7. derive cell start/end ranges from the sorted keys.
     */
    const int alive = particle_probe.particle_count;
    ///< Number of active particles currently in the fluid probe.

    if (alive <= 0) {
        ///< Empty particle set: reset all structures and exit.
        reset();
        return;
    }

    atlas::logger::info() << "\n"
                          << "Building SpatialHashingSearcher for " << alive << " particles.";

    prepare_buffers(alive);
    ///< Allocate all required buffers for the active-particle count.

    init_indices_iota(alive);
    ///< Initialize `d_indices = [0, 1, 2, ...]`.

    compute_keys(alive, particle_probe.pos);
    ///< Compute the flattened grid cell key of each particle.

    sort_by_key(alive);
    ///< Group particles by cell key.

    build_cell_ranges(alive);
    ///< Convert sorted keys into compact cell begin/end ranges.
}

template <typename T>
SpatialHashingProbe<T>
SpatialHashingSearcher<T>::make_device_probe() noexcept {
    /**
     * @brief Create a device-visible probe exposing the current search structure.
     *
     * @return A populated @ref SpatialHashingProbe object.
     *
     * @details
     * The runtime expects exactly one authoritative probe instance to be created
     * for a searcher. The probe contains only lightweight POD-style state:
     * - domain lower corner,
     * - grid size,
     * - inverse cell size,
     * - cell size,
     * - raw pointers to sorted indices and cell ranges.
     */
    ++_probe_count;
    ///< Track the number of probes created from this searcher.

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    SpatialHashingProbe<T> probe {};
    ///< Default-constructed probe returned if no valid domain exists.

    if (!_domain) return probe;
    ///< Safety guard: an unbound searcher cannot produce a meaningful probe.

    probe.lower_corner = _domain->lower_corner();
    ///< Export the grid origin.

    probe.grid_size = _domain->grid_size();
    ///< Export the grid resolution.

    probe.inv_h = _domain->inverse_cell_size();
    ///< Export reciprocal cell size.

    probe.cell_size = _domain->cell_size();
    ///< Export direct cell size.

    probe.indices = atlas::raw_pointer_cast(d_indices_ptr);
    ///< Export raw pointer to sorted particle indices.

    probe.cell_start = atlas::raw_pointer_cast(d_cell_start_ptr);
    ///< Export raw pointer to per-cell start offsets.

    probe.cell_end = atlas::raw_pointer_cast(d_cell_end_ptr);
    ///< Export raw pointer to per-cell end offsets.

    return probe;
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    /**
     * @brief Store the domain dependency in the builder.
     *
     * @param domain Domain used to define the spatial hash grid.
     * @return Reference to the builder.
     */
    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
SpatialHashingSearcher<T>::Builder::validate() const {
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * The domain is mandatory because the searcher requires it to define
     * the hash grid geometry and allocate cell-based buffers.
     */
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher::Builder: domain must not be null.";
}

template <typename T>
SpatialHashingSearcher<T>
SpatialHashingSearcher<T>::Builder::build() const {
    /**
     * @brief Build a @ref SpatialHashingSearcher by value.
     *
     * @return Constructed spatial hashing searcher.
     */
    validate();
    return SpatialHashingSearcher<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<SpatialHashingSearcher<T>>
SpatialHashingSearcher<T>::Builder::make_host_shared() const {
    /**
     * @brief Build a host-shared spatial hashing searcher.
     *
     * @return Host-shared pointer to the constructed searcher.
     */
    validate();
    return atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);
}

} // namespace atlas::system
