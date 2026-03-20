#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

#include <cmath>

namespace atlas::system {

// ============================================================
// SpatialHashingProbe
// ============================================================

template <typename T>
bool
SpatialHashingProbe<T>::is_valid_cell(const int ix, const int iy, const int iz) const noexcept {
    // Bounds check in grid coordinates.
    return (ix >= 0 && ix < grid_size.x && iy >= 0 && iy < grid_size.y && iz >= 0 && iz < grid_size.z);
}

template <typename T>
int
SpatialHashingProbe<T>::cell_index(const int ix, const int iy, const int iz) const noexcept {
    // Row-major linearization: x + y*X + z*X*Y.
    return ix + iy * grid_size.x + iz * grid_size.x * grid_size.y;
}

template <typename T>
template <typename Func>
void
SpatialHashingProbe<T>::for_each_neighbor(int p, const Vector3<T>* pos, Func&& func) const {
    // NOTE:
    //  - Uses precomputed cell ranges [cell_start[cell], cell_end[cell]) over `indices`.
    //  - Early-out if `func(q)` returns true.
    //  - Range::single scans only the particle's cell (no distance check).
    //  - Otherwise scans cells overlapping a cube of half-cell radius, then prunes by r^2.

    const Vector3<T> xp = pos[p];

    // Valid ijk bounds.
    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };

    // ------------------------------------------------------------
    // Single-cell mode: iterate all particles in the same cell.
    // ------------------------------------------------------------
    if (range == NeighborSearchRange::single) {
        // Convert position to grid coordinates.
        const Vector3<T> rel = (xp - lower_corner) * inv_h;

        // Clamp to ensure we never index outside the grid.
        Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();
        ijk              = atlas::math::clamp(ijk, lo, hi);

        const int cell  = cell_index(ijk.x, ijk.y, ijk.z);
        const int begin = cell_start[cell];
        if (begin < 0) return; // Cell is empty.

        const int end = cell_end[cell];
        for (int k = begin; k < end; ++k) {
            const int q = indices[k];
            if (q == p) continue;
            if (func(q)) return;
        }
        return;
    }

    // ------------------------------------------------------------
    // Multi-cell mode: scan nearby cells then apply distance pruning.
    // ------------------------------------------------------------
    const T r  = T(0.5) * cell_size; // Search radius (half cell).
    const T r2 = r * r;

    // Compute an AABB around xp in grid coordinates.
    const Vector3<T> rel_min = (xp - Vector3<T> { r, r, r } - lower_corner) * inv_h;
    const Vector3<T> rel_max = (xp + Vector3<T> { r, r, r } - lower_corner) * inv_h;

    Vector3<int> ijk_min = atlas::math::floor(rel_min).template cast_to<int>();
    Vector3<int> ijk_max = atlas::math::floor(rel_max).template cast_to<int>();

    ijk_min = atlas::math::clamp(ijk_min, lo, hi);
    ijk_max = atlas::math::clamp(ijk_max, lo, hi);

    // Iterate candidate cells.
    for (int iz = ijk_min.z; iz <= ijk_max.z; ++iz) {
        for (int iy = ijk_min.y; iy <= ijk_max.y; ++iy) {
            for (int ix = ijk_min.x; ix <= ijk_max.x; ++ix) {
                const int cell  = cell_index(ix, iy, iz);
                const int begin = cell_start[cell];
                if (begin < 0) continue; // Empty cell.

                const int end = cell_end[cell];
                for (int k = begin; k < end; ++k) {
                    const int q = indices[k];
                    if (q == p) continue;

                    // Squared distance to avoid sqrt.
                    const Vector3<T> dx = pos[q] - xp;
                    const T dist2       = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

                    if (dist2 <= r2) {
                        if (func(q)) return;
                    }
                }
            }
        }
    }
}

// ============================================================
// SpatialHashingSearcher
// ============================================================

template <typename T>
SpatialHashingSearcher<T>::SpatialHashingSearcher(DomainHostPtr<T> domain,
                                                  const NeighborSearchRange range)
    : _domain(std::move(domain))
    , _range(range) {

    // Domain is mandatory because grid sizing and mapping depends on it.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher: domain must not be null.";
    reset();
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder
SpatialHashingSearcher<T>::builder() noexcept {
    // Fluent builder entry point.
    return Builder {};
}

template <typename T>
void
SpatialHashingSearcher<T>::reset() noexcept {
    // Reset per-build temporary buffers to empty state and resize cell tables
    // to domain size (so probe generation always has valid pointers).
    d_keys.resize(0);
    d_indices.resize(0);

    const auto num_of_cells = _domain->number_of_cells();
    d_cell_start.resize(num_of_cells);
    d_cell_end.resize(num_of_cells);

    // Cache raw pointers for device kernels.
    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
std::uint32_t
SpatialHashingSearcher<T>::linear_key(const int ix, const int iy, const int iz, const Vector3<int>& gs) noexcept {
    // Same linearization as probe::cell_index, but returned as uint32 key.
    return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
}

template <typename T>
void
SpatialHashingSearcher<T>::prepare_buffers(const int alive) {
    // Allocate per-particle arrays and per-cell tables for the current build.
    const auto num_of_cells = _domain->number_of_cells();

    d_keys.resize(static_cast<std::size_t>(alive));
    d_indices.resize(static_cast<std::size_t>(alive));
    d_cell_start.resize(num_of_cells);
    d_cell_end.resize(num_of_cells);

    d_keys_ptr       = atlas::raw_pointer_cast(d_keys.data());
    d_indices_ptr    = atlas::raw_pointer_cast(d_indices.data());
    d_cell_start_ptr = atlas::raw_pointer_cast(d_cell_start.data());
    d_cell_end_ptr   = atlas::raw_pointer_cast(d_cell_end.data());
}

template <typename T>
void
SpatialHashingSearcher<T>::init_indices_iota(int n_active) {
    // Initialize indices = [0..n_active).
    int* indices_ptr = d_indices_ptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        n_active,
        [=] ATLAS_ALL_DEVICE(const int i) {
            indices_ptr[i] = i;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::compute_keys(int alive, const Vector3<T>* pos) {
    // Compute per-particle cell key from position.
    const atlas::device_ptr<std::uint32_t> keys_ptr(d_keys_ptr);

    const Vector3<T> lc   = _domain->lower_corner();
    const T inv_h         = _domain->inverse_cell_size();
    const Vector3<int> gs = _domain->grid_size();

    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = gs - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(int i) {
            const Vector3<T> rel = (pos[i] - lc) * inv_h;

            // Map to grid coordinates and clamp.
            Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();
            ijk              = atlas::math::clamp(ijk, lo, hi);

            keys_ptr[i] = linear_key(ijk.x, ijk.y, ijk.z, gs);
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::sort_by_key(const int active) const {
    // Sort keys and permute indices to match key order.
    const atlas::device_ptr<std::uint32_t> keys_begin(d_keys_ptr);
    const atlas::device_ptr<std::uint32_t> keys_end = keys_begin + active;
    const atlas::device_ptr<int> idx_begin(d_indices_ptr);

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(keys_begin, keys_end, idx_begin);
}

template <typename T>
void
SpatialHashingSearcher<T>::build_cell_ranges(int alive) {
    // Build [cell_start, cell_end) ranges for each occupied cell.
    // Assumes keys are sorted.
    const auto num_of_cells = _domain->number_of_cells();

    // Initialize all cells to empty markers.
    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_start_ptr,
        d_cell_start_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    atlas::parallel_fill<ExecutionPolicy::device>(
        d_cell_end_ptr,
        d_cell_end_ptr + static_cast<std::ptrdiff_t>(num_of_cells),
        -1);

    const std::uint32_t* keys = atlas::raw_pointer_cast<std::uint32_t>(d_keys_ptr);
    int* cell_start           = atlas::raw_pointer_cast(d_cell_start_ptr);
    int* cell_end             = atlas::raw_pointer_cast(d_cell_end_ptr);
    const int count           = alive;

    // Mark boundaries whenever the key changes.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const std::uint32_t key = keys[i];
            if (i == 0 || key != keys[i - 1]) cell_start[key] = i;
            if (i == count - 1 || key != keys[i + 1]) cell_end[key] = i + 1;
        });
}

template <typename T>
void
SpatialHashingSearcher<T>::build(const system::ParticleDeviceProbe<T>& particle_probe) {
    // ------------------------------------------------------------
    // Rebuild the spatial hash from scratch for the current frame/step.
    //
    // Goal:
    //   Build a cell -> [begin,end) index range table so that neighbor queries
    //   can iterate only the particles that fall into relevant grid cells.
    //
    // Data layout after build():
    //   - d_indices : permutation of particle ids [0..particle_count)
    //   - d_keys    : cell key for each entry in d_indices (sorted by key)
    //   - d_cell_start[key], d_cell_end[key]:
    //         contiguous range in d_indices that belongs to cell 'key'
    //
    // Neighbor query then becomes:
    //   for a cell key:
    //     for k in [cell_start[key], cell_end[key]):
    //         q = indices[k]
    //         ... test / accept neighbor q ...
    //
    // Complexity:
    //   - Key computation:   O(particle_count)
    //   - Sort by key:       O(particle_count log particle_count) (or near-linear depending on backend)
    //   - Range building:    O(particle_count) + O(num_cells) for init
    //
    // All heavy work is done on the device via parallel primitives.
    // ------------------------------------------------------------

    // Number of active/particle_count particles to be inserted into the hash grid.
    const int alive = particle_probe.particle_count;

    // Early out: if no particles are particle_count, keep data structures consistent
    // by clearing buffers and leaving cell ranges empty.
    if (alive <= 0) {
        reset(); // Clears keys/indices and sizes cell arrays to domain cell count.
        return;
    }

    // Optional diagnostic: log rebuild size (useful for profiling / sanity checks).
    atlas::logger::info() << "\n"
                          << "Building SpatialHashingSearcher for " << alive << " particles.";

    // ------------------------------------------------------------
    // Step 0) Allocate / resize buffers for this build.
    //
    // We size d_keys and d_indices to 'particle_count', and cell arrays to 'num_cells'.
    // This ensures all subsequent device kernels have valid contiguous storage.
    // ------------------------------------------------------------
    prepare_buffers(alive);

    // ------------------------------------------------------------
    // Step 1) Initialize the index array to an identity mapping.
    //
    // d_indices[i] = i for i in [0, particle_count)
    //
    // Why keep an index array?
    //   We will sort by cell key; instead of moving particle data, we sort indices.
    //   This keeps particle arrays (pos, vel, etc.) untouched and cheap to access.
    // ------------------------------------------------------------
    init_indices_iota(alive);

    // ------------------------------------------------------------
    // Step 2) Compute a spatial hash key per particle.
    //
    // For each particle i:
    //   - Convert world position -> grid coordinates:
    //       rel = (pos[i] - lower_corner) * inv_h
    //   - Convert to integer cell coordinate:
    //       ijk = floor(rel)
    //   - Clamp ijk to [0 .. grid_size-1] to keep positions within the domain grid.
    //   - Convert ijk -> linear cell key:
    //       key = ix + iy*gs.x + iz*gs.x*gs.y
    //
    // Result:
    //   d_keys[i] holds the cell id (key) for particle i.
    // ------------------------------------------------------------
    compute_keys(alive, particle_probe.pos);

    // ------------------------------------------------------------
    // Step 3) Sort particles by their cell key.
    //
    // We perform a key-value sort:
    //   keys:    d_keys
    //   values:  d_indices
    //
    // After sorting:
    //   - d_keys is non-decreasing
    //   - d_indices is permuted in the same way, so particles that share a cell
    //     become contiguous in d_indices.
    //
    // Important:
    //   We do NOT reorder particle positions; we only reorder indices.
    // ------------------------------------------------------------
    sort_by_key(alive);

    // ------------------------------------------------------------
    // Step 4) Build per-cell ranges (start/end offsets into the sorted index list).
    //
    // We produce:
    //   d_cell_start[key] = first position in the sorted arrays where d_keys == key
    //   d_cell_end[key]   = one-past-last position where d_keys == key
    //
    // Implementation strategy:
    //   - Initialize cell_start/end to -1 for all cells (meaning: empty cell).
    //   - In parallel over sorted i:
    //       if i==0 or keys[i] != keys[i-1], then start[keys[i]] = i
    //       if i==last or keys[i] != keys[i+1], then end[keys[i]] = i+1
    //
    // With these two arrays, a query for a given cell is O(1) to locate its
    // particle span, and O(k) to iterate k particles in that cell.
    // ------------------------------------------------------------
    build_cell_ranges(alive);
}

template <typename T>
SpatialHashingProbe<T>
SpatialHashingSearcher<T>::make_device_probe() noexcept {
    // Build a POD-like "view" (probe) of the searcher state that is safe to copy
    // into CUDA kernels / device lambdas.
    //
    // Key idea:
    // - SpatialHashingSearcher owns the device buffers (keys/indices/ranges).
    // - SpatialHashingProbe only *references* those buffers via raw pointers,
    //   plus cached domain parameters needed for world->cell mapping.
    // - This keeps kernels lightweight (no virtual calls, no owning containers).

    // ------------------------------------------------------------
    // Diagnostic: ensure we don't accidentally create multiple probes.
    // ------------------------------------------------------------
    // The probe stores raw pointers into this object's internal buffers.
    // If multiple probes are created and outlive rebuild/reset cycles, it's easy
    // to accidentally use a stale probe (dangling pointers / mismatched ranges).
    //
    // By enforcing "exactly one" probe per system (typical design: the system
    // creates a single probe once per build/update step), we reduce the chance
    // of pointer lifetime bugs and make misuse obvious during development.
    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The domain device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Start with a default/empty probe.
    // If we early-out, this "null probe" is still safe to pass around, but it
    // will not report any valid neighbors because its pointers are null and
    // grid_size is (0,0,0) (depending on your default constructor).
    SpatialHashingProbe<T> probe {};
    if (!_domain) return probe; // No domain => cannot define grid mapping.

    // ------------------------------------------------------------
    // Cache domain parameters needed by device-side neighbor traversal.
    // ------------------------------------------------------------
    // These are copied by value into the probe so that kernels don't need to
    // dereference the domain pointer (which may not be device-resident).
    //
    // lower_corner : converts world positions to domain-relative coordinates.
    // grid_size    : bounds for clamping cell coordinates and linear indexing.
    // inv_h        : precomputed 1/h to replace division with multiplication.
    // cell_size    : used to define the neighbor query radius for multi-cell mode.
    probe.lower_corner = _domain->lower_corner();
    probe.grid_size    = _domain->grid_size();
    probe.inv_h        = _domain->inverse_cell_size();
    probe.cell_size    = _domain->cell_size();

    // Neighborhood policy (single-cell vs. radius-based multi-cell search).
    // The probe uses this to decide whether to scan only the containing cell
    // or also adjacent/overlapping cells (and whether to apply a distance filter).
    probe.range = _range;

    // ------------------------------------------------------------
    // Attach raw device pointers for fast cell iteration.
    // ------------------------------------------------------------
    // Memory layout after build():
    // - d_indices : permutation of particle ids, sorted by cell key.
    // - d_cell_start/end : per-cell half-open ranges into d_indices.
    //
    // For any linear cell id 'c':
    //   k in [cell_start[c], cell_end[c]) are entries in d_indices belonging to cell 'c'.
    // This makes "iterate all particles in cell" a simple contiguous loop.
    //
    // Note:
    // - We intentionally store raw pointers (not device_ptr wrappers) inside the probe
    //   because the probe is meant to be trivially copyable and minimal.
    // - These pointers remain valid only as long as this searcher instance is particle_count
    //   and its buffers are not rebuilt/reset/resized.
    probe.indices    = atlas::raw_pointer_cast(d_indices_ptr);
    probe.cell_start = atlas::raw_pointer_cast(d_cell_start_ptr);
    probe.cell_end   = atlas::raw_pointer_cast(d_cell_end_ptr);

    return probe;
}

// ============================================================
// Builder
// ============================================================

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Store the simulation domain (grid bounds + discretization).
    // - This is a required dependency: the searcher needs the domain's
    //   lower/upper corners, grid resolution, and cell size to hash particles.
    // - We take ownership via move to avoid ref-count churn / extra copies.
    _domain = std::move(domain);
    return *this; // Enable fluent chaining: builder.with_domain(...).with_range(...).
}

template <typename T>
typename SpatialHashingSearcher<T>::Builder&
SpatialHashingSearcher<T>::Builder::with_range(const NeighborSearchRange range) noexcept {
    // Configure the neighbor query mode.
    // Typical semantics:
    //  - NeighborSearchRange::single : query only the particle's own cell
    //  - otherwise                  : query a small neighborhood region and prune by distance
    // (Exact behavior is implemented in SpatialHashingProbe::for_each_neighbor.)
    _range = range;
    return *this; // Fluent chaining.
}

template <typename T>
void
SpatialHashingSearcher<T>::Builder::validate() const {
    // Validate builder state before constructing an instance.
    // We fail early with a clear message instead of letting a null domain
    // crash later when building buffers or computing hash keys.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SpatialHashingSearcher::Builder: domain must not be null.";
}

template <typename T>
SpatialHashingSearcher<T>
SpatialHashingSearcher<T>::Builder::build() const {
    // Construct a value-type SpatialHashingSearcher.
    // - Performs validation first.
    // - Passes the configured domain and range into the searcher constructor.
    // NOTE: This returns by value (stack object). If you want shared ownership,
    // use make_host_shared().
    validate();
    return SpatialHashingSearcher<T>(_domain, _range);
}

template <typename T>
atlas::host_shared_ptr<SpatialHashingSearcher<T>>
SpatialHashingSearcher<T>::Builder::make_host_shared() const {
    // Construct a heap-allocated SpatialHashingSearcher wrapped in host_shared_ptr.
    // Useful when:
    //  - the searcher is shared across multiple systems/components
    //  - lifetime should be managed via reference counting
    // Validation is performed to keep error behavior consistent with build().
    validate();
    return atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain, _range);
}

} // namespace atlas::system
