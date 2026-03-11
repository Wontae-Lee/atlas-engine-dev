#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/domain/domain.h>
#include <atlas/searcher/searcher.h>

namespace atlas::system {

/**
 * @brief Lightweight, device-friendly view of the spatial hashing data structure.
 *
 * @details
 * `SpatialHashingProbe` is a **non-owning**, POD-like object meant to be copied by
 * value into GPU kernels (or used in tight CPU loops) without allocations.
 *
 * It exposes exactly the information needed to traverse a uniform-grid spatial
 * hash built by @ref SpatialHashingSearcher:
 *
 * - **Grid geometry**: `lower_corner`, `grid_size`, `inv_h`, `cell_size`
 * - **Neighbor policy**: `range` (single-cell vs. multi-cell neighborhood)
 * - **Cell→particle mapping**: `indices`, `cell_start`, `cell_end`
 *
 * @par Data model
 * The domain is discretized into a regular 3D grid of `grid_size` cells.
 * Each particle position is mapped to an integer cell coordinate:
 * @code
 * rel = (pos - lower_corner) * inv_h;
 * ijk = floor(rel);      // (ix, iy, iz)
 * @endcode
 * The integer coordinate is then converted to a single **linear cell id** (key)
 * using row-major order:
 * @f[
 *   c = ix + iy \cdot g_x + iz \cdot g_x \cdot g_y
 * @f]
 * where @f$(g_x, g_y, g_z) = \text{grid\_size}@f$.
 *
 * Particles are then **sorted by their cell id** (key). The resulting permutation
 * is stored in `indices`, and each cell's contiguous segment in `indices` is stored
 * in `cell_start`/`cell_end`.
 *
 * @par Neighbor iteration
 * Neighbor traversal is performed by enumerating the relevant cells around the
 * query particle (according to `range`), then iterating each cell's contiguous
 * segment in `indices`:
 * @code
 * for k in [cell_start[c], cell_end[c]):
 *   q = indices[k];
 *   // candidate neighbor q
 * @endcode
 *
 * @par Ownership / lifetime
 * This struct **does not own** `indices`, `cell_start`, or `cell_end`.
 * The referenced memory must remain valid for the entire lifetime of any kernel
 * launch or use site where this probe is accessed.
 *
 * @tparam T Floating-point scalar type (typically `float` or `double`).
 *
 * @note
 * - Intended for GPU device code; member functions are annotated with
 *   `ATLAS_ALL_DEVICE` where applicable.
 * - A default-constructed probe is typically "empty" until filled by a searcher.
 *
 * @see SpatialHashingSearcher
 */
template <typename T>
struct SpatialHashingProbe {
    /// @brief Lower (minimum) corner of the domain in world coordinates.
    Vector3<T> lower_corner {};

    /// @brief Grid resolution (cells per axis). Must be positive for valid probes.
    Vector3<int> grid_size { 0, 0, 0 };

    /// @brief Reciprocal cell size, `inv_h = 1 / cell_size`, used for fast world→grid conversion.
    T inv_h = T(1);

    /// @brief Uniform grid cell edge length in world units.
    T cell_size = T(1);

    /**
     * @brief Neighborhood policy that determines which cells are scanned.
     *
     * @details
     * - `NeighborSearchRange::single` typically scans only the cell containing the
     *   query particle (or an immediate local region, depending on implementation).
     * - `NeighborSearchRange::multiple` scans a wider neighborhood and may apply
     *   an additional distance filter.
     *
     * The exact visited set is defined by `for_each_neighbor()`.
     */
    NeighborSearchRange range = NeighborSearchRange::single;

    /**
     * @brief Particle permutation after sorting by cell key.
     *
     * @details
     * `indices[k]` gives the particle id (index into simulation arrays) corresponding
     * to the k-th entry in the key-sorted order.
     *
     * Combined with `cell_start/end`, this provides O(1) access to the contiguous
     * span of particle ids belonging to any cell.
     *
     * @warning Must point to valid device-accessible memory when used in device code.
     */
    const int* indices {};

    /**
     * @brief Start offset (inclusive) into `indices` for each linear cell id.
     *
     * @details
     * For cell id `c`, the particles in that cell occupy:
     * @f[
     *   k \in [\text{cell\_start}[c],\; \text{cell\_end}[c])
     * @f]
     *
     * Empty cells are commonly marked by `cell_start[c] == -1` (implementation detail),
     * or by `cell_start[c] == cell_end[c]`.
     */
    const int* cell_start {};

    /**
     * @brief End offset (exclusive) into `indices` for each linear cell id.
     *
     * @see cell_start
     */
    const int* cell_end {};

    /**
     * @brief Check whether a cell coordinate is inside the grid bounds.
     *
     * @param ix Cell x-index.
     * @param iy Cell y-index.
     * @param iz Cell z-index.
     * @return `true` iff `0 <= ix < grid_size.x`, `0 <= iy < grid_size.y`, `0 <= iz < grid_size.z`.
     *
     * @note
     * This is designed to be branch-friendly and usable on both CPU and GPU.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid_cell(int ix, int iy, int iz) const noexcept;

    /**
     * @brief Convert a 3D cell coordinate to a linear cell id (row-major).
     *
     * @param ix Cell x-index.
     * @param iy Cell y-index.
     * @param iz Cell z-index.
     * @return Linearized cell id (also used as the sorting key).
     *
     * @warning No bounds check is performed. Use @ref is_valid_cell if needed.
     *
     * @see SpatialHashingSearcher::linear_key
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    cell_index(int ix, int iy, int iz) const noexcept;

    /**
     * @brief Iterate over candidate neighbors of particle `p` and invoke a callback.
     *
     * @details
     * This method performs a **cell-based candidate enumeration**:
     * 1) Compute the query particle's cell coordinate from `pos[p]`.
     * 2) Determine which cell(s) to scan based on @ref range.
     * 3) For each scanned cell, iterate its contiguous particle span via
     *    `cell_start/end`, mapping each entry through `indices`.
     * 4) Invoke `func(q)` for every candidate particle index `q`.
     * 5) If `func(q)` returns `true`, the iteration terminates early.
     *
     * @par Candidate vs. true neighbors
     * The method enumerates **candidates**. Depending on policy, it may:
     * - return all particles in the scanned cells (fast, more candidates), or
     * - additionally apply a radius/distance test (fewer candidates).
     *
     * Self-candidates (`q == p`) are typically skipped.
     *
     * @tparam Func Callable type; must be callable from the current compilation
     *         target (device-callable when compiled for CUDA).
     *
     * @param p    Query particle id.
     * @param pos  Pointer to particle positions array (device pointer in GPU kernels).
     * @param func Callback invoked for each candidate neighbor id `q`.
     *
     * @note
     * - Designed to be inlined into kernels for minimal overhead.
     * - The callback should be lightweight; heavy computations belong outside this loop.
     *
     * @warning `indices`, `cell_start`, and `cell_end` must be valid and consistent
     *          with the domain parameters, or behavior is undefined.
     */
    template <typename Func>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    for_each_neighbor(int p, const Vector3<T>* pos, Func&& func) const;
};

/**
 * @brief Uniform-grid spatial hashing neighbor searcher for particle systems.
 *
 * @details
 * `SpatialHashingSearcher` constructs a uniform-grid spatial hash over a set of
 * particles to accelerate neighborhood queries.
 *
 * The searcher owns GPU buffers that store:
 * - a per-particle cell key array (`d_keys`)
 * - a permutation array (`d_indices`) sorted by key
 * - per-cell index ranges (`d_cell_start`, `d_cell_end`)
 *
 * After building, a lightweight @ref SpatialHashingProbe can be produced for
 * device-side traversal without exposing internal ownership.
 *
 * @par Typical usage
 * @code
 * auto searcher = SpatialHashingSearcher<float>::builder()
 *                   .with_domain(domain)
 *                   .with_range(NeighborSearchRange::single)
 *                   .make_host_shared();
 *
 * searcher->build(particle_probe);
 * auto probe = searcher->make_device_probe();
 *
 * // inside a kernel:
 * probe.for_each_neighbor(p, pos, [&](int q) { ... });
 * @endcode
 *
 * @par Build pipeline
 * A typical `build()` performs:
 * 1) Resize/allocate buffers for the current alive particle count and grid cell count.
 * 2) Compute each particle's cell id (`linear_key`) from its position.
 * 3) Sort particles by cell id (permute indices by key).
 * 4) Build `cell_start/end` so each cell maps to a contiguous segment of indices.
 *
 * @par Complexity
 * - Key computation:  @f$\mathcal{O}(N)@f$
 * - Sorting:          depends on backend, commonly @f$\mathcal{O}(N \log N)@f$
 * - Range building:   @f$\mathcal{O}(N + C)@f$ where `C` is number of cells
 *
 * @tparam T Floating-point scalar type (typically `float` or `double`).
 *
 * @note
 * - The searcher stores a host-side domain pointer and uses it to obtain
 *   grid parameters during key computation and probe creation.
 * - The probe returned by `make_device_probe()` is only valid while the searcher’s
 *   internal buffers remain alive and unchanged.
 *
 * @see SpatialHashingProbe, Domain
 */
template <typename T>
class SpatialHashingSearcher final : public Searcher<T> {
public:
    /// @brief Fluent builder for configuring and constructing the searcher.
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an "unbound" searcher. A valid domain must be set via the builder
     * or by using the domain constructor before calling `build()`.
     */
    SpatialHashingSearcher() = default;

    /**
     * @brief Construct a spatial hashing searcher bound to a domain.
     *
     * @param domain Simulation domain defining bounds, grid resolution, and cell size.
     * @param range  Neighborhood scan policy to be used by probes.
     *
     * @pre `domain != nullptr`
     *
     * @note
     * The constructor typically calls `reset()` internally to size cell buffers
     * to the domain's cell count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit SpatialHashingSearcher(
        DomainHostPtr<T> domain,
        NeighborSearchRange range = NeighborSearchRange::single);

    /// @brief Virtual destructor (default).
    ~SpatialHashingSearcher() override = default;

    /**
     * @brief Build the spatial hash for the current set of alive particles.
     *
     * @details
     * Uses `particle_probe` (positions + alive count) to rebuild keys, sort indices,
     * and rebuild cell ranges.
     *
     * After a successful build, @ref make_device_probe produces a valid view for
     * device-side neighbor traversal.
     *
     * @param particle_probe Device-side probe providing at least:
     *        - `alive` number of active particles
     *        - `pos` pointer to positions (device pointer)
     *
     * @note
     * If `particle_probe.alive <= 0`, implementations typically call `reset()`
     * and return early.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const system::ParticleDeviceProbe<T>& particle_probe) override;

    /**
     * @brief Create a device-friendly probe for neighbor traversal.
     *
     * @details
     * Captures domain parameters (lower corner, grid size, inverse cell size, cell size),
     * neighbor range policy, and raw device pointers to internal buffers.
     *
     * @return A non-owning @ref SpatialHashingProbe that can be copied into kernels.
     *
     * @warning The returned probe becomes invalid if:
     *  - this searcher is destroyed,
     *  - buffers are resized (e.g., a later `build()` with a different alive count),
     *  - `reset()` is called.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Reset internal buffers and invalidate the current hash structure.
     *
     * @details
     * Clears key/index buffers and initializes cell range arrays to an empty state
     * sized to the domain's cell count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    // ------------------------------------------------------------
    // Build steps (exposed for testing/benchmarking; may be made private)
    // ------------------------------------------------------------

    /**
     * @brief Resize/allocate internal buffers for `alive` particles and domain cells.
     *
     * @param alive Number of active particles.
     *
     * @details
     * Ensures:
     * - `d_keys.size()     == alive`
     * - `d_indices.size()  == alive`
     * - `d_cell_start/end` sized to `domain->number_of_cells()`
     *
     * Also updates cached raw pointers (`*_ptr`) for fast device access.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_buffers(int alive);

    /**
     * @brief Initialize the permutation array as `indices[i] = i`.
     *
     * @param n_active Number of active particles.
     *
     * @details
     * This provides the identity permutation before sorting by key. Sorting then
     * rearranges `indices` rather than moving particle data arrays.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int n_active);

    /**
     * @brief Compute per-particle linear cell keys from positions.
     *
     * @param alive Number of active particles.
     * @param pos   Pointer to device positions array.
     *
     * @details
     * For each particle, computes a clamped integer cell coordinate and converts
     * it to a linear key via @ref linear_key.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_keys(int alive, const Vector3<T>* pos);

    /**
     * @brief Sort `(key, index)` pairs by key so particles in the same cell become contiguous.
     *
     * @param active Number of active particles.
     *
     * @details
     * After sorting:
     * - `d_keys` is non-decreasing
     * - `d_indices` matches the same permutation, yielding contiguous per-cell spans.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int active) const;

    /**
     * @brief Build per-cell `[start,end)` ranges into the sorted `indices` array.
     *
     * @param alive Number of active particles.
     *
     * @details
     * Initializes all cell ranges as empty (commonly -1), then scans boundaries
     * in the sorted key array to write start and end offsets for each encountered key.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

    /**
     * @brief Convert a 3D cell coordinate into a 32-bit row-major linear key.
     *
     * @param ix Cell x-index.
     * @param iy Cell y-index.
     * @param iz Cell z-index.
     * @param gs Grid size (cells per axis).
     * @return Linear cell key in `[0, gs.x*gs.y*gs.z)`.
     *
     * @note Must be consistent with @ref SpatialHashingProbe::cell_index.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

private:
    /// @brief Diagnostic counter: number of probes created via @ref make_device_probe.
    std::uint64_t _probe_count = 0;

    /// @brief Domain defining bounds and grid discretization (host-side handle).
    DomainHostPtr<T> _domain {};

    /// @brief Neighbor scanning policy used by produced probes.
    NeighborSearchRange _range = NeighborSearchRange::single;

    // -------------------------
    // Owned device buffers
    // -------------------------
    /// @brief Per-particle cell keys (one key per particle_count particle).
    DeviceBuffer<std::uint32_t> d_keys;

    /// @brief Sorted permutation of particle ids (one index per particle_count particle).
    DeviceBuffer<int> d_indices;

    /// @brief Per-cell inclusive start offsets into @ref d_indices.
    DeviceBuffer<int> d_cell_start;

    /// @brief Per-cell exclusive end offsets into @ref d_indices.
    DeviceBuffer<int> d_cell_end;

    // -------------------------
    // Cached raw pointers
    // -------------------------
    /// @brief Raw device pointer to @ref d_keys (cached for kernels/primitives).
    std::uint32_t* d_keys_ptr = nullptr;

    /// @brief Raw device pointer to @ref d_indices (cached for kernels/primitives).
    int* d_indices_ptr = nullptr;

    /// @brief Raw device pointer to @ref d_cell_start (cached for kernels/primitives).
    int* d_cell_start_ptr = nullptr;

    /// @brief Raw device pointer to @ref d_cell_end (cached for kernels/primitives).
    int* d_cell_end_ptr = nullptr;
};

/**
 * @brief Fluent builder for @ref SpatialHashingSearcher.
 *
 * @details
 * The builder configures mandatory and optional parameters, validates them,
 * and constructs either:
 * - a searcher by value (`build()`), or
 * - a shared host instance (`make_host_shared()`).
 *
 * @tparam T Floating-point scalar type.
 *
 * @note
 * `with_domain()` is required; the domain must not be null.
 */
template <typename T>
class SpatialHashingSearcher<T>::Builder final {
public:
    /// @brief Construct an empty builder with default range.
    Builder() = default;

    /**
     * @brief Set the domain used for hashing and grid discretization.
     *
     * @param domain Host pointer to a domain object (must be non-null).
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(DomainHostPtr<T> domain) noexcept;

    /**
     * @brief Set the neighbor scanning policy used by generated probes.
     *
     * @param range Neighborhood policy.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_range(NeighborSearchRange range) noexcept;

    /**
     * @brief Build a configured searcher (returned by value).
     *
     * @return Configured @ref SpatialHashingSearcher instance.
     *
     * @throws std::invalid_argument (or project-specific exception)
     *         if required configuration is missing (e.g., null domain).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingSearcher<T>
    build() const;

    /**
     * @brief Build a configured searcher as a host-shared pointer.
     *
     * @return Shared pointer to a configured @ref SpatialHashingSearcher instance.
     *
     * @throws std::invalid_argument (or project-specific exception)
     *         if required configuration is missing (e.g., null domain).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpatialHashingSearcher<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate builder state and throw if configuration is invalid.
     *
     * @details
     * Typical checks:
     * - `_domain` must be non-null.
     * Additional checks may be added (e.g., domain grid size validity).
     */
    void
    validate_or_throw() const;

private:
    /// @brief Domain handle to be passed into the constructed searcher.
    DomainHostPtr<T> _domain {};

    /// @brief Configured neighborhood policy (defaults to single).
    NeighborSearchRange _range = NeighborSearchRange::single;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for the system spatial hashing searcher.
 *
 * @tparam T Floating-point scalar type.
 *
 * @see atlas::system::SpatialHashingSearcher
 */
template <typename T>
using SpatialHashingSearcher = system::SpatialHashingSearcher<T>;

/**
 * @brief Host-shared pointer alias for @ref SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<system::SpatialHashingSearcher<T>>;

/**
 * @brief Device-shared pointer alias for @ref SpatialHashingSearcher.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<system::SpatialHashingSearcher<T>>;

} // namespace atlas

// Implementation header for templates.
#include <atlas/searcher/spatial_hashing_searcher.hpp>
