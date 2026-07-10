#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher_view.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <cstdint>

namespace atlas {

/**
 * @brief Uniform-grid spatial hash that buckets particles into cells for neighbour lookup.
 *
 * Given the particle positions of a `Fluid`, this classifies every particle
 * into a cubic cell of a uniform grid (`grid_size` cells of edge `cell_size`,
 * anchored at `lower_corner`) and produces, on the device, the arrays a DSMC
 * solver needs to iterate the particles of any one cell without scanning the
 * whole population: a sorted particle-index array plus per-cell `[start, end)`
 * ranges into it. The grid is the same one the `Universe` defines, so cell
 * linear keys line up one-to-one with `Universe` cell indices.
 *
 * The classify pipeline is a standard sort-based spatial hash:
 *   1. `compute_keys`      — key[i] = linear cell key of particle i; index[i] = i.
 *   2. `sort_keys`         — sort (key, index) by key so equal-cell particles are contiguous.
 *   3. `build_cell_ranges` — scan run boundaries to fill `cell_start`/`cell_end`.
 *   4. `write_cell_counts` — publish per-cell counts into the `Universe` number-particle field.
 * All four steps run on the GPU. All heavy state lives in `DeviceBuffer`s, so
 * the object is host-only and move-constructed; a device lambda sees it through
 * the trivially-copyable `SpatialHashingSearcherView` returned by `view()`.
 *
 * @note The per-step methods and the static grid helpers are public only so the
 *       `[=] __host__ __device__` lambdas inside them (and inside callers) are
 *       not defined in a private/protected member function, which nvcc forbids
 *       for extended lambdas. They are not meant as a general public API; call
 *       `classify()`.
 */
class SpatialHashingSearcher final {
public:
    /// Fluent builder that gathers grid parameters and validates before constructing.
    class Builder;

public:
    /**
     * @brief Construct an empty 1x1x1 searcher with unit cell size.
     *
     * Calls `reset()` so `cell_start`/`cell_end` are sized to `cell_count()` and filled
     * with the empty sentinel, exactly as the explicit constructor does. Without that the
     * accessors would report one cell while backing no storage, and a consumer reading
     * `cell_count()` entries would run off the end of an empty buffer.
     *
     * Use `builder()` for real grids.
     */
    ATLAS_HOST
    SpatialHashingSearcher();

    /**
     * @brief Construct over an explicit uniform grid and allocate its per-cell arrays.
     *
     * Caches `inverse_cell_size = 1/cell_size` and `cell_count = grid_size.x *
     * grid_size.y * grid_size.z`, then calls `reset()` so `cell_start`/`cell_end`
     * are sized to the cell count and filled with the empty sentinel. No particle
     * arrays are allocated yet; the first `classify()` sizes them.
     *
     * @param lower_corner Minimum corner of the grid, in world units.
     * @param cell_size Edge length of a cubic cell, in world units; must be > 0 (unchecked here).
     * @param grid_size Cell count per axis; each component must be >= 1 (unchecked here).
     */
    ATLAS_HOST
    SpatialHashingSearcher(const Float3& lower_corner,
                           float cell_size,
                           const Int3& grid_size);

    /// @return A fresh, empty `Builder`.
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Bucket `particle_count` particles into the grid and publish per-cell counts.
     *
     * Runs the full four-step pipeline on the device. If the input is unusable —
     * `positions` null, `particle_count <= 0`, or `particle_count` larger than the
     * position buffer — the searcher is `reset()` to the empty state and only the
     * (all-zero) cell counts are written, so the resulting `view()` reports no
     * particles rather than reading out of bounds.
     *
     * @param positions Fluid position field; its device buffer supplies particle
     *        coordinates. May be null (treated as "nothing to classify").
     * @param number_particle Destination for per-cell particle counts (the
     *        `Universe` number-particle field). May be null — then counts are
     *        skipped with a warning; the DSMC solver and Knudsen codec need them.
     * @param particle_count Number of live particles to read from the front of
     *        the position buffer.
     */
    ATLAS_HOST void
    classify(const FluidPositionState* positions,
             UniverseNumberParticleState* number_particle,
             int particle_count);

    /**
     * @brief Return to the "no particles classified" state.
     *
     * Clears the key/index arrays to length 0 and refills every `cell_start`/
     * `cell_end` entry with the -1 empty sentinel. Leaves the grid geometry
     * (corner, size, cell count) untouched.
     */
    ATLAS_HOST void
    reset();

    /// @return The grid's minimum world-space corner.
    ATLAS_NODISCARD ATLAS_HOST Float3
    lower_corner() const noexcept {
        return _lower_corner;
    }

    /// @return The per-axis cell counts of the grid.
    ATLAS_NODISCARD ATLAS_HOST Int3
    grid_size() const noexcept {
        return _grid_size;
    }

    /// @return The cubic cell edge length, in world units.
    ATLAS_NODISCARD ATLAS_HOST float
    cell_size() const noexcept {
        return _cell_size;
    }

    /// @return The cached reciprocal `1/cell_size`, used to map positions to cells.
    ATLAS_NODISCARD ATLAS_HOST float
    inverse_cell_size() const noexcept {
        return _inverse_cell_size;
    }

    /// @return Total number of cells, `grid_size.x * grid_size.y * grid_size.z`.
    ATLAS_NODISCARD ATLAS_HOST int
    cell_count() const noexcept {
        return _cell_count;
    }

    /**
     * @brief Gather the four device pointers into a device-capturable view.
     *
     * @return A `SpatialHashingSearcherView` aliasing this searcher's buffers.
     *         The pointers are valid only until the next `classify()`/`reset()`
     *         or destruction of this object.
     */
    ATLAS_NODISCARD ATLAS_HOST SpatialHashingSearcherView
    view() const noexcept;

    /// @return Device pointer to the sorted linear-cell-key array (one per sorted slot), or null if empty.
    ATLAS_NODISCARD ATLAS_HOST const std::uint32_t*
    cell_key() const noexcept;

    /// @return Device pointer to the sorted original-particle-index array, or null if empty.
    ATLAS_NODISCARD ATLAS_HOST const int*
    indices() const noexcept;

    /// @return Device pointer to the per-cell range-start array (-1 sentinel for empty cells).
    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_start() const noexcept;

    /// @return Device pointer to the per-cell range-end (exclusive) array (-1 sentinel for empty cells).
    ATLAS_NODISCARD ATLAS_HOST const int*
    cell_end() const noexcept;

    /**
     * @brief Flatten integer cell coordinates to a single linear cell key.
     *
     * Row-major (x fastest, then y, then z). Callable on host and device.
     *
     * @param ix Cell index on x, assumed already in `[0, grid_size.x)`.
     * @param iy Cell index on y, assumed already in `[0, grid_size.y)`.
     * @param iz Cell index on z, assumed already in `[0, grid_size.z)`.
     * @param grid_size Per-axis cell counts.
     * @return The linear key `ix + iy*Gx + iz*Gx*Gy`.
     * @note No bounds check; feed only in-range coordinates (e.g. from `cell_for`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const int ix, const int iy, const int iz, const Int3& grid_size) noexcept {
        return static_cast<std::uint32_t>(ix + iy * grid_size.x + iz * grid_size.x * grid_size.y);
    }

    /**
     * @brief `Int3` overload of `linear_key`.
     * @param cell Integer cell coordinates, assumed in range on every axis.
     * @param grid_size Per-axis cell counts.
     * @return The linear key of `cell`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const Int3& cell, const Int3& grid_size) noexcept {
        return linear_key(cell.x, cell.y, cell.z, grid_size);
    }

    /**
     * @brief Map a world position to the integer cell coordinates that contain it.
     *
     * Computes `floor((position - lower_corner) / cell_size)` then clamps to
     * `[0, grid_size-1]` on every axis, so particles on or beyond the boundary
     * are attributed to the nearest edge cell rather than producing an
     * out-of-range key. Callable on host and device.
     *
     * @param position World-space particle position.
     * @param lower_corner Grid minimum corner.
     * @param inverse_cell_size Reciprocal cell edge length (`1/cell_size`).
     * @param grid_size Per-axis cell counts.
     * @return Clamped integer cell coordinates in `[0, grid_size)` on every axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Int3
    cell_for(const Float3& position,
             const Float3& lower_corner,
             const float inverse_cell_size,
             const Int3& grid_size) noexcept {
        const Int3 cell = atlas::to_vector3i(atlas::floor((position - lower_corner) * inverse_cell_size));
        return atlas::clamp(cell, Int3(0, 0, 0), grid_size - Int3(1, 1, 1));
    }

    /**
     * @brief Test whether integer cell coordinates lie inside the grid.
     *
     * Unlike `cell_for`, this does not clamp; it reports membership. Useful when
     * walking to a neighbour cell that may fall outside the domain. Callable on
     * host and device.
     *
     * @param cell Integer cell coordinates to test.
     * @param grid_size Per-axis cell counts.
     * @return True iff `0 <= cell < grid_size` on every axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    contains_cell(const Int3& cell, const Int3& grid_size) noexcept {
        return atlas::all(cell >= Int3(0, 0, 0))
            && atlas::all(cell < grid_size);
    }

    /**
     * @brief Pipeline step 1: compute each particle's cell key and seed its index.
     *
     * For every particle `i` in `[0, alive)`, writes `_keys[i]` = linear key of
     * the cell containing `positions[i]` and `_indices[i] = i`. Runs one device
     * thread per particle. Public because of the nvcc extended-lambda rule; call
     * `classify()` instead.
     *
     * @param alive Number of particles to process; `_keys`/`_indices` must already
     *        be sized to at least `alive`.
     * @param positions Device pointer to the particle positions.
     */
    ATLAS_HOST void
    compute_keys(int alive, const Float3* positions);

    /**
     * @brief Pipeline step 2: sort `(_keys, _indices)` by key so cells are contiguous.
     *
     * A key-sort over the first `alive` entries; afterwards `_keys` is ascending
     * and `_indices` is permuted to match, grouping each cell's particle indices
     * into one contiguous run that `build_cell_ranges` can bound. Public because
     * of the nvcc extended-lambda rule.
     *
     * @param alive Number of valid entries at the front of the arrays.
     */
    ATLAS_HOST void
    sort_keys(int alive);

    /**
     * @brief Pipeline step 3: fill `cell_start`/`cell_end` from the sorted keys.
     *
     * First resets all cell ranges to the -1 sentinel, then, with one thread per
     * sorted particle, detects run boundaries in the sorted `_keys`: a slot that
     * differs from its predecessor opens its cell's range, one that differs from
     * its successor closes it (as an exclusive end). Cells with no particles keep
     * the -1 sentinel. Public because of the nvcc extended-lambda rule.
     *
     * @param alive Number of sorted particles to scan.
     */
    ATLAS_HOST void
    build_cell_ranges(int alive);

    /**
     * @brief Pipeline step 4: publish per-cell particle counts into the Universe field.
     *
     * Writes `count = end - start` (0 for empty cells, whose start is -1) as a
     * `float` into each cell of `number_particle`, on the device. No-ops with a
     * warning if `number_particle` is null or its length differs from
     * `cell_count`. Const because it only reads this searcher's ranges. Public
     * because of the nvcc extended-lambda rule.
     *
     * @param number_particle Destination per-cell count field; may be null.
     */
    ATLAS_HOST void
    write_cell_counts(UniverseNumberParticleState* number_particle) const;

private:
    /// Total cells in the grid; length of `_cell_start`/`_cell_end`. Cached product of `_grid_size`.
    int _cell_count = 1;

    /// Cubic cell edge length in world units; must be positive.
    float _cell_size = 1.0f;

    /// Cached `1/_cell_size` so position-to-cell mapping avoids a per-particle divide.
    float _inverse_cell_size = 1.0f;

    /// World-space minimum corner the grid is anchored at.
    Float3 _lower_corner { 0.0f, 0.0f, 0.0f };

    /// Per-axis cell counts of the grid.
    Int3 _grid_size { 1, 1, 1 };

    /// Per-particle linear cell keys; sorted ascending after `sort_keys`. Sized to particle count.
    DeviceBuffer<std::uint32_t> _keys;

    /// Original particle indices, permuted to match sorted `_keys`. Sized to particle count.
    DeviceBuffer<int> _indices;

    /// Per-cell first sorted slot, -1 if empty. Sized to `_cell_count`.
    DeviceBuffer<int> _cell_start;

    /// Per-cell one-past-last sorted slot (exclusive), -1 if empty. Sized to `_cell_count`.
    DeviceBuffer<int> _cell_end;
};

/**
 * @brief Fluent builder for `SpatialHashingSearcher`.
 *
 * Collects the grid parameters — corner, cell size, per-axis cell counts —
 * through `with_*` setters, `validate()`s them, and constructs the searcher.
 * The parameters usually come straight from the `Universe` via `with_universe`
 * so the hash grid matches the simulation grid exactly.
 */
class SpatialHashingSearcher::Builder final {
public:
    /// Construct a builder holding the default 1x1x1 unit grid.
    Builder() = default;

    /**
     * @brief Copy the grid geometry from a `Universe`.
     *
     * Sets lower corner, cell size and grid size to the universe's, so the
     * searcher's cells coincide with the universe's cells. This is the normal
     * way to configure the builder.
     *
     * @param universe The simulation domain to mirror.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_universe(const Universe& universe);

    /**
     * @brief Override the grid's minimum corner.
     * @param lower_corner World-space corner to anchor the grid at.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_lower_corner(const Float3& lower_corner) noexcept;

    /**
     * @brief Override the cubic cell edge length.
     * @param cell_size World-unit edge length; `validate()` later requires it > 0.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_cell_size(float cell_size) noexcept;

    /**
     * @brief Override the per-axis cell counts.
     * @param grid_size Cells per axis; `validate()` later requires each >= 1.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_grid_size(const Int3& grid_size) noexcept;

    /**
     * @brief Validate the parameters and construct a searcher by value.
     * @return A ready `SpatialHashingSearcher` with its cell arrays allocated.
     * @throws std::invalid_argument if `cell_size <= 0` or any grid axis < 1.
     */
    ATLAS_NODISCARD ATLAS_HOST SpatialHashingSearcher
    build() const;

    /**
     * @brief Build and wrap the searcher in a host shared pointer.
     * @return `host_shared_ptr` owning a freshly built searcher.
     * @throws std::invalid_argument on the same conditions as `build()`.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<SpatialHashingSearcher>
    make_host_shared() const;

private:
    /**
     * @brief Throw unless the accumulated parameters describe a usable grid.
     * @throws std::invalid_argument if `cell_size <= 0` or any component of
     *         `grid_size` is < 1.
     */
    ATLAS_HOST void
    validate() const;

private:
    /// Pending cubic cell edge length, world units.
    float _cell_size = 1.0f;

    /// Pending grid minimum corner, world space.
    Float3 _lower_corner { 0.0f, 0.0f, 0.0f };

    /// Pending per-axis cell counts.
    Int3 _grid_size { 1, 1, 1 };
};

/// Owning host handle to a `SpatialHashingSearcher` (the form the `System` holds).
using SpatialHashingSearcherHostPtr = atlas::host_shared_ptr<SpatialHashingSearcher>;

/// Device-shared-pointer alias for a `SpatialHashingSearcher`, kept for symmetry with other modules.
using SpatialHashingSearcherDevicePtr = atlas::device_shared_ptr<SpatialHashingSearcher>;

}