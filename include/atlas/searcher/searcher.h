#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/universe/universe.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

/**
 * @file searcher.h
 * @brief Host-only base class for particle spatial partitioning: sorts
 *        particles into `Universe` grid cells (the shared machinery
 *        every `Searcher` subclass uses, regardless of its own
 *        secondary structure — k-d tree, octree, quadtree), plus the
 *        generic kernel-radius neighbor-list builder `SphSolver`
 *        consumes.
 *
 * @details
 * ### Operating principle — cell sort (`build()`'s shared pipeline)
 * Every concrete searcher's `build()` follows the same sequence over
 * this base class's protected helpers:
 * 1. `init_indices_iota`: `_indices[i] = i` for every alive particle —
 *    the array that will end up holding particle indices grouped by
 *    cell.
 * 2. `compute_grid_keys`: assigns each particle a linear cell key
 *    (`linear_key`, a flattened `x + y*nx + z*nx*ny` index — the
 *    standard 3D-to-1D grid flattening) via `cell_for` (world position
 *    -> clamped grid cell).
 * 3. `sort_by_key`: sorts `_indices` by cell key — after this,
 *    `_indices` is a permutation of `[0, alive)` grouped contiguously by
 *    cell (a counting-sort-equivalent result via key sort, not an
 *    explicit bucket pass).
 * 4. `build_cell_ranges`: derives `_cell_start`/`_cell_end` (each
 *    cell's `[start, end)` slice of the now cell-sorted `_indices`)
 *    from the sorted keys.
 * This gives every subclass the same `indices()`/`cell_start()`/
 * `cell_end()` triple (already used throughout Atlas —
 * `DsmcProbe`/`SphProbe`/`SinkProbe`/`SourceProbe`/`CodecProbe` all
 * consume exactly this layout) regardless of whether the subclass then
 * layers a secondary acceleration structure (a k-d tree, octree, or
 * quadtree) on top for its own specialized queries.
 *
 * ### Operating principle — kernel-radius neighbor lists (`build_cell_neighbors`)
 * SPH needs, for every particle, every other particle within the
 * kernel's support radius (`sqrt(radius_squared) = cell_size`, so
 * candidates only need checking in the `3x3x3` block of cells around a
 * particle's own cell — assuming the grid's cell size equals the
 * kernel support). This is a classic two-pass CSR (compressed sparse
 * row) construction, the same shape as
 * `DsmcFlattenWorkload`/`Sink::compact_fluid_particles`'s
 * count-then-scan-then-write pattern:
 * 1. `detail::SearcherNeighborCount` (one thread per particle): walks
 *    the `3x3x3` neighbor-cell block, counts candidates within
 *    `radius_squared` that also pass the caller-supplied
 *    `CandidateFilter` (a compile-time functor — e.g. species
 *    filtering — so different callers can reuse this same neighbor
 *    walk with different inclusion criteria without runtime branching
 *    overhead), and writes each particle's neighbor count.
 * 2. `atlas::exclusive_scan` turns per-particle counts into per-particle
 *    write offsets — the same offset-computation idea as
 *    `DsmcFlattenWorkload::build`.
 * 3. `detail::SearcherNeighborWrite` re-walks the identical neighbor
 *    search (deliberately duplicating pass 1's traversal rather than
 *    caching candidates from it, trading recomputation for not needing
 *    an intermediate unbounded-size buffer) and scatters each accepted
 *    neighbor index into its particle's slice of `_neighbor_indices`.
 * The resulting `neighbor_offsets()`/`neighbor_indices()` pair is
 * exactly the CSR neighbor list `SphProbe`/`sph_solver.h` iterate.
 */

namespace atlas {

namespace detail {

    template <typename CandidateFilter>
    struct SearcherNeighborCount;

    template <typename CandidateFilter>
    struct SearcherNeighborWrite;

    struct SearcherNeighborTotal;

}

/**
 * @brief Host-only base class implementing the shared cell-sort and
 *        kernel-radius-neighbor-list machinery every concrete searcher
 *        (`SpatialHashingSearcher`, `KdTreeSearcher`, `OctreeSearcher`,
 *        `QuadtreeSearcher`) builds on. See this file's top-of-file
 *        documentation for both pipelines.
 */
class Searcher {
public:
    Searcher() = default;

    ATLAS_HOST
    Searcher(UniverseHostPtr universe, FluidHostPtr fluid);

    virtual ~Searcher() = default;

    Searcher(const Searcher&) = default;
    Searcher&
    operator=(const Searcher&)
        = default;
    Searcher(Searcher&&) noexcept = default;
    Searcher&
    operator=(Searcher&&) noexcept = default;

    /** @brief (Re)builds the spatial partition from the current fluid
     *  particle positions; see this file's top-of-file cell-sort
     *  pipeline. */
    ATLAS_HOST virtual void
    build()
        = 0;

    /** @brief Marks the partition stale so the next `build()` actually
     *  recomputes it (implementations may skip rebuilding while not
     *  invalidated, as an optimization when positions haven't changed). */
    ATLAS_HOST virtual void
    invalidate() noexcept;

    /** @brief Clears all buffers back to an empty state. */
    ATLAS_HOST virtual void
    reset() noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual Float3
    lower_corner() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual Int3
    grid_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    inverse_cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    indices() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    cell_start() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    cell_end() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    neighbor_offsets() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual const int*
    neighbor_indices() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST virtual int
    neighbor_count() const noexcept;

    /** @brief Flattens a 3D cell index into a single grid-linear key
     *  (`x + y*nx + z*nx*ny`), used both to sort particles by cell and
     *  to index `cell_start`/`cell_end`. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const int ix, const int iy, const int iz, const Int3& gs) noexcept {
        return static_cast<std::uint32_t>(ix + iy * gs.x + iz * gs.x * gs.y);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const Int3& cell, const Int3& gs) noexcept {
        return linear_key(cell.x, cell.y, cell.z, gs);
    }

    /** @brief World-space position to (clamped-in-range) grid cell
     *  index; the shared position-to-cell mapping every searcher/
     *  consumer uses. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Int3
    cell_for(const Float3& position,
             const Float3& lower_corner,
             const float inverse_cell_size,
             const Int3& grid_size) noexcept {
        const Int3 cell = atlas::to_vector3i(atlas::floor((position - lower_corner) * inverse_cell_size));
        return atlas::clamp(cell, Int3(0, 0, 0), grid_size - Int3(1, 1, 1));
    }

    /** @brief Whether `cell` is within `[0, grid_size)` on every axis. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    contains_cell(const Int3& cell, const Int3& grid_size) noexcept {
        return atlas::all(cell >= Int3(0, 0, 0))
            && atlas::all(cell < grid_size);
    }

    /** @brief Number of grid cells a search radius `length` spans,
     *  rounded up (`ceil(length / cell_size)`) — the neighbor-cell
     *  block radius to walk for a query of that reach. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static int
    search_radius_for(const float length, const float cell_size) noexcept {
        return static_cast<int>(std::ceil(length / cell_size));
    }

protected:
    /** @brief Throws if `_universe`/`_fluid` is null (`owner` names the
     *  caller for the error message). */
    ATLAS_HOST void
    validate_dependencies(const char* owner) const;

    ATLAS_HOST const Float3*
    position_ptr() const noexcept;

    /** @brief Current live particle count from `_fluid`. */
    ATLAS_HOST int
    active_count() const noexcept;

    /** @brief Resizes `_keys`/`_indices` to `alive` and the per-cell
     *  `_cell_start`/`_cell_end` to the universe's cell count. */
    ATLAS_HOST void
    prepare_grid_buffers(int alive);

public:
    /** @brief `_indices[i] = i` for `i` in `[0, alive)` — cell-sort
     *  pipeline step 1; see this file's top-of-file documentation. */
    ATLAS_HOST void
    init_indices_iota(int alive);

    /** @brief Assigns each particle's grid-linear key from `positions`
     *  — cell-sort pipeline step 2. */
    ATLAS_HOST void
    compute_grid_keys(int alive, const Float3* positions);

protected:
    /** @brief Sorts `_indices` by `_keys` — cell-sort pipeline step 3. */
    ATLAS_HOST void
    sort_by_key(int alive);

public:
    /** @brief Derives `_cell_start`/`_cell_end` from the now cell-sorted
     *  `_indices` — cell-sort pipeline step 4. */
    ATLAS_HOST void
    build_cell_ranges(int alive);

protected:
    /**
     * @brief Builds the kernel-radius (`cell_size`) neighbor list via
     *        the two-pass count/scan/write CSR construction described
     *        in this file's top-of-file documentation. `filter` is
     *        applied to every candidate pair alongside the radius test,
     *        letting callers (e.g. species-restricted neighbor queries)
     *        reuse this same cell walk with their own inclusion rule.
     */
    template <typename CandidateFilter>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_neighbors(const int alive, const Float3* positions, CandidateFilter filter) {
        _neighbor_offsets.resize(static_cast<std::size_t>(alive + 1));
        _neighbor_counts.resize(static_cast<std::size_t>(alive));

        auto* counts        = atlas::raw_pointer_cast(_neighbor_counts.data());
        const auto* indices = atlas::raw_pointer_cast(_indices.data());
        const auto* start   = atlas::raw_pointer_cast(_cell_start.data());
        const auto* end     = atlas::raw_pointer_cast(_cell_end.data());
        const Float3 lc     = _universe->lower_corner();
        const float inv_h   = _universe->inverse_cell_size();
        const float radius2 = _universe->cell_size() * _universe->cell_size();
        const Int3 gs       = _universe->grid_size();

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            alive,
            detail::SearcherNeighborCount<CandidateFilter> {
                counts,
                indices,
                start,
                end,
                positions,
                filter,
                lc,
                inv_h,
                radius2,
                gs });

        atlas::exclusive_scan<ExecutionPolicy::device>(
            _neighbor_counts.begin(),
            _neighbor_counts.begin() + static_cast<std::ptrdiff_t>(alive),
            _neighbor_offsets.begin(),
            0);

        const int total = finalize_neighbor_offsets(alive);
        _neighbor_count = total;
        _neighbor_indices.resize(static_cast<std::size_t>(total));
        if (total == 0) {
            return;
        }

        auto* neighbors     = atlas::raw_pointer_cast(_neighbor_indices.data());
        const auto* offsets = atlas::raw_pointer_cast(_neighbor_offsets.data());
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            alive,
            detail::SearcherNeighborWrite<CandidateFilter> {
                neighbors,
                offsets,
                indices,
                start,
                end,
                positions,
                filter,
                lc,
                inv_h,
                radius2,
                gs });
    }

    /** @brief Completes the exclusive-scan offsets with the total
     *  neighbor count (`offsets[alive]`), returning that total. */
    ATLAS_HOST int
    finalize_neighbor_offsets(int alive);

protected:
    /** @brief Clears the neighbor-list buffers back to empty. */
    ATLAS_HOST void
    clear_neighbors();

protected:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    DeviceBuffer<std::uint32_t> _keys;

    DeviceBuffer<int> _indices;

    DeviceBuffer<int> _cell_start;

    DeviceBuffer<int> _cell_end;

    DeviceBuffer<int> _neighbor_offsets;

    DeviceBuffer<int> _neighbor_indices;

    DeviceBuffer<int> _neighbor_counts;

    DeviceBuffer<int> _neighbor_total_count;

    int _neighbor_count {};

    bool _is_invalidated { true };
};

namespace detail {

    /** @brief Pass 1 of `build_cell_neighbors`: counts, per particle,
     *  how many `3x3x3`-neighbor-cell candidates pass the radius test
     *  and `CandidateFilter`. */
    template <typename CandidateFilter>
    struct SearcherNeighborCount {
        int* counts {};
        const int* indices {};
        const int* start {};
        const int* end {};
        const Float3* positions {};
        CandidateFilter filter {};
        Float3 lower_corner {};
        float inverse_cell_size {};
        float radius_squared {};
        Int3 grid_size {};

        ATLAS_ALL_DEVICE void
        operator()(const int i) const {
            const Float3 pi = positions[i];
            const Int3 cell = Searcher::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

            int count = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Int3 neighbor_cell = cell + Int3(dx, dy, dz);
                        if (!Searcher::contains_cell(neighbor_cell, grid_size)) continue;

                        const std::uint32_t key = Searcher::linear_key(neighbor_cell, grid_size);
                        const int first         = start[key];
                        if (first < 0) continue;

                        const int last = end[key];
                        for (int cursor = first; cursor < last; ++cursor) {
                            const int j = indices[cursor];
                            if (j == i) continue;

                            const Float3 pj = positions[j];
                            if (!filter(i, j, pi, pj)) continue;

                            const Float3 delta = pj - pi;
                            if (delta.length_squared() <= radius_squared) {
                                ++count;
                            }
                        }
                    }
                }
            }

            counts[i] = count;
        }
    };

    /** @brief Pass 3 of `build_cell_neighbors`: re-walks the identical
     *  neighbor search and scatters accepted neighbor indices into
     *  each particle's `[offsets[i], offsets[i+1])` slice. */
    template <typename CandidateFilter>
    struct SearcherNeighborWrite {
        int* neighbors {};
        const int* offsets {};
        const int* indices {};
        const int* start {};
        const int* end {};
        const Float3* positions {};
        CandidateFilter filter {};
        Float3 lower_corner {};
        float inverse_cell_size {};
        float radius_squared {};
        Int3 grid_size {};

        ATLAS_ALL_DEVICE void
        operator()(const int i) const {
            const Float3 pi = positions[i];
            const Int3 cell = Searcher::cell_for(pi, lower_corner, inverse_cell_size, grid_size);

            int write = offsets[i];
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Int3 neighbor_cell = cell + Int3(dx, dy, dz);
                        if (!Searcher::contains_cell(neighbor_cell, grid_size)) continue;

                        const std::uint32_t key = Searcher::linear_key(neighbor_cell, grid_size);
                        const int first         = start[key];
                        if (first < 0) continue;

                        const int last = end[key];
                        for (int cursor = first; cursor < last; ++cursor) {
                            const int j = indices[cursor];
                            if (j == i) continue;

                            const Float3 pj = positions[j];
                            if (!filter(i, j, pi, pj)) continue;

                            const Float3 delta = pj - pi;
                            if (delta.length_squared() <= radius_squared) {
                                neighbors[write++] = j;
                            }
                        }
                    }
                }
            }
        }
    };

    /** @brief Single-thread functor computing the final total neighbor
     *  count and closing out the offsets array, used by
     *  `finalize_neighbor_offsets`. */
    struct SearcherNeighborTotal {
        int* total {};
        int* offsets {};
        const int* counts {};
        int alive {};
        int last {};

        ATLAS_ALL_DEVICE void
        operator()(int) const {
            total[0]       = offsets[last] + counts[last];
            offsets[alive] = total[0];
        }
    };

}

using SearcherHostPtr = atlas::host_shared_ptr<Searcher>;

using SearcherDevicePtr = atlas::device_shared_ptr<Searcher>;

}
