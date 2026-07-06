#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>

/**
 * @file dsmc_flatten_workload.h
 * @brief Builds a load-balanced global candidate list out of each cell's
 *        (highly non-uniform) NTC candidate count, for
 *        `DsmcCollisionWorkloadType::flatten`.
 *
 * @details
 * ### Background
 * Under `DsmcCollisionWorkloadType::cell`, one device thread handles all
 * of a cell's NTC candidates serially. If particle density (and hence
 * candidate count) varies a lot between cells — common near shocks,
 * inlets, or boundary layers — some threads do far more work than
 * others, and the launch's wall-clock time is bottlenecked by the
 * busiest cell. `DsmcFlattenWorkload` instead concatenates every cell's
 * candidates into one flat, globally-indexed list before launch (an
 * exclusive prefix sum over per-cell counts gives each cell's starting
 * offset — `collision_offsets` — and `collision_cells` maps each flat
 * index back to its owning cell), so the collision kernel can be
 * launched with one thread per *candidate* instead of per *cell*,
 * distributing work evenly regardless of cell occupancy.
 */

namespace atlas {

/**
 * @brief Owns the flattened (cell -> global-index) candidate mapping
 *        consumed by `DsmcSolver::apply_flattened_collision`. See this
 *        file's top-of-file documentation for why it exists.
 */
class DsmcFlattenWorkload final {
public:
    /** Exclusive prefix sum of per-cell candidate counts: cell `c`'s
     *  candidates occupy flat indices `[collision_offsets[c],
     *  collision_offsets[c] + count[c])`. */
    atlas::DeviceBuffer<int> collision_offsets {};
    /** Owning cell index for each flat candidate slot (inverse of
     *  `collision_offsets`, precomputed so a kernel thread can recover
     *  its cell in O(1)). */
    atlas::DeviceBuffer<int> collision_cells {};
    /** Per-cell candidate counts actually used to build this flattening
     *  (post any solver/allocation filtering). */
    atlas::DeviceBuffer<int> filtered_collision_counts {};
    /** Single-element scratch buffer holding the total candidate count
     *  (device-side reduction target). */
    atlas::DeviceBuffer<int> total_count_buffer {};
    /** Total number of flattened candidates (mirrors
     *  `total_count_buffer`, cached host-side after `build()`). */
    int flattened_collision_count {};

    /** @brief The exclusive-prefix-sum offsets buffer (`collision_offsets`). */
    ATLAS_HOST ATLAS_NODISCARD const atlas::DeviceBuffer<int>&
    offsets() const noexcept;

    /** @brief Releases all buffers and resets `flattened_collision_count`
     *  to `0`. */
    ATLAS_HOST void
    clear();

    /**
     * @brief Rebuilds the flattening from `collision_count_ptr` (one
     *        entry per cell), optionally restricted to the cells
     *        selected by `allocated_solver_ptr`/`index` (see
     *        `Solver::solve`'s multi-solver partitioning).
     * @return `false` (after clearing all buffers) if `cell_count <= 0`,
     *         `collision_count_ptr` is null, or the (possibly filtered)
     *         total candidate count is zero — nothing to flatten/launch;
     *         `true` otherwise. `collision_cells` is derived from
     *         `collision_offsets` via a per-candidate binary search
     *         (upper-bound over the exclusive prefix sum).
     */
    ATLAS_HOST bool
    build(int* collision_count_ptr,
          int cell_count,
          const int* allocated_solver_ptr = nullptr,
          int index                       = 0);
};

}
