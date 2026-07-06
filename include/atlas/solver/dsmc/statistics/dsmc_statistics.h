#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_probe.h>

/**
 * @file dsmc_statistics.h
 * @brief Default (species-aware) implementation of the per-cell NTC
 *        candidate-count measurement `DsmcSolver::measure_collision_statistics`
 *        delegates to. See `dsmc_solver.h`'s top-of-file documentation
 *        for the `N_candidates` formula this computes.
 *
 * @details
 * Per cell, per step: samples a handful of candidate pairs (exhaustively
 * for small cells, `count < 5`; 8 hashed-random pairs otherwise, trading
 * exact-max for a cheap probabilistic estimate in dense cells) and, for
 * each, evaluates the *actual* `DsmcKernel::sigma_g` (cross-section times
 * relative speed) for that pair's real species — raising the cell's
 * running `max_sigma_g` bound if any sample exceeds it. From the
 * resulting bound, computes `N_candidates = C(count, 2) * max_sigma_g *
 * statistical_weight * dt / cell_volume` (`C(count,2) = count*(count-1)/2`,
 * the number of distinct pairs in the cell), carries the fractional part
 * across steps in `collision_remainder_ptr`, and writes the integer part
 * to `collision_count_ptr` for `DsmcSolver::apply_collision` to draw and
 * test. See `DsmcSimpleStatistics` for a cheaper variant that skips the
 * per-sample cross-section evaluation.
 */

namespace atlas {

/**
 * @brief Species-aware NTC candidate-count measurement (samples the true
 *        `sigma * g` per candidate pair). See this file's top-of-file
 *        documentation for the full per-cell algorithm.
 */
class DsmcStatistics {
public:
    ATLAS_HOST virtual ~DsmcStatistics() = default;

    /**
     * @brief Updates every cell's (or, if `allocated_solver` is given,
     *        only this solver's allocated cells') NTC candidate-count
     *        bookkeeping in `probe` for timestep `dt`.
     * @return `true` (this base implementation always succeeds).
     */
    ATLAS_HOST virtual bool
    measure(const DsmcProbe& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            float dt) const;

public:
    ATLAS_HOST static void
    measure_cells(const DsmcProbe& probe, const int* allocated_solver_ptr, int index, float dt);

private:
};

}
