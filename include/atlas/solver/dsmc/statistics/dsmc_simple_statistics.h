#pragma once

#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

/**
 * @file dsmc_simple_statistics.h
 * @brief Cheaper NTC candidate-count measurement: bounds `max_sigma_g`
 *        by sampled *relative speed* alone, skipping
 *        `DsmcKernel::sigma_g`'s per-sample cross-section evaluation.
 *
 * @details
 * Identical cell-sampling and `N_candidates` formula to `DsmcStatistics`
 * (see that file), except the per-sample bound update uses
 * `max(max_relative_speed)` directly in place of `max(sigma * g)`. See
 * `dsmc_simple_solver.h`'s top-of-file documentation for why this is
 * still a correct (if less tightly-bounded) NTC estimate: DSMC's
 * accept/reject step in `DsmcSolver::collide_indexed_pair` self-corrects
 * for a loose bound, so this only trades extra rejected candidates for
 * cheaper per-cell statistics, not correctness.
 */

namespace atlas {

/**
 * @brief Relative-speed-only NTC candidate-count measurement (no
 *        per-sample cross-section evaluation). See this file's
 *        top-of-file documentation for the tradeoff against
 *        `DsmcStatistics`.
 */
class DsmcSimpleStatistics final : public DsmcStatistics {
public:
    /** @brief Overrides `DsmcStatistics::measure` with the relative-
     *  speed-only `max_sigma_g` bound. */
    ATLAS_HOST bool
    measure(const DsmcProbe& probe,
            const DeviceBuffer<int>* allocated_solver,
            int index,
            float dt) const override;

public:
    ATLAS_HOST static void
    measure_cells(const DsmcProbe& probe, const int* allocated_solver_ptr, int index, float dt);

private:
};

}
