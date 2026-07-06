#pragma once

#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>

/**
 * @file dsmc_simple_solver.h
 * @brief `DsmcSolver` wired to `DsmcSimpleStatistics`: purely elastic
 *        collisions (no internal-energy exchange), the cheapest complete
 *        DSMC solver configuration.
 *
 * @details
 * The minimal concrete solver: it adds nothing to `DsmcSolver`'s NTC
 * collision selection and `DsmcKernel` scattering (see `dsmc_solver.h`)
 * beyond selecting `DsmcSimpleStatistics` for the per-cell candidate-count
 * measurement — a cheaper `max_sigma_g` estimate that samples only the
 * *relative speed* of a handful of candidate pairs and uses it directly
 * as the running bound (skipping `DsmcKernel::sigma_g`'s actual
 * cross-section evaluation per sample), versus the base `DsmcStatistics`
 * used by `DsmcSolver`/`DsmcEnergyExchangeSolver` by default, which
 * samples the true species-aware `sigma * g` per candidate pair. Because
 * NTC's expected accepted-collision count is self-correcting regardless
 * of how tight the `max_sigma_g` bound is (a looser bound draws more
 * candidates but accepts each at a proportionally lower probability, and
 * `DsmcSolver::collide_indexed_pair` keeps raising the per-cell bound
 * whenever it sees an actual `sigma_g` exceeding it — see
 * `dsmc_solver.h`), skipping the cross-section evaluation here only
 * costs extra rejected-candidate work, not correctness; use this when
 * `DsmcStatistics`'s per-sample kernel evaluations are the bottleneck.
 */

namespace atlas {

/**
 * @brief Elastic-only DSMC solver (`DsmcSimpleStatistics` +
 *        `DsmcSolver`'s NTC/kernel pipeline, no internal-energy
 *        exchange). See this file's top-of-file documentation.
 */
class DsmcSimpleSolver final : public DsmcSolver {
public:
    using Base = DsmcSolver;
    using Base::Base;

    class Builder;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    /** @brief `DsmcSimpleStatistics::measure` — NTC candidate-count
     *  measurement using a relative-speed-only `max_sigma_g` estimate
     *  (see this file's top-of-file documentation). */
    ATLAS_HOST bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

private:
    DsmcSimpleStatistics _simple_statistics {};
};

/**
 * @brief Fluent builder for `DsmcSimpleSolver`. Validation (`validate()`,
 *        run by `build()`/`make_host_shared()`) requires non-null
 *        `_universe`/`_fluid`/`_searcher`.
 */
class DsmcSimpleSolver::Builder final {
public:
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST Builder&
    with_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST void
    validate() const;

    ATLAS_HOST ATLAS_NODISCARD DsmcSimpleSolver
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<DsmcSimpleSolver>
    make_host_shared() const;

private:
    UniverseHostPtr _universe {};
    FluidHostPtr _fluid {};
    SearcherHostPtr _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };
};

using DsmcSimpleSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSimpleSolver>;

using DsmcSimpleSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSimpleSolver>;

}
