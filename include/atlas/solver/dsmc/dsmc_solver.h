#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>
#include <atlas/solver/solver.h>

#include <cstdint>

/**
 * @file dsmc_solver.h
 * @brief The DSMC collision solver: per grid cell, per timestep, selects
 *        which particle pairs actually collide (Bird's no-time-counter
 *        scheme) and applies the configured `DsmcKernel` to each.
 *
 * @details
 * ### Background — the no-time-counter (NTC) collision selection scheme
 * A physical gas has far too many molecules to collide every pair; DSMC
 * instead simulates a much smaller number of representative particles
 * and must decide, per cell per timestep, *how many* pairs to actually
 * collide so the simulated collision rate matches the real gas's. The
 * expected number of collisions in a cell of `N` particles, volume `V`,
 * over timestep `dt` is (kinetic theory, mean free path derivation):
 * ```
 * N_coll = 0.5 * N * (N-1) * F_N * <sigma * g> * dt / V
 * ```
 * where `F_N` is the statistical weight (real molecules per simulated
 * particle) and `<sigma * g>` is the mean of cross-section times
 * relative speed over all pairs — expensive to compute exactly every
 * step. Bird's NTC scheme (this file) avoids that: candidate pairs are
 * drawn at random from the cell, and each is accepted (collided) with
 * probability `(sigma * g) / (sigma * g)_max`, where `(sigma * g)_max`
 * is a per-cell running upper bound (`DsmcProbe::max_sigma_g_ptr`,
 * updated whenever a sampled pair exceeds it — see
 * `collide_indexed_pair`). Because rejected pairs are cheap (one hashed
 * sample, no kernel evaluation), the number of *candidate* pairs drawn
 * — `N_candidates = 0.5 * N * (N-1) * F_N * (sigma*g)_max * dt / V` — can
 * safely overestimate the true collision count using a conservative
 * bound, and the acceptance test statistically thins it back down to the
 * correct rate without ever needing the true mean `<sigma * g>`.
 * `DsmcStatistics::measure` (see that file) is what actually computes
 * `collision_remainder`/`collision_count` — the (possibly fractional,
 * carried over between steps via `collision_remainder_ptr`) candidate
 * count per cell — from this formula; this file consumes that count and
 * performs the accept/reject test and the collision itself.
 *
 * ### Operating principle
 * `solve(dt)` runs the whole per-step pipeline: ensure fluid/searcher
 * state exists, rebuild `_probe` (a `DsmcProbe` view into the fluid's
 * velocity/species/internal-energy buffers and the searcher's spatial
 * cell partition — see `DsmcSolver::make_probe`), call
 * `measure_collision_statistics` to update each cell's candidate count
 * (`DsmcStatistics`/`DsmcSimpleStatistics`), then `apply_collision` to
 * actually run NTC selection and collide accepted pairs — either
 * per-cell (`DsmcCollisionWorkloadType::cell`, one device thread walks
 * all candidates of one cell) or on a pre-flattened global candidate
 * list (`::flatten`, `DsmcFlattenWorkload`, better load-balanced when
 * cell occupancy is highly non-uniform).
 *
 * `collide_pair`/`collide_indexed_pair` implement one NTC trial:
 * 1. `sample_distinct_pair`/`particle_at` pick two *distinct* particle
 *    slots from the cell's `[begin, end)` range in the searcher's sorted
 *    index array (`indices_ptr`) via hashed sampling — `rhs_local` is
 *    resampled over `count - 1` and shifted past `lhs_local` if it would
 *    collide with it, guaranteeing `lhs_local != rhs_local` in one draw
 *    rather than a reject-and-retry loop.
 * 2. `sigma_g = DsmcKernel::sigma_g(...)` — the pair's actual
 *    `cross_section(g) * g`.
 * 3. If `sigma_g` exceeds the cell's running `max_sigma_g_ptr`, the
 *    bound is raised in place (this is what keeps NTC's estimate
 *    self-correcting across the run without needing a global maximum
 *    computed up front).
 * 4. Accept with probability `sigma_g / max_sigma_g` (clamped to `1`)
 *    against one more hashed uniform; on acceptance, apply
 *    `DsmcKernel::operator()` (the collision model's scattering, see
 *    `dsmc_kernel.h`/`hard_sphere_kernel.h`/etc.) to the pair's
 *    velocities in place.
 *
 * Every random draw in this file uses `atlas::sample_hashed_unit_interval`/
 * `sample_hashed_index` keyed by `(cell, collision_seed + stream + salt)`
 * rather than a stateful RNG stream — `stream` uniquely identifies a
 * (cell, local collision index) pair via
 * `cell * DSMC_CELL_STREAM_MULTIPLIER + local_collision`, so every
 * candidate draw across every cell and every timestep (`collision_seed`
 * changes per step) is independent and reproducible without any
 * per-thread RNG state to allocate or synchronize — the same
 * stateless-hash-as-RNG pattern used throughout Atlas's device code (see
 * `maxwellian_surface_interaction.h`).
 *
 * ### References
 * - G. A. Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
 *   Flows," Oxford University Press, 1994, ch. 5 (the no-time-counter
 *   collision selection scheme this file implements).
 */

namespace atlas {

/**
 * @brief Selects how `DsmcSolver::apply_collision` iterates over NTC
 *        candidate pairs.
 */
enum struct DsmcCollisionWorkloadType : int {
    /** One device thread per cell, walking all of that cell's
     *  candidates serially — simple, but imbalanced if cell occupancy
     *  varies widely. */
    cell,
    /** Candidates across all cells pre-flattened into one global list
     *  (`DsmcFlattenWorkload`) before launch, for even load balancing. */
    flatten
};

/**
 * @brief DSMC collision solver: NTC candidate selection plus the
 *        configured `DsmcKernel`'s collision outcome. See this file's
 *        top-of-file documentation for the full NTC derivation and
 *        per-step pipeline.
 */
class DsmcSolver : public Solver {
public:
    using Probe = atlas::DsmcProbe;

    DsmcSolver() = default;

    ATLAS_HOST
    DsmcSolver(UniverseHostPtr universe,
               FluidHostPtr fluid,
               SearcherHostPtr searcher,
               DsmcKernelType kernel_type              = DsmcKernelType::hard_sphere,
               DsmcCollisionWorkloadType workload_type = DsmcCollisionWorkloadType::cell) noexcept;

    ~DsmcSolver() override = default;

    /** @brief Full per-step pipeline (ensure state, rebuild probe,
     *  measure statistics, apply collisions) for the solver's own
     *  fluid/universe/searcher. See this file's top-of-file docs. */
    ATLAS_HOST void
    solve(float dt) override;

    /** @brief Same pipeline as `solve(dt)`, but restricted to the
     *  cell/index subset selected by `allocated_solver` — used when
     *  multiple solvers share one universe partitioned by region (see
     *  `Solver::solve` in `solver.h`). */
    ATLAS_HOST void
    solve(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    ATLAS_NODISCARD ATLAS_HOST DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST DsmcCollisionWorkloadType
    workload_type() const noexcept;

    ATLAS_HOST void
    set_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    /** @brief Allocates/resizes the per-cell NTC bookkeeping buffers
     *  (`max_sigma_g`, `collision_remainder`, `collision_count`) to
     *  match the universe's current cell count, if not already sized. */
    ATLAS_HOST void
    ensure_states();

    /** @brief Zeros the per-cell NTC bookkeeping buffers (e.g. after a
     *  particle repartition invalidates the running `max_sigma_g`
     *  bounds and any carried-over `collision_remainder`). */
    ATLAS_HOST virtual void
    reset_states();

    /** @brief Rebuilds `_probe` from the current fluid/universe/searcher
     *  state (`DsmcSolver::make_probe`); must be called whenever
     *  those buffers may have been reallocated. */
    ATLAS_HOST void
    make_probe() noexcept;

    /**
     * @brief Updates each cell's NTC candidate count for this step
     *        (`DsmcStatistics`/`DsmcSimpleStatistics::measure`, see this
     *        file's top-of-file documentation for the underlying
     *        `N_candidates` formula).
     * @return `false` if statistics could not be measured (e.g. the
     *         probe is not currently valid); `true` otherwise.
     */
    ATLAS_HOST virtual bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, float dt);

    /** @brief Runs NTC candidate selection and collides accepted pairs
     *  for every cell, dispatching to the per-cell or flattened
     *  workload per `workload_type()`. */
    ATLAS_HOST virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, float dt);

    /**
     * @brief One NTC trial for the `local_collision`-th candidate of
     *        `cell`: resolves `lhs_local`/`rhs_local` (positions within
     *        `[begin, end)` of the cell's sorted particle-index range)
     *        to actual particle indices and forwards to
     *        `collide_indexed_pair`.
     * @return Whether a collision was accepted and applied.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 float max_sigma_g) noexcept;

    /**
     * @brief Core NTC accept/reject test and collision application for
     *        one already-resolved particle pair: computes `sigma_g`,
     *        raises `max_sigma_g_ptr[cell]` if exceeded, accepts with
     *        probability `sigma_g / max_sigma_g`, and on acceptance
     *        applies `probe.kernel`'s scattering in place. See this
     *        file's top-of-file documentation for the full NTC
     *        derivation.
     * @return `false` on any degenerate input (invalid/equal particle
     *         indices, unknown species, zero cross-section) or on
     *         rejection; `true` iff the pair actually collided.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         float max_sigma_g) noexcept;

    /**
     * @brief Draws two distinct local slot indices in `[0, count)` via
     *        hashed sampling in one shot: `rhs_local` is drawn over
     *        `[0, count-1)` and shifted past `lhs_local` if it would
     *        land on or after it, guaranteeing `lhs_local != rhs_local`
     *        without a reject-and-resample loop.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    sample_distinct_pair(int& lhs_local,
                         int& rhs_local,
                         int cell,
                         int count,
                         std::uint64_t seed,
                         std::uint64_t stream) noexcept;

    /**
     * @brief Resolves a local slot index `nth` within `[begin, end)` of
     *        the searcher's sorted particle-index array to an actual
     *        particle index, or `-1` on any out-of-range/invalid input
     *        (used as the sentinel `collide_indexed_pair` checks for).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static int
    particle_at(int nth,
                int begin,
                int end,
                int particle_count,
                const int* indices_ptr) noexcept;

protected:
    ATLAS_HOST virtual void
    apply_flattened_collision(const DeviceBuffer<int>* allocated_solver, int index);

    DsmcProbe _probe {};

    DsmcKernel _kernel {};

    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };

    DsmcFlattenWorkload _flatten_workload {};

    DsmcStatistics _statistics {};

    std::uint64_t _collision_seed = 0;

public:
    ATLAS_HOST void
    apply_cell_collisions(const int* allocated_solver_ptr, int index);

    ATLAS_HOST void
    launch_flattened_collisions();
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcSolver::collide_pair(const Probe& probe,
                         const int cell,
                         const int local_collision,
                         const int begin,
                         const int end,
                         const int lhs_local,
                         const int rhs_local,
                         const float max_sigma_g) noexcept {
    const int particle_i = particle_at(
        lhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    const int particle_j = particle_at(
        rhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return collide_indexed_pair(
        probe,
        cell,
        stream,
        particle_i,
        particle_j,
        max_sigma_g);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcSolver::collide_indexed_pair(const Probe& probe,
                                 const int cell,
                                 const std::uint64_t stream,
                                 const int particle_i,
                                 const int particle_j,
                                 const float max_sigma_g) noexcept {

    if (particle_i < 0 || particle_j < 0 || particle_i == particle_j) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];
    if (species_i >= static_cast<std::size_t>(probe.species_count)
        || species_j >= static_cast<std::size_t>(probe.species_count)) {
        return false;
    }

    Float3 lhs_velocity = probe.velocity_ptr[particle_i];
    Float3 rhs_velocity = probe.velocity_ptr[particle_j];

    const float relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const float sigma_g                = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);

    if (!(sigma_g > 0.0f)) {
        return false;
    }

    float local_max_sigma_g = max_sigma_g;
    if (sigma_g > local_max_sigma_g) {
        local_max_sigma_g           = sigma_g;
        probe.max_sigma_g_ptr[cell] = sigma_g;
    }

    float accept_probability = sigma_g / local_max_sigma_g;
    if (accept_probability > 1.0f) {
        accept_probability = 1.0f;
    }

    const float accept_sample = atlas::sample_hashed_unit_interval(
        cell,
        probe.collision_seed + stream + atlas::DSMC_COLLISION_ACCEPT_SALT);

    if (accept_sample >= accept_probability) {
        return false;
    }

    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;

    return true;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcSolver::sample_distinct_pair(int& lhs_local,
                                 int& rhs_local,
                                 const int cell,
                                 const int count,
                                 const std::uint64_t seed,
                                 const std::uint64_t stream) noexcept {
    lhs_local = atlas::sample_hashed_index(
        cell,
        count,
        seed + stream + atlas::DSMC_COLLISION_LHS_SALT);

    rhs_local = atlas::sample_hashed_index(
        cell,
        count - 1,
        seed + stream + atlas::DSMC_COLLISION_RHS_SALT);

    if (rhs_local >= lhs_local) {
        ++rhs_local;
    }
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
DsmcSolver::particle_at(const int nth,
                        const int begin,
                        const int end,
                        const int particle_count,
                        const int* indices_ptr) noexcept {

    const int sorted_index = begin + nth;

    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }

    const int particle_index = indices_ptr[sorted_index];

    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSolver>;

using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSolver>;

}
