#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

enum struct DsmcCollisionWorkloadType : int {
    cell,
    flatten
};

/**
 * @brief Direct Simulation Monte Carlo solver for particle-based rarefied-gas collisions.
 *
 * @tparam T Floating-point scalar type used by the solver, such as `float` or `double`.
 *
 * @details
 * `DsmcSolver` performs cell-local stochastic collisions between particles using
 * the DSMC no-time-counter style workflow:
 *
 * 1. Build a spatial hashing structure for the current particle positions.
 * 2. Measure per-cell collision statistics, including particle count, maximum
 *    relative speed, maximum `sigma * g`, and sampled collision count.
 * 3. Select random particle pairs inside each cell.
 * 4. Accept or reject each candidate pair using the local maximum `sigma * g`.
 * 5. Apply the configured DSMC collision kernel to accepted pairs.
 *
 * The solver stores transient per-cell collision states in the `Universe`.
 * Particle velocity and species data are accessed through the `Fluid`.
 *
 * @note
 * This class is designed to launch device-side parallel loops from host-side
 * solver entry points. The actual pair collision routine is marked as
 * `ATLAS_ALL_DEVICE` so that it can be executed inside device kernels.
 */
template <typename T>
class DsmcSolver : public Solver<T> {
public:
    /**
     * @brief Probe type used to pass raw solver data into device-side DSMC kernels.
     *
     * @details
     * The probe stores raw pointers to fluid states, universe states, searcher
     * buffers, particle properties, scalar simulation constants, and the selected
     * DSMC collision kernel. It is copied into device lambdas before launching
     * parallel loops.
     */
    using Probe = atlas::system::DsmcProbe<T>;

    /**
     * @brief Creates an empty DSMC solver.
     *
     * @details
     * The default-constructed solver does not own valid universe, fluid, or
     * searcher references. It is mainly provided for object construction,
     * serialization-like workflows, or delayed initialization.
     */
    DsmcSolver() = default;

    /**
     * @brief Constructs a DSMC solver with simulation objects and a collision kernel.
     *
     * @param universe Shared host pointer to the simulation universe.
     * @param fluid Shared host pointer to the particle fluid storage.
     * @param searcher Shared host pointer to the spatial hashing searcher.
     * @param kernel_type DSMC collision kernel type used for pair collisions.
     *
     * @details
     * The constructor forwards the universe, fluid, and spatial searcher to the
     * base solver, initializes the DSMC kernel, and ensures that all required
     * per-cell collision states exist in the universe.
     *
     * @post
     * The universe contains all DSMC cell states required by this solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type = DsmcKernelType::hard_sphere,
               DsmcCollisionWorkloadType workload_type = DsmcCollisionWorkloadType::cell) noexcept;

    /**
     * @brief Destroys the DSMC solver.
     */
    ~DsmcSolver() override = default;

    /**
     * @brief Advances DSMC collision processing for all cells.
     *
     * @param dt Simulation time step.
     *
     * @details
     * This overload applies the solver to every spatial cell by forwarding to
     * `solve(nullptr, 0, dt)`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    /**
     * @brief Advances DSMC collision processing for cells assigned to this solver.
     *
     * @param allocated_solver Optional per-cell solver assignment buffer.
     * @param index Solver index used to select cells from `allocated_solver`.
     * @param dt Simulation time step.
     *
     * @details
     * If `allocated_solver` is not null, only cells whose assigned solver index
     * equals `index` are processed. If it is null, all cells are processed.
     *
     * The method ensures DSMC states, rebuilds the spatial hashing structure,
     * creates a device probe, measures per-cell collision statistics, and applies
     * sampled DSMC collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    /**
     * @brief Returns the current DSMC collision kernel type.
     *
     * @return Configured DSMC kernel type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcCollisionWorkloadType
    workload_type() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    /**
     * @brief Ensures that all DSMC per-cell universe states exist and have valid size.
     *
     * @details
     * The following states are created or resized to match the current number of
     * universe cells:
     *
     * - `UniverseNumberParticleState<T>`
     * - `UniverseMaxRelativeSpeedState<T>`
     * - `UniverseMaxSigmaGState<T>`
     * - `UniverseCollisionCountState<int>`
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    /**
     * @brief Resets all DSMC per-cell collision states.
     *
     * @details
     * This function first calls `ensure_states()` and then resets particle counts,
     * maximum relative speeds, maximum `sigma * g` values, and sampled collision
     * counts.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset_states();

    /**
     * @brief Builds the DSMC probe used by device-side collision kernels.
     *
     * @details
     * The probe caches raw pointers to fluid states, universe states, spatial
     * hashing arrays, particle properties, scalar constants, and the collision
     * kernel. It also advances the internal collision seed so each solve step uses
     * a different deterministic random stream.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    make_probe() noexcept;

    /**
     * @brief Measures per-cell DSMC collision statistics.
     *
     * @param allocated_solver Optional per-cell solver assignment buffer.
     * @param index Solver index used to filter cells when `allocated_solver` is set.
     * @param dt Simulation time step.
     *
     * @return Always returns `true` after launching the measurement pass.
     *
     * @details
     * For each selected cell, this method computes:
     *
     * - Number of particles in the cell.
     * - Maximum relative speed among particle pairs.
     * - Maximum collision-rate term `sigma * g`.
     * - Stochastically rounded number of candidate collisions.
     *
     * The candidate collision count follows the DSMC no-time-counter estimate
     * based on pair count, maximum `sigma * g`, statistical weight, time step,
     * and cell volume.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Applies sampled DSMC pair collisions inside each selected cell.
     *
     * @param allocated_solver Optional per-cell solver assignment buffer.
     * @param index Solver index used to filter cells when `allocated_solver` is set.
     * @param dt Simulation time step.
     *
     * @details
     * This method reads the per-cell collision counts produced by
     * `measure_collision_statistics()`, samples random candidate pairs, and
     * delegates pair acceptance and velocity update to `collide_pair()`.
     *
     * @note
     * The current implementation does not use `dt` directly in this stage because
     * the time-step dependence is already included in the measured collision count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Attempts one DSMC collision between two locally indexed particles.
     *
     * @param probe Device-side DSMC probe containing raw simulation data.
     * @param cell Cell index currently being processed.
     * @param local_collision Local collision attempt index inside the cell.
     * @param begin First sorted-particle offset of the cell.
     * @param end One-past-last sorted-particle offset of the cell.
     * @param lhs_local Local index of the first candidate particle in the cell.
     * @param rhs_local Local index of the second candidate particle in the cell.
     * @param max_sigma_g Maximum `sigma * g` value measured for the cell.
     *
     * @return `true` if the candidate pair is accepted and velocities are updated;
     *         otherwise `false`.
     *
     * @details
     * The method maps local cell indices to global particle indices, computes the
     * pair-specific `sigma * g`, accepts the collision with probability
     * `sigma_g / max_sigma_g`, and applies the configured DSMC collision kernel
     * to accepted pairs.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 T max_sigma_g) noexcept;

    /**
     * @brief Attempts one DSMC collision between two global particle indices.
     *
     * @details
     * This is the device hot path used after pair sampling has already resolved
     * local cell offsets to global particle indices.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         T max_sigma_g) noexcept;

    /**
     * @brief Samples two distinct local particle indices for one collision trial.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    sample_distinct_pair(int& lhs_local,
                         int& rhs_local,
                         int cell,
                         int count,
                         std::uint64_t seed,
                         std::uint64_t stream) noexcept;

    /**
     * @brief Returns the global particle index for a local cell particle index.
     *
     * @param nth Local particle index inside the cell.
     * @param begin First sorted-particle offset of the cell.
     * @param end One-past-last sorted-particle offset of the cell.
     * @param particle_count Total number of particles in the fluid.
     * @param indices_ptr Sorted particle-index array from the spatial searcher.
     *
     * @return Valid global particle index on success, or `-1` if the local index
     *         or resolved particle index is invalid.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    particle_at(int nth,
                int begin,
                int end,
                int particle_count,
                const int* indices_ptr) noexcept;

protected:
    /**
     * @brief Device-side data probe used during DSMC measurement and collision passes.
     */
    DsmcProbe<T> _probe {};

    /**
     * @brief Collision kernel object that evaluates `sigma * g` and updates velocities.
     */
    DsmcKernel<T> _kernel {};

    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };

    DsmcFlattenWorkload<T> _flatten_workload {};

    DsmcStatistics<T> _statistics {};

    /**
     * @brief Monotonic seed used to decorrelate DSMC stochastic samples between steps.
     */
    std::uint64_t _collision_seed = 0;
};

}

namespace atlas {

/**
 * @brief Public alias for the system-level DSMC solver.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

/**
 * @brief Host shared-pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

/**
 * @brief Device shared-pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

using DsmcCollisionWorkloadType = atlas::system::DsmcCollisionWorkloadType;

}

#include <atlas/solver/dsmc/dsmc_solver.hpp>
