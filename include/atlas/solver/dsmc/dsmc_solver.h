#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the common DSMC solver base class for cell-wise collision scheduling.
 *
 * This header defines `atlas::system::DsmcSolver<T>`, an abstract Direct
 * Simulation Monte Carlo solver base. The class prepares per-cell collision
 * statistics and delegates the actual collision-application strategy to derived
 * solvers.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

/**
 * @brief Abstract base class for DSMC collision solvers.
 *
 * `DsmcSolver<T>` provides the shared DSMC collision-preparation pipeline used
 * by concrete DSMC solver variants. It stores the universe, fluid, and spatial
 * searcher dependencies directly, maintains a cached probe over commonly used
 * runtime data, and computes per-cell no-time-counter collision statistics.
 *
 * The main solve path is:
 *
 * @code
 * solve(dt)
 *     -> solve(nullptr, 0, dt)
 *
 * solve(allocated_solver, index, dt)
 *     -> validate universe/fluid/searcher dependencies
 *     -> ensure_states()
 *     -> searcher->build()
 *     -> validate dt > 0
 *     -> make_probe()
 *     -> measure_cell_collision_statistics(allocated_solver, index, dt)
 *     -> apply_collision(allocated_solver, index, dt)
 * @endcode
 *
 * When `allocated_solver` is provided, only cells whose allocation entry equals
 * `index` are assigned non-zero collision statistics. Non-matching cells are
 * reset to zero for particle count, maximum relative speed, maximum sigma-g,
 * and collision count.
 *
 * Derived classes implement only @ref apply_collision. All common state setup,
 * per-cell statistics, and stochastic collision-count rounding are handled by
 * this base class.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
class DsmcSolver : public Solver<T> {
public:
    /**
     * @brief Cached raw-pointer view over DSMC runtime data.
     *
     * `DsmcSolverProbe` is populated by @ref make_probe and stored in `_probe`.
     * It groups raw pointers to fluid states, universe collision-statistic states,
     * spatial-searcher cell ranges, scalar metadata, the selected DSMC kernel,
     * and the collision seed used for stochastic rounding.
     *
     * Required fluid states:
     *
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`.
     *
     * Required universe states:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseMaxSigmaGState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * The probe does not own memory. It is valid only as long as the underlying
     * buffers are not reallocated.
     */
    struct DsmcSolverProbe {
        /**
         * @brief Raw pointer to mutable per-particle velocity data.
         *
         * Points to `FluidVelocityState<T>::data()`. Derived collision solvers
         * use this pointer to read and update particle velocities.
         */
        Vector3<T>* velocity_ptr {};

        /**
         * @brief Raw pointer to per-particle species indices.
         *
         * Points to `FluidSpeciesState<T>::data()`. Species ids are used to
         * resolve material properties for collision cross-section evaluation.
         */
        const std::size_t* species_ptr {};

        /**
         * @brief Raw pointer to per-species material properties.
         *
         * Points to `fluid->particle_properties()`. The DSMC kernel reads these
         * properties when evaluating cross sections.
         */
        const MaterialProperties<T>* properties_ptr {};

        /**
         * @brief Raw pointer to per-cell particle-count output.
         *
         * Points to `UniverseNumberParticleState<T>::data()`.
         */
        T* number_particle_ptr {};

        /**
         * @brief Raw pointer to per-cell maximum relative speed output.
         *
         * Points to `UniverseMaxRelativeSpeedState<T>::data()`.
         */
        T* max_relative_speed_ptr {};

        /**
         * @brief Raw pointer to per-cell maximum sigma-g output.
         *
         * Points to `UniverseMaxSigmaGState<T>::data()`. Each value stores the
         * maximum product of collision cross section and relative speed observed
         * among particle pairs in the corresponding cell.
         */
        T* max_sigma_g_ptr {};

        /**
         * @brief Raw pointer to per-cell collision-count output.
         *
         * Points to `UniverseCollisionCountState<int>::data()`.
         */
        int* collision_count_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * For a cell, the range `[cell_start_ptr[cell], cell_end_ptr[cell])`
         * indexes into this array to obtain particle indices belonging to that cell.
         */
        const int* indices_ptr {};

        /**
         * @brief Raw pointer to the first sorted index for each cell.
         */
        const int* cell_start_ptr {};

        /**
         * @brief Raw pointer to one-past-the-last sorted index for each cell.
         */
        const int* cell_end_ptr {};

        /**
         * @brief Number of particles reported by the fluid.
         */
        int particle_count {};

        /**
         * @brief Number of cells reported by the universe.
         */
        int num_of_cells {};

        /**
         * @brief Volume of one universe cell.
         *
         * Must be positive for collision-statistics measurement to proceed.
         */
        T cell_volume {};

        /**
         * @brief Statistical weight of simulation particles.
         *
         * Used in the no-time-counter collision-count estimate.
         */
        T statistical_weight {};

        /**
         * @brief Runtime DSMC collision kernel.
         *
         * Copied from the solver's configured kernel during @ref make_probe.
         */
        DsmcKernel<T> kernel {};

        /**
         * @brief Seed used for stochastic fractional collision-count rounding.
         *
         * The seed is copied from `_collision_seed` and then `_collision_seed` is
         * incremented whenever @ref make_probe succeeds.
         */
        std::uint64_t collision_seed {};
    };

public:
    /**
     * @brief Constructs an empty DSMC solver.
     *
     * All dependencies are initialized to null-equivalent values. Calling
     * @ref solve on an empty solver is safe: the solver resets available states
     * when possible and returns without applying collisions.
     */
    DsmcSolver() = default;

    /**
     * @brief Constructs a DSMC solver with simulation dependencies and a kernel type.
     *
     * The constructor stores the universe, fluid, and searcher dependencies,
     * constructs the runtime DSMC kernel from `kernel_type`, and calls
     * @ref ensure_states to create required universe states when a universe is
     * available.
     *
     * @param universe Universe containing per-cell DSMC statistic states.
     * @param fluid Fluid containing particle velocity, species, and material data.
     * @param searcher Spatial hashing searcher used to group particles by cell.
     * @param kernel_type DSMC collision kernel model used for cross-section evaluation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroys the DSMC solver through the base interface.
     */
    ~DsmcSolver() override = default;

    /**
     * @brief Runs DSMC collision solving without solver-allocation filtering.
     *
     * This overload forwards to:
     *
     * @code
     * solve(nullptr, 0, dt);
     * @endcode
     *
     * All cells are eligible for collision-statistics measurement and collision
     * application.
     *
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument If `dt` is not positive after required
     *                              dependencies have been validated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    /**
     * @brief Runs DSMC collision solving with optional cell filtering.
     *
     * The function validates required dependencies, ensures required universe
     * states exist, rebuilds the spatial searcher, validates `dt`, refreshes the
     * cached probe, measures per-cell collision statistics, and then calls
     * @ref apply_collision.
     *
     * If `allocated_solver` is non-null, only cells satisfying:
     *
     * @code
     * allocated_solver[cell] == index
     * @endcode
     *
     * receive non-zero collision statistics. Non-matching cells are reset.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument If `dt` is not positive after required
     *                              dependencies have been validated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    /**
     * @brief Returns the configured DSMC kernel type.
     *
     * @return Kernel type stored in the runtime DSMC kernel.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    /**
     * @brief Ensures required universe-side DSMC statistic states exist.
     *
     * If a universe is configured, this function creates or resizes the following
     * states to `universe->number_of_cells()`:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseMaxSigmaGState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * Existing states are preserved but resized when their buffer size differs
     * from the current universe cell count. If no universe is configured, the
     * function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    /**
     * @brief Resets universe-side DSMC statistic states to zero.
     *
     * If a universe exists and reports a positive number of cells, this function
     * calls @ref ensure_states and resets the following states:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseMaxSigmaGState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * If no universe exists, or if the universe has no cells, the function returns
     * without work.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset_states();

    /**
     * @brief Refreshes the cached DSMC probe from current runtime data.
     *
     * The function resolves required fluid states, required universe states,
     * searcher buffers, scalar metadata, and the runtime kernel into `_probe`.
     * On success, `_probe.collision_seed` receives the current collision seed and
     * `_collision_seed` is incremented.
     *
     * This function does not validate that searcher pointers are non-null, that
     * particle count is positive, that material properties are non-empty, or that
     * cell volume is positive. Those checks are performed later by
     * @ref measure_cell_collision_statistics where needed.
     *
     * @retval true Required dependencies and required states were found.
     * @retval false A required dependency or required state was missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    /**
     * @brief Computes per-cell DSMC no-time-counter collision statistics.
     *
     * The function uses the cached `_probe` built by @ref make_probe. It first
     * validates the global preconditions:
     *
     * - at least two particles,
     * - positive number of cells,
     * - non-null material-property pointer,
     * - positive cell volume.
     *
     * If these checks fail, all DSMC statistic states are reset and the function
     * returns `false`.
     *
     * For each cell, the device pass:
     *
     * - optionally filters by `allocated_solver[cell] == index`,
     * - reads the searcher range `[cell_start[cell], cell_end[cell])`,
     * - scans all unordered particle pairs in the cell,
     * - computes maximum squared relative speed,
     * - computes maximum `sigma * g` using @ref sigma_g,
     * - writes particle count, maximum relative speed, and maximum sigma-g,
     * - estimates the no-time-counter collision count,
     * - stochastically rounds the fractional part using
     *   `atlas::sampling::sample_hashed_unit_interval`,
     * - writes the final non-negative collision count.
     *
     * The no-time-counter estimate is:
     *
     * @code
     * ntc_pair_count = count * (count - 1) * 0.5;
     * ntc_count = ntc_pair_count * max_sigma_g * statistical_weight * dt / cell_volume;
     * @endcode
     *
     * If `ntc_count` exceeds `std::numeric_limits<int>::max()`, the collision
     * count is saturated to the maximum `int` value.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Positive time-step size used in the no-time-counter estimate.
     *
     * @retval true The statistics pass was launched after successful precondition checks.
     * @retval false Required cached data or global DSMC preconditions were unavailable.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Returns the global particle index at a checked cell-local offset.
     *
     * The function computes:
     *
     * @code
     * sorted_index = begin + nth;
     * @endcode
     *
     * and returns `indices_ptr[sorted_index]` only when both the sorted index and
     * resulting particle index are valid.
     *
     * Despite the name, this helper does not scan forward to skip invalid entries.
     * It checks only the direct sorted entry at `begin + nth`.
     *
     * @param nth Zero-based local offset from `begin`.
     * @param begin First sorted-index position belonging to the cell.
     * @param end One-past-the-last sorted-index position belonging to the cell.
     * @param particle_count Global particle-count bound.
     * @param indices_ptr Searcher sorted particle-index array.
     *
     * @return Global particle index, or `-1` if the offset or particle index is invalid.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    nth_valid_particle(int nth,
                       int begin,
                       int end,
                       int particle_count,
                       const int* indices_ptr) noexcept;

    /**
     * @brief Computes the DSMC sigma-g value for a particle pair.
     *
     * The helper receives squared relative speed, computes the relative speed,
     * evaluates the kernel cross section for the two species material properties,
     * and returns:
     *
     * @code
     * sigma_g = cross_section * relative_speed;
     * @endcode
     *
     * If `relative_speed_squared` is not positive, the function returns zero.
     *
     * @param kernel Runtime DSMC kernel wrapper.
     * @param properties_ptr Raw pointer to per-species material properties.
     * @param species_i Species index of the first particle.
     * @param species_j Species index of the second particle.
     * @param relative_speed_squared Squared magnitude of pair relative velocity.
     *
     * @return Product of collision cross section and relative speed, or zero for
     *         non-positive squared relative speed.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sigma_g(const DsmcKernel<T>& kernel,
            const MaterialProperties<T>* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            T relative_speed_squared) noexcept;

    /**
     * @brief Applies collisions using the concrete derived-solver strategy.
     *
     * This pure virtual stage is called only after per-cell collision statistics
     * have been measured successfully. Derived classes use `_probe` and the
     * universe collision-count states to select particle pairs, accept/reject
     * collisions, and update particle velocities.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Time-step size associated with this collision solve.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt)
        = 0;

protected:
    /**
     * @brief Cached probe populated by @ref make_probe.
     *
     * Derived collision implementations may read this probe during
     * @ref apply_collision. It should be refreshed once per solve step before
     * collision statistics and collision application.
     */
    DsmcSolverProbe _probe {};

    /**
     * @brief Runtime DSMC collision kernel.
     */
    DsmcKernel<T> _kernel {};

    /**
     * @brief Monotonic seed source for stochastic collision-count rounding.
     *
     * The current seed is copied into `_probe.collision_seed` whenever
     * @ref make_probe succeeds, then incremented for the next solve pass.
     */
    std::uint64_t _collision_seed = 0;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_solver.hpp>
