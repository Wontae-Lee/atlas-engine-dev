#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the common DSMC solver base class and collision-workload utilities.
 *
 * This header defines `atlas::system::DsmcSolver<T>`, an abstract base class
 * that implements the shared Direct Simulation Monte Carlo collision scheduling
 * pipeline. Derived classes provide only the concrete collision-application
 * strategy.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Abstract base class for DSMC collision solvers.
 *
 * `DsmcSolver<T>` extends @ref Solver with DSMC-specific collision scheduling.
 * It owns the selected collision kernel, creates the universe states required
 * for collision statistics, computes per-cell collision counts, and exposes a
 * flattened collision workload for derived solvers.
 *
 * The public solve path is:
 *
 * @code
 * solve(dt)
 *     -> solve(nullptr, 0, dt)
 *
 * solve(allocated_solver, index, dt)
 *     -> build_collision_workload(allocated_solver, index, dt)
 *     -> apply_collisions(allocated_solver, index, dt)
 * @endcode
 *
 * `build_collision_workload` validates the runtime context, checks that `dt` is
 * positive, and measures per-cell collision statistics. The final collision
 * traversal is delegated to @ref apply_collisions.
 *
 * When `allocated_solver` is provided, per-cell statistics are generated only
 * for cells whose assignment equals `index`; other cells are explicitly reset to
 * zero collision statistics.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
class DsmcSolver : public Solver<T> {
public:
    /**
     * @brief Raw-pointer view over common DSMC runtime data.
     *
     * `DsmcSolverProbe` is populated by @ref make_probe and is intended to be
     * captured by value in device kernels. It contains pointers to fluid particle
     * state, universe collision-statistic state, searcher cell ranges, optional
     * codec solver-allocation data, and scalar simulation metadata.
     *
     * Required states for a valid probe are:
     *
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`,
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * The allocated-solver pointer is optional and remains `nullptr` when the
     * solver is running in the non-codec path.
     */
    struct DsmcSolverProbe {
        /**
         * @brief Raw pointer to mutable per-particle velocity data.
         *
         * Points to `FluidVelocityState<T>::data()`.
         */
        Vector3<T>* velocity_ptr {};

        /**
         * @brief Raw pointer to per-particle species indices.
         *
         * Points to `FluidSpeciesState<T>::data()`.
         */
        const std::size_t* species_ptr {};

        /**
         * @brief Raw pointer to per-species material properties.
         *
         * Points to `fluid->particle_properties()`. Cross-section evaluation uses
         * the material properties associated with each particle pair.
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
         * @brief Raw pointer to per-cell collision-count output.
         *
         * Points to `UniverseCollisionCountState<int>::data()`.
         */
        int* collision_count_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * For a cell, `[cell_start_ptr[cell], cell_end_ptr[cell])` indexes into
         * this array to obtain particle indices belonging to that cell.
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
         * @brief Optional raw pointer to per-cell solver allocation data.
         *
         * When non-null, cells whose allocation entry does not match the current
         * solver index are skipped and reset during collision-statistics
         * measurement.
         */
        const int* allocated_solver_ptr {};

        /**
         * @brief Number of particles reported by the fluid.
         */
        int particle_count {};

        /**
         * @brief Number of cells reported by the universe.
         */
        int num_of_cells {};

        /**
         * @brief Number of species material-property entries.
         */
        int num_of_properties {};

        /**
         * @brief Volume of one universe cell.
         *
         * Must be positive for collision-statistics measurement to proceed.
         */
        T cell_volume {};

        /**
         * @brief Statistical weight of the fluid particles.
         *
         * Used in the no-time-counter collision-count estimate.
         */
        T statistical_weight {};

        /**
         * @brief Selected DSMC kernel type.
         */
        DsmcKernelType kernel_type { DsmcKernelType::hard_sphere };

        /**
         * @brief Runtime kernel wrapper copied into device work.
         */
        DsmcKernel<T> kernel {};
    };

public:
    /**
     * @brief Constructs an empty DSMC solver.
     *
     * Dependencies are initialized by the @ref Solver base class default state.
     * Calling @ref solve on an unconfigured solver is safe: the collision context
     * fails to initialize and no collisions are applied.
     */
    DsmcSolver() = default;

    /**
     * @brief Constructs a DSMC solver with simulation dependencies and a kernel type.
     *
     * The constructor forwards the universe, fluid, and searcher to the
     * @ref Solver base class, constructs the runtime DSMC kernel wrapper from
     * `kernel_type`, stores the selected kernel tag, and calls
     * @ref ensure_universe_states.
     *
     * @param universe Universe containing per-cell collision-statistic states.
     * @param fluid Fluid containing particle velocity, species, and material data.
     * @param searcher Spatial hashing searcher used to group particles by cell.
     * @param kernel_type Collision kernel model used for cross-section evaluation.
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
     * @brief Runs DSMC collision solving without codec-based cell filtering.
     *
     * This overload forwards to:
     *
     * @code
     * solve(nullptr, 0, dt);
     * @endcode
     *
     * All cells with valid collision data are eligible for statistics measurement
     * and collision application.
     *
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument If `dt` is not positive after the collision
     *                              context has initialized successfully.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) final;

    /**
     * @brief Runs DSMC collision solving with optional codec-based cell filtering.
     *
     * The function first calls @ref build_collision_workload. If workload
     * construction returns `false`, no collision application is attempted. If it
     * succeeds, @ref apply_collisions is invoked with the same arguments.
     *
     * When `allocated_solver` is non-null, only cells with:
     *
     * @code
     * allocated_solver[cell] == index
     * @endcode
     *
     * contribute non-zero collision statistics.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when filtering through `allocated_solver`.
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument If `dt` is not positive after the collision
     *                              context has initialized successfully.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) final;

    /**
     * @brief Returns the selected DSMC collision kernel type.
     *
     * @return Kernel type stored by this solver.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    /**
     * @brief Returns the prefix offsets for the flattened collision workload.
     *
     * After @ref build_flattened_collision_workload, entry `cell` stores the
     * starting offset of that cell's collision entries in
     * @ref flattened_collision_cells.
     *
     * @return Const reference to the collision-offset buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    /**
     * @brief Returns the flattened list of cells that own collision entries.
     *
     * The flattened buffer contains each cell repeated according to its
     * per-cell collision count. For example, if cell 7 has three scheduled
     * collisions, cell 7 appears three times in this buffer.
     *
     * @return Const reference to the flattened collision-cell buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    flattened_collision_cells() const noexcept;

    /**
     * @brief Ensures DSMC collision-statistic universe states and offsets exist.
     *
     * If a universe is configured, this function creates the following missing
     * states with `universe->number_of_cells()` elements:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * Existing states are left unchanged. The solver-owned
     * `_collision_offsets` buffer is resized to the universe cell count.
     *
     * If no universe is configured, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Clears collision statistics and temporary collision-workload buffers.
     *
     * If no universe exists, or if the universe has no cells, this function clears
     * `_collision_offsets` and `_flattened_collision_cells`.
     *
     * Otherwise, it ensures required universe states exist, fills the following
     * per-cell buffers with zero on the device:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`,
     * - `_collision_offsets`,
     *
     * and clears `_flattened_collision_cells`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data();

    /**
     * @brief Builds the per-cell DSMC collision workload for the current step.
     *
     * This function performs:
     *
     * @code
     * if (!initialize_collision_context()) return false;
     * if (!(dt > 0)) throw std::invalid_argument(...);
     * return measure_cell_collision_statistics(allocated_solver, index, dt);
     * @endcode
     *
     * It does not call @ref build_flattened_collision_workload in the current
     * implementation. Derived solvers that require the flattened workload should
     * call @ref build_flattened_collision_workload explicitly before consuming
     * @ref flattened_collision_cells.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Positive time-step size.
     *
     * @retval true Collision statistics were measured successfully.
     * @retval false Required context or collision input data was unavailable.
     *
     * @throw std::invalid_argument If `dt` is not positive after context
     *                              initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_collision_workload(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Initializes the DSMC collision context.
     *
     * The function validates that universe, fluid, and searcher dependencies
     * exist. If any are missing, collision data is reset and initialization fails.
     *
     * On success path, it:
     *
     * - ensures required universe states exist,
     * - rebuilds the spatial searcher via `searcher->build()`,
     * - verifies required fluid and universe states.
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
     * - `UniverseCollisionCountState<int>`.
     *
     * If any required state is missing after setup, collision data is reset and
     * initialization fails.
     *
     * @retval true Collision context is ready.
     * @retval false Required dependencies or states are missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_collision_context() noexcept;

    /**
     * @brief Populates a raw-pointer probe for DSMC collision kernels.
     *
     * The probe resolves fluid velocity/species data, material properties,
     * universe collision-statistic buffers, searcher ranges, optional allocated
     * solver data, and scalar metadata.
     *
     * Unlike @ref measure_cell_collision_statistics, this function does not check
     * particle count, property count, cell count, cell volume, or whether searcher
     * pointers are non-null. It returns `true` once the required dependencies and
     * required states exist. Consumers must validate any additional constraints
     * they need before dereferencing pointers in device code.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer. When
     *                         null, `probe.allocated_solver_ptr` is null.
     * @param probe Output probe populated with raw pointers and metadata.
     *
     * @retval true Required dependencies and required states were found.
     * @retval false A required dependency or state was missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe(const DeviceBuffer<int>* allocated_solver, DsmcSolverProbe& probe) noexcept;

    /**
     * @brief Computes per-cell DSMC collision statistics.
     *
     * This function builds a probe and validates the basic DSMC preconditions:
     *
     * - at least two particles,
     * - positive number of cells,
     * - at least one species material-property entry,
     * - positive cell volume.
     *
     * If these checks fail, collision data is reset and the function returns
     * `false`.
     *
     * For every cell, the device pass:
     *
     * - optionally filters the cell by `allocated_solver[cell] == index`,
     * - reads the searcher range `[cell_start[cell], cell_end[cell])`,
     * - computes the number of particles in the cell,
     * - scans all unordered particle pairs,
     * - computes maximum relative speed,
     * - computes the maximum `sigma * g` using
     *   `DsmcKernel<T>::cross_section(...)`,
     * - estimates an NTC collision count,
     * - clamps that count to `[0, max_pairs]`,
     * - writes per-cell particle count, maximum relative speed, and collision count.
     *
     * The NTC estimate is:
     *
     * @code
     * pair_count = count * (count - 1) * 0.5;
     * ntc_count = pair_count * max_sigma_g * statistical_weight * dt / cell_volume;
     * collisions = floor(ntc_count);
     * @endcode
     *
     * Cells with no matching solver allocation, invalid/empty ranges, fewer than
     * two particles, or non-positive `max_sigma_g` receive zero collision count.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Positive time-step size used in the NTC estimate.
     *
     * @retval true The device pass was launched after successful precondition checks.
     * @retval false Required probe data or DSMC preconditions were unavailable.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Builds a flattened collision-cell workload from per-cell collision counts.
     *
     * The function reads `UniverseCollisionCountState<int>`, computes exclusive
     * prefix offsets into `_collision_offsets`, and fills
     * `_flattened_collision_cells` so each cell appears once per scheduled
     * collision.
     *
     * For each cell:
     *
     * @code
     * for local_collision in [0, collision_count[cell]):
     *     flattened_collision_cells[offset[cell] + local_collision] = cell;
     * @endcode
     *
     * If the collision-count state is missing, collision data is reset and the
     * function returns `false`. If the total number of collisions is zero,
     * `_flattened_collision_cells` is cleared and the function returns `false`.
     *
     * @retval true A non-empty flattened workload was created.
     * @retval false Collision-count state was missing or total collisions were zero.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload() noexcept;

    /**
     * @brief Returns the global particle index for the N-th sorted entry in a cell range.
     *
     * This helper computes:
     *
     * @code
     * sorted_index = begin + nth;
     * @endcode
     *
     * and returns `indices_ptr[sorted_index]` only if both the sorted index and
     * resulting particle index are within bounds.
     *
     * Despite the name, this function does not scan over invalid entries to find
     * the N-th valid particle. It checks only the direct sorted entry at
     * `begin + nth`.
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
     * @brief Converts an unordered-pair ordinal into local pair indices.
     *
     * The helper maps a zero-based ordinal in the conceptual upper-triangular
     * pair list to:
     *
     * - `lhs_local`, written by reference,
     * - right-hand local index, returned by value.
     *
     * Pairs are enumerated row by row:
     *
     * @code
     * (0, 1), (0, 2), ..., (0, count - 1),
     * (1, 2), (1, 3), ..., (1, count - 1),
     * ...
     * @endcode
     *
     * If the ordinal is out of range, the function returns `-1`.
     *
     * @param count Number of local particles in the cell.
     * @param ordinal Zero-based unordered-pair ordinal.
     * @param lhs_local Output left-hand local index.
     *
     * @return Right-hand local index, or `-1` if the ordinal is out of range.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    pair_ordinal_to_rhs(int count, int ordinal, int& lhs_local) noexcept;

    /**
     * @brief Applies scheduled collisions using the derived solver strategy.
     *
     * This pure virtual function is called by @ref solve only after
     * @ref build_collision_workload succeeds. Derived classes implement the
     * actual pair selection, collision acceptance, and velocity update strategy.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Time-step size associated with this collision solve.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt)
        = 0;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DsmcSolverProbe& probe, int index, T dt);

protected:
    /**
     * @brief Runtime DSMC kernel wrapper used by derived collision code.
     */
    DsmcKernel<T> _kernel {};

    /**
     * @brief Selected DSMC kernel model.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    /**
     * @brief Per-cell exclusive offsets into the flattened collision workload.
     *
     * Resized to the universe cell count by @ref ensure_universe_states and
     * populated by @ref build_flattened_collision_workload.
     */
    DeviceBuffer<int> _collision_offsets {};

    /**
     * @brief Flattened list of collision-owning cells.
     *
     * Populated by @ref build_flattened_collision_workload. Each cell appears
     * once for each scheduled collision in that cell.
     */
    DeviceBuffer<int> _flattened_collision_cells {};
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