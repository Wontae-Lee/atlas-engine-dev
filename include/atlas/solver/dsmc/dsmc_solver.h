#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the common DSMC solver base class and collision-workload utilities.
 *
 * @details
 * This header defines @ref atlas::system::DsmcSolver, the abstract base class
 * for Direct Simulation Monte Carlo (DSMC) collision solvers.
 *
 * DSMC collision solvers update particle velocities through statistically
 * selected binary particle collisions. Unlike force-based particle solvers, DSMC
 * does not continuously integrate pairwise forces during the collision step.
 * Instead, particles are grouped into spatial cells, collision opportunities are
 * estimated per cell, and concrete derived solvers decide how particle pairs are
 * selected and processed.
 *
 * This base class provides the shared DSMC infrastructure:
 *
 * - construction and storage of the selected DSMC collision kernel,
 * - creation of required universe-side collision-statistic states,
 * - runtime probe construction for device kernels,
 * - per-cell particle and collision statistics,
 * - optional hybrid-solver cell filtering,
 * - optional flattened collision-workload construction.
 *
 * Derived classes are responsible for implementing only the final collision
 * application stage through @ref atlas::system::DsmcSolver::apply_collisions.
 *
 * The common solve path is:
 *
 * @code
 * solve(dt)
 *     -> solve(nullptr, 0, dt)
 *
 * solve(allocated_solver, index, dt)
 *     -> initialize_collision_context()
 *     -> make_probe(allocated_solver, probe)
 *     -> build_collision_workload(probe, index, dt)
 *     -> apply_collisions(probe, index, dt)
 * @endcode
 *
 * When an allocated-solver buffer is provided, only cells whose solver allocation
 * equals the requested solver index are included in collision-statistics
 * measurement and later collision processing.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Abstract base class for DSMC collision solvers.
 *
 * @details
 * `DsmcSolver<T>` extends `Solver<T>` with the shared infrastructure required by
 * DSMC collision models.
 *
 * The class does not define one fixed particle-pair selection policy. Instead,
 * it prepares the cell-local collision data that derived classes can consume.
 * Different derived classes may then implement different collision strategies,
 * for example:
 *
 * - selecting disjoint local pairs,
 * - cycling through all unordered local pairs,
 * - using a flattened collision workload,
 * - applying a stochastic pair-selection rule.
 *
 * @section dsmc_solver_base_responsibility Base-class responsibility
 *
 * The base class is responsible for:
 *
 * - validating required universe, fluid, and searcher dependencies,
 * - ensuring required universe-side states exist,
 * - rebuilding the spatial hashing searcher before collision work,
 * - preparing a raw-pointer probe for device kernels,
 * - measuring per-cell DSMC collision statistics,
 * - storing per-cell collision counts,
 * - exposing optional flattened collision workload buffers.
 *
 * @section dsmc_solver_derived_responsibility Derived-class responsibility
 *
 * Derived classes implement @ref apply_collisions. That function receives a
 * fully prepared @ref DsmcSolverProbe and decides how scheduled collision work is
 * converted into actual particle-pair velocity updates.
 *
 * The collision physics itself is delegated to the configured
 * `DsmcKernel<T>`. The derived solver usually selects two particles, reads their
 * species properties, copies their velocities into local variables, invokes the
 * kernel, and writes the updated velocities back.
 *
 * @section dsmc_solver_collision_statistics Collision-statistics estimate
 *
 * For each selected cell, the base class estimates how many collision attempts
 * should be performed. The estimate uses the number of unordered particle pairs,
 * the maximum value of `sigma * g`, the fluid statistical weight, the time step,
 * and the cell volume.
 *
 * Conceptually:
 *
 * @code
 * pair_count = count * (count - 1) * 0.5;
 * ntc_count  = pair_count * max_sigma_g * statistical_weight * dt / cell_volume;
 * collisions = floor(ntc_count);
 * @endcode
 *
 * The resulting collision count is stored per cell in
 * `UniverseCollisionCountState<int>`.
 *
 * @tparam T Floating-point scalar type used by the DSMC solver.
 *
 * @note
 * This base class schedules and describes collision work. The actual particle
 * pair traversal is implemented by derived solvers.
 *
 * @see DsmcKernel
 * @see Solver
 * @see SpatialHashingSearcher
 */
template <typename T>
class DsmcSolver : public Solver<T> {
public:
    /**
     * @brief Raw-pointer view over common DSMC runtime data.
     *
     * @details
     * `DsmcSolverProbe` is populated by @ref make_probe and is designed to be
     * captured by value inside device kernels. It stores raw pointers to the
     * particle states, universe collision-statistic states, searcher-generated
     * cell ranges, optional hybrid-solver allocation data, and scalar metadata
     * needed by DSMC kernels.
     *
     * The probe itself does not own memory. It is a lightweight view over buffers
     * owned by the universe, fluid, searcher, and solver.
     *
     * Required fluid states for a valid probe:
     *
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`.
     *
     * Required universe states for a valid probe:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * The allocated-solver pointer is optional. It is `nullptr` when the solver is
     * running without codec-based or hybrid-solver cell filtering.
     */
    struct DsmcSolverProbe {
        /**
         * @brief Raw pointer to mutable per-particle velocity data.
         *
         * @details
         * Points to `FluidVelocityState<T>::data()`. Derived collision solvers
         * read from and write to this buffer when applying binary collision
         * updates.
         */
        Vector3<T>* velocity_ptr {};

        /**
         * @brief Raw pointer to per-particle species indices.
         *
         * @details
         * Points to `FluidSpeciesState<T>::data()`. Each entry maps a particle to
         * an entry in the material-property array.
         */
        const std::size_t* species_ptr {};

        /**
         * @brief Raw pointer to per-species material properties.
         *
         * @details
         * Points to `fluid->particle_properties()`. Collision kernels use these
         * properties to evaluate cross sections and to update velocities for a
         * particle pair.
         */
        const MaterialProperties<T>* properties_ptr {};

        /**
         * @brief Raw pointer to per-cell particle-count output.
         *
         * @details
         * Points to `UniverseNumberParticleState<T>::data()`. The base solver
         * writes the number of valid particles measured in each cell.
         */
        T* number_particle_ptr {};

        /**
         * @brief Raw pointer to per-cell maximum relative speed output.
         *
         * @details
         * Points to `UniverseMaxRelativeSpeedState<T>::data()`. The base solver
         * stores the maximum relative speed found among valid unordered particle
         * pairs in each cell.
         */
        T* max_relative_speed_ptr {};

        /**
         * @brief Raw pointer to per-cell collision-count output.
         *
         * @details
         * Points to `UniverseCollisionCountState<int>::data()`. The base solver
         * stores the estimated number of collision attempts for each cell.
         */
        int* collision_count_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * @details
         * For a cell, the half-open range:
         *
         * @code
         * [cell_start_ptr[cell], cell_end_ptr[cell])
         * @endcode
         *
         * indexes into this array. Each entry contains a global particle index.
         */
        const int* indices_ptr {};

        /**
         * @brief Raw pointer to the first sorted-index position for each cell.
         */
        const int* cell_start_ptr {};

        /**
         * @brief Raw pointer to one-past-the-last sorted-index position for each cell.
         */
        const int* cell_end_ptr {};

        /**
         * @brief Optional raw pointer to per-cell solver allocation data.
         *
         * @details
         * When this pointer is non-null, cells whose allocation entry does not
         * match the current solver index are skipped by collision-statistics
         * measurement and by derived solver collision application.
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
         * @brief Number of material-property entries.
         */
        int num_of_properties {};

        /**
         * @brief Volume of one universe cell.
         *
         * @details
         * Must be positive for NTC collision-count estimation.
         */
        T cell_volume {};

        /**
         * @brief Statistical weight represented by each simulation particle.
         *
         * @details
         * Used in the NTC collision-count estimate to scale simulation-particle
         * pair statistics toward the represented physical particle population.
         */
        T statistical_weight {};

        /**
         * @brief Selected DSMC kernel type.
         */
        DsmcKernelType kernel_type { DsmcKernelType::hard_sphere };

        /**
         * @brief Runtime collision-kernel wrapper copied into device work.
         */
        DsmcKernel<T> kernel {};
    };

public:
    /**
     * @brief Constructs an empty DSMC solver.
     *
     * @details
     * The default constructor leaves the solver without configured universe,
     * fluid, or searcher dependencies. Calling `solve()` on such an object is
     * safe because collision-context initialization fails and no collision kernels
     * are launched.
     */
    DsmcSolver() = default;

    /**
     * @brief Constructs a DSMC solver with simulation dependencies and a kernel type.
     *
     * @details
     * The constructor forwards the universe, fluid, and searcher to the base
     * `Solver<T>` class, constructs the runtime DSMC kernel wrapper from
     * @p kernel_type, stores the selected kernel type, and ensures required
     * universe-side DSMC states exist.
     *
     * @param universe Universe containing cell topology and DSMC per-cell states.
     * @param fluid Fluid containing particle velocities, species, and material
     *        properties.
     * @param searcher Spatial hashing searcher used to group particles by cell.
     * @param kernel_type Collision kernel model used for cross-section evaluation
     *        and pair velocity updates.
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
     * @brief Runs DSMC collision solving without cell filtering.
     *
     * @details
     * This overload forwards to:
     *
     * @code
     * solve(nullptr, 0, dt);
     * @endcode
     *
     * Since no allocated-solver buffer is provided, all valid cells are eligible
     * for collision-statistics measurement and derived collision application.
     *
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument Thrown if @p dt is not positive after the
     *        collision context initializes successfully.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) final;

    /**
     * @brief Runs DSMC collision solving with optional per-cell solver filtering.
     *
     * @details
     * The function executes the common DSMC collision pipeline:
     *
     * @code
     * initialize_collision_context()
     * make_probe(allocated_solver, probe)
     * build_collision_workload(probe, index, dt)
     * apply_collisions(probe, index, dt)
     * @endcode
     *
     * If @p allocated_solver is non-null, only cells satisfying:
     *
     * @code
     * allocated_solver[cell] == index
     * @endcode
     *
     * participate in collision-statistics measurement. Cells assigned to other
     * solver indices are reset to zero collision statistics.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when filtering through @p allocated_solver.
     * @param dt Positive time-step size used for collision-count estimation.
     *
     * @throw std::invalid_argument Thrown if @p dt is not positive after the
     *        collision context initializes successfully.
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
     * @details
     * After @ref build_flattened_collision_workload succeeds, entry `cell` stores
     * the starting offset of that cell's collision entries in
     * @ref flattened_collision_cells.
     *
     * @return Const reference to the collision-offset buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    /**
     * @brief Returns the flattened list of cells that own collision entries.
     *
     * @details
     * The flattened buffer contains each cell repeated according to its scheduled
     * collision count. For example, if cell `7` has three scheduled collisions,
     * the flattened buffer contains three entries equal to `7`.
     *
     * @return Const reference to the flattened collision-cell buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    flattened_collision_cells() const noexcept;

    /**
     * @brief Ensures DSMC collision-statistic universe states and offset storage exist.
     *
     * @details
     * If a universe is configured, this function creates the following missing
     * states with `universe->number_of_cells()` elements:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`.
     *
     * Existing states are left unchanged. The solver-owned
     * `_collision_offsets` buffer is also resized to the universe cell count.
     *
     * If no universe is configured, this function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Clears collision statistics and temporary collision-workload buffers.
     *
     * @details
     * If no universe exists, or if the universe has no cells, this function clears
     * `_collision_offsets` and `_flattened_collision_cells`.
     *
     * Otherwise, it ensures required universe states exist and fills the following
     * buffers with zero on the device:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseMaxRelativeSpeedState<T>`,
     * - `UniverseCollisionCountState<int>`,
     * - `_collision_offsets`.
     *
     * The flattened collision-cell buffer is cleared because a reset state has no
     * scheduled collision entries.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data();

    /**
     * @brief Builds the per-cell DSMC collision workload for the current step.
     *
     * @details
     * In the current implementation, this function delegates to
     * @ref measure_cell_collision_statistics:
     *
     * @code
     * return measure_cell_collision_statistics(probe, index, dt);
     * @endcode
     *
     * It does not automatically build the flattened collision-cell list.
     * Derived solvers that require the flattened representation should call
     * @ref build_flattened_collision_workload explicitly after collision counts
     * have been measured.
     *
     * @param probe Raw-pointer runtime data prepared by @ref make_probe.
     * @param index Solver index used when the probe contains solver-allocation data.
     * @param dt Positive time-step size.
     *
     * @retval true Collision statistics were measured successfully.
     * @retval false Required context or collision input data was unavailable.
     *
     * @throw std::invalid_argument Thrown if @p dt is not positive after context
     *        initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_collision_workload(const DsmcSolverProbe& probe, int index, T dt);

    /**
     * @brief Initializes the DSMC collision context.
     *
     * @details
     * The function validates that universe, fluid, and searcher dependencies
     * exist. If any dependency is missing, collision data is reset and
     * initialization fails.
     *
     * On success, the function:
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
     * @details
     * The probe resolves:
     *
     * - fluid velocity data,
     * - fluid species data,
     * - fluid material-property data,
     * - universe collision-statistic buffers,
     * - searcher sorted-index ranges,
     * - optional allocated-solver data,
     * - scalar metadata,
     * - the selected DSMC kernel.
     *
     * This function checks for required dependencies and required states. It does
     * not fully validate all numerical preconditions such as positive particle
     * count, positive cell count, positive cell volume, or non-null searcher
     * device pointers. Those checks are performed by collision-workload
     * construction before launching device work.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer. When
     *        null, `probe.allocated_solver_ptr` is set to null.
     * @param probe Output probe populated with raw pointers and scalar metadata.
     *
     * @retval true Required dependencies and required states were found.
     * @retval false A required dependency or state was missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe(const DeviceBuffer<int>* allocated_solver, DsmcSolverProbe& probe) noexcept;

    /**
     * @brief Computes per-cell DSMC collision statistics.
     *
     * @details
     * This function measures how many collision attempts should be scheduled in
     * each cell for the current time step.
     *
     * Before launching device work, it validates the basic DSMC preconditions:
     *
     * - at least two particles,
     * - positive number of cells,
     * - at least one species material-property entry,
     * - positive cell volume,
     * - positive time step.
     *
     * If these checks fail, collision data is reset and the function returns
     * `false`, except for a non-positive time step, which raises an exception
     * after context initialization succeeds.
     *
     * For every cell, the device pass:
     *
     * - optionally filters the cell by `allocated_solver[cell] == index`,
     * - reads the searcher range `[cell_start[cell], cell_end[cell])`,
     * - counts valid particles in the cell,
     * - scans all unordered valid particle pairs,
     * - computes the maximum relative speed,
     * - computes the maximum `sigma * g`,
     * - estimates an NTC collision count,
     * - clamps the result to the number of available unordered pairs,
     * - writes particle count, maximum relative speed, and collision count.
     *
     * The NTC estimate is:
     *
     * @code
     * pair_count = count * (count - 1) * 0.5;
     * ntc_count  = pair_count * max_sigma_g * statistical_weight * dt / cell_volume;
     * collisions = floor(ntc_count);
     * @endcode
     *
     * Cells with no matching solver allocation, invalid or empty ranges, fewer
     * than two valid particles, or non-positive `max_sigma_g` receive zero
     * collision count.
     *
     * @param probe Raw-pointer runtime data prepared by @ref make_probe.
     * @param index Solver index used when the probe contains solver-allocation data.
     * @param dt Positive time-step size used in the NTC estimate.
     *
     * @retval true The device pass was launched after successful precondition checks.
     * @retval false Required probe data or DSMC preconditions were unavailable.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DsmcSolverProbe& probe, int index, T dt);

    /**
     * @brief Builds a flattened collision-cell workload from per-cell collision counts.
     *
     * @details
     * The function reads `UniverseCollisionCountState<int>`, computes exclusive
     * prefix offsets into `_collision_offsets`, and fills
     * `_flattened_collision_cells` so that each cell appears once per scheduled
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
     * function returns `false`. If the total number of scheduled collisions is
     * zero, `_flattened_collision_cells` is cleared and the function returns
     * `false`.
     *
     * @retval true A non-empty flattened workload was created.
     * @retval false Collision-count state was missing or total collisions were zero.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload() noexcept;

    /**
     * @brief Returns the global particle index at a local sorted offset.
     *
     * @details
     * This helper computes:
     *
     * @code
     * sorted_index = begin + nth;
     * @endcode
     *
     * and returns:
     *
     * @code
     * indices_ptr[sorted_index]
     * @endcode
     *
     * only when both the sorted index and resulting global particle index are
     * valid.
     *
     * Despite the name, this function does not scan over invalid entries to find
     * the N-th valid particle. It checks only the direct sorted entry at
     * `begin + nth`.
     *
     * @param nth Zero-based local offset from @p begin.
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
     * @details
     * The helper maps a zero-based ordinal in the conceptual upper-triangular
     * pair list to:
     *
     * - @p lhs_local, written by reference,
     * - the right-hand local index, returned by value.
     *
     * Pairs are enumerated row by row:
     *
     * @code
     * ordinal 0: (0, 1)
     * ordinal 1: (0, 2)
     * ...
     * ordinal count - 2: (0, count - 1)
     * next: (1, 2)
     * next: (1, 3)
     * ...
     * @endcode
     *
     * For `count = 4`, the ordinal mapping is:
     *
     * @code
     * 0 -> (0, 1)
     * 1 -> (0, 2)
     * 2 -> (0, 3)
     * 3 -> (1, 2)
     * 4 -> (1, 3)
     * 5 -> (2, 3)
     * @endcode
     *
     * If the ordinal is outside the valid range, the function returns `-1`.
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
     * @details
     * This pure virtual function is called by @ref solve after collision workload
     * construction succeeds.
     *
     * Derived classes implement the concrete strategy for converting per-cell
     * collision counts into particle-pair velocity updates.
     *
     * Examples:
     *
     * - a disjoint-pair solver may process `(0, 1), (2, 3), ...`,
     * - an NTC solver may cycle through unordered local pair ordinals,
     * - another solver may consume the flattened collision-cell workload.
     *
     * @param probe Raw-pointer runtime data prepared by @ref make_probe.
     * @param index Solver index used when the probe contains solver-allocation data.
     * @param dt Time-step size associated with this collision solve.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collisions(const DsmcSolverProbe& probe, int index, T dt)
        = 0;

protected:
    /**
     * @brief Runtime DSMC kernel wrapper used by derived collision code.
     *
     * @details
     * The kernel object is copied into probes and device kernels. It performs
     * cross-section evaluation and velocity updates according to the selected
     * `DsmcKernelType`.
     */
    DsmcKernel<T> _kernel {};

    /**
     * @brief Selected DSMC kernel model.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    /**
     * @brief Per-cell exclusive offsets into the flattened collision workload.
     *
     * @details
     * Resized to the universe cell count by @ref ensure_universe_states and
     * populated by @ref build_flattened_collision_workload.
     */
    DeviceBuffer<int> _collision_offsets {};

    /**
     * @brief Flattened list of collision-owning cells.
     *
     * @details
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
 * @tparam T Floating-point scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Floating-point scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::DsmcSolver`.
 *
 * @tparam T Floating-point scalar type used by the DSMC solver.
 */
template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_solver.hpp>