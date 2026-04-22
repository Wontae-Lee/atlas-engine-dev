#pragma once

/**
 * @file dsmc_solver.h
 * @brief Declares the common DSMC solver base class shared by DSMC variants.
 *
 * This header defines `DsmcSolver<T>`, an abstract base class that provides the
 * common infrastructure required by DSMC collision solvers.
 *
 * Responsibilities of this base class include:
 * - storing the selected DSMC collision kernel
 * - managing common collision-workload buffers
 * - preparing universe states required by DSMC collision scheduling
 * - building flattened collision worklists
 * - exposing helper utilities for cell-local particle indexing
 *
 * Derived classes are responsible only for the final collision-application step
 * through `apply_collisions(...)`, while this base class handles the shared
 * orchestration and precomputation stages.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Abstract base class for DSMC collision solvers.
 *
 * `DsmcSolver<T>` provides the common runtime pipeline used by DSMC-based solvers.
 * It inherits from `Solver<T>` and adds DSMC-specific support for:
 * - kernel selection
 * - collision workload construction
 * - collision-statistics measurement
 * - flattened collision scheduling buffers
 *
 * The typical high-level flow is:
 * 1. ensure required universe states exist
 * 2. reset collision-related temporary data
 * 3. measure per-cell collision statistics
 * 4. build a flattened collision workload
 * 5. delegate actual collision application to the derived solver
 *
 * Derived classes specialize only the final collision execution strategy, such as:
 * - classic NTC sampling
 * - disjoint local-pair processing
 * - other cell-local pairing/scheduling schemes
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcSolver : public Solver<T> {
public:
    /**
     * @brief Construct a default-initialized DSMC solver base.
     *
     * The solver remains unconfigured until the universe, fluid, and searcher
     * dependencies are provided through the parameterized constructor of this
     * class or of a derived class.
     */
    DsmcSolver() = default;

    /**
     * @brief Construct a DSMC solver base with all required runtime dependencies.
     *
     * This constructor stores:
     * - the target universe
     * - the target fluid
     * - the spatial hashing searcher
     * - the selected kernel type
     *
     * It also prepares the runtime kernel wrapper used during collision execution.
     *
     * @param universe Host-side universe used for per-cell collision statistics.
     * @param fluid Host-side fluid containing particle states and material data.
     * @param searcher Spatial hashing searcher used for cell-local particle grouping.
     * @param kernel_type Concrete DSMC kernel model used for collision resolution.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroy the solver base.
     */
    ~DsmcSolver() override = default;

    /**
     * @brief Solve collisions for the full cell set.
     *
     * This overload runs the DSMC collision pipeline over all cells handled by
     * the associated universe/searcher configuration.
     *
     * Internally, the function typically:
     * - prepares the collision context
     * - builds the collision workload
     * - invokes the derived solver's `apply_collisions(...)`
     *
     * @param dt Time step used by collision scheduling and solver execution.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) final;

    /**
     * @brief Solve collisions for a filtered subset of cells.
     *
     * When `allocated_solver` is provided, only cells whose assignment matches
     * `index` are considered by the workload-construction and collision-application
     * stages. This enables partitioned solver scheduling across the universe.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step used by collision scheduling and solver execution.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) final;

    /**
     * @brief Return the configured DSMC kernel type.
     *
     * @return Active kernel type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    /**
     * @brief Return the prefix-sum offsets of the flattened collision workload.
     *
     * This buffer is typically used to locate the collision-work entries belonging
     * to each cell after the collision workload has been flattened into a compact
     * linear representation.
     *
     * @return Reference to the collision-offset buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    /**
     * @brief Return the flattened list of cells that contribute collision work.
     *
     * This buffer stores a compact, flattened representation of collision-carrying
     * cells and is used by solver implementations that iterate over the collision
     * workload in linear form rather than by sparse cell scans alone.
     *
     * @return Reference to the flattened collision-cell buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    flattened_collision_cells() const noexcept;

    /**
     * @brief Ensure that all universe states required by DSMC scheduling exist.
     *
     * DSMC workload construction depends on certain per-cell universe states such as:
     * - number of particles
     * - maximum relative speed
     * - collision count
     *
     * This function creates or prepares those states when necessary so later
     * scheduling stages can rely on them being present.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Reset temporary collision scheduling data.
     *
     * This function clears or reinitializes solver-owned collision buffers such as:
     * - `_collision_offsets`
     * - `_flattened_collision_cells`
     *
     * It is typically called before rebuilding the collision workload for a new step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data();

    /**
     * @brief Build the complete collision workload for the current step.
     *
     * This is the high-level orchestration entry point for collision scheduling.
     * It generally performs:
     * - collision-context initialization
     * - per-cell collision-statistics measurement
     * - flattened workload construction
     *
     * The final return value indicates whether a usable collision workload was
     * successfully constructed.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step used when estimating collision statistics.
     * @return True when the collision workload was successfully built.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_collision_workload(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Initialize the DSMC collision context.
     *
     * This function prepares the base solver state before collision statistics are
     * measured. It typically includes:
     * - ensuring required universe states exist
     * - resetting temporary collision buffers
     *
     * @return True when initialization succeeded and collision scheduling may continue.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_collision_context() noexcept;

    /**
     * @brief Measure per-cell collision statistics needed for workload construction.
     *
     * This stage computes or refreshes collision-control quantities stored in the
     * universe state, such as:
     * - particle count per cell
     * - maximum relative speed
     * - number of collisions to attempt in each cell
     *
     * These statistics are then used by derived solvers when selecting or applying
     * collision pairs.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step used in collision-statistics estimation.
     * @return True when the measurement stage completed successfully.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Build a compact flattened collision workload representation.
     *
     * After per-cell collision statistics have been measured, this function converts
     * sparse cell-wise collision information into compact device buffers suitable
     * for downstream solver execution.
     *
     * Typical outputs are:
     * - `_collision_offsets`
     * - `_flattened_collision_cells`
     *
     * @return True when the flattened workload was built successfully.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload() noexcept;

    /**
     * @brief Resolve the N-th valid particle index inside a cell-local sorted range.
     *
     * Given:
     * - a desired local ordinal `nth`
     * - the sorted cell range `[begin, end)`
     * - the global particle count
     * - the searcher's sorted particle-index array
     *
     * this helper returns the corresponding valid global particle index, skipping
     * invalid entries if necessary.
     *
     * This function is useful for derived solvers that work in cell-local index
     * space but must eventually read/write global particle-state buffers.
     *
     * @param nth Zero-based ordinal of the valid particle to locate inside the cell.
     * @param begin First sorted-index position belonging to the cell.
     * @param end One-past-last sorted-index position belonging to the cell.
     * @param particle_count Global particle-count bound used for safety checks.
     * @param indices_ptr Pointer to the searcher's sorted particle-index array.
     * @return Global particle index, or a negative value when no such valid particle exists.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    nth_valid_particle(int nth,
                       int begin,
                       int end,
                       int particle_count,
                       const int* indices_ptr) noexcept;

    /**
     * @brief Map a pair ordinal to the corresponding right-hand local index.
     *
     * This helper is used by pair-enumeration logic that conceptually traverses
     * unordered particle pairs inside a cell.
     *
     * For a cell-local particle count `count`, a pair ordinal can be decoded into:
     * - `lhs_local`, returned by reference
     * - rhs local index, returned as the function result
     *
     * This allows pair iteration without explicitly materializing all pair
     * combinations in memory.
     *
     * @param count Number of particles in the local cell.
     * @param ordinal Zero-based ordinal of the pair in the conceptual pair list.
     * @param lhs_local Output left-hand local particle index.
     * @return Right-hand local particle index.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    pair_ordinal_to_rhs(int count, int ordinal, int& lhs_local) noexcept;

    /**
     * @brief Apply collisions using the concrete strategy implemented by a derived solver.
     *
     * This is the only strategy-specific stage in the DSMC base workflow.
     * All common preparation and workload-construction logic is handled by
     * `DsmcSolver<T>`, while derived classes implement how accepted collisions
     * are actually traversed and applied.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step associated with the current collision step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt)
        = 0;

protected:
    /**
     * @brief Runtime-dispatchable collision kernel wrapper.
     *
     * This object stores the concrete collision model currently selected for the solver.
     */
    DsmcKernel<T> _kernel {};

    /**
     * @brief Tag describing which concrete collision kernel is active.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    /**
     * @brief Prefix offsets into the flattened collision workload.
     *
     * This buffer is typically used to locate where each cell's collision work
     * begins inside a flattened workload representation.
     */
    DeviceBuffer<int> _collision_offsets {};

    /**
     * @brief Flattened list of collision-carrying cells.
     *
     * This compact device buffer stores the cells that contribute collision work
     * after sparse per-cell statistics have been flattened into a linear schedule.
     */
    DeviceBuffer<int> _flattened_collision_cells {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcSolver<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

/**
 * @brief Host-shared-pointer alias for `DsmcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `DsmcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_solver.hpp>