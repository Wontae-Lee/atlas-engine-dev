#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief DSMC solver variant that flattens cell-local collision work into a linear work queue.
 *
 * @tparam T Floating-point scalar type used by the DSMC solver.
 *
 * @details
 * `DsmcFlattenSolver` extends `DsmcSolver` by converting per-cell collision counts
 * into a flat list of collision work items. Instead of launching one device thread
 * per cell and looping over all collision attempts inside that thread, this solver
 * builds a linear collision schedule and launches one device work item per sampled
 * collision attempt.
 *
 * The flattening process is:
 *
 * 1. Read per-cell collision counts computed by `measure_cell_collision_statistics()`.
 * 2. Optionally filter collision counts using an external solver-allocation map.
 * 3. Compute exclusive offsets from the scheduled collision counts.
 * 4. Fill `collision_cells`, where each flattened work index maps back to a cell.
 * 5. Launch collision attempts over the flattened collision range.
 *
 * This can expose more parallelism when many collision attempts are concentrated
 * in a relatively small number of cells.
 *
 * @note
 * This solver reuses the DSMC pair-collision logic from `DsmcSolver`.
 */
template <typename T>
class DsmcFlattenSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Probe type used to pass DSMC state and raw pointers into device kernels.
     */
    using Probe = atlas::system::DsmcProbe<T>;

    /**
     * @brief Inherits constructors from the base DSMC solver.
     *
     * @details
     * This allows `DsmcFlattenSolver` to be constructed with the same universe,
     * fluid, searcher, and kernel-type arguments as `DsmcSolver`.
     */
    using DsmcSolver<T>::DsmcSolver;

    /**
     * @brief Builder type for constructing validated `DsmcFlattenSolver` instances.
     */
    class Builder;

    /**
     * @brief Creates a builder for `DsmcFlattenSolver`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Exclusive prefix offsets of scheduled collision counts.
     *
     * @details
     * For cell `i`, `collision_offsets[i]` stores the first flattened work index
     * assigned to that cell. The number of entries for the cell is determined by
     * the scheduled collision count of that cell.
     */
    atlas::DeviceBuffer<int> collision_offsets {};

    /**
     * @brief Flattened mapping from collision work index to cell index.
     *
     * @details
     * Each element stores the cell that owns the corresponding flattened collision
     * attempt. For example, if `collision_cells[k] == c`, then flattened work item
     * `k` performs one DSMC collision attempt inside cell `c`.
     */
    atlas::DeviceBuffer<int> collision_cells {};

    /**
     * @brief Per-cell collision counts after optional solver-allocation filtering.
     *
     * @details
     * This buffer is used only when an external `allocated_solver` map is provided.
     * Cells not assigned to the requested solver index receive a scheduled collision
     * count of zero.
     */
    atlas::DeviceBuffer<int> filtered_collision_counts {};

    /**
     * @brief Total number of flattened collision work items.
     *
     * @details
     * This value equals the sum of all scheduled per-cell collision counts after
     * optional filtering.
     */
    int flattened_collision_count {};

    /**
     * @brief Returns the collision-offset buffer.
     *
     * @return Const reference to the exclusive collision-offset buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::DeviceBuffer<int>&
    offsets() const noexcept;

    /**
     * @brief Clears all flattening buffers and resets the flattened work count.
     *
     * @details
     * This releases the current collision schedule by resizing all internal device
     * buffers to zero and setting `flattened_collision_count` to zero.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear();

    /**
     * @brief Builds the flattened collision schedule from per-cell collision counts.
     *
     * @param collision_count_ptr Device pointer to per-cell collision counts.
     * @param num_of_cells Number of simulation cells.
     * @param allocated_solver_ptr Optional device pointer to per-cell solver assignments.
     * @param index Solver index used to filter cells when `allocated_solver_ptr` is set.
     *
     * @return `true` if at least one flattened collision work item is generated;
     *         otherwise `false`.
     *
     * @details
     * This method prepares the buffers required by the flattened collision pass.
     * If `allocated_solver_ptr` is provided, collision counts are first filtered so
     * that only cells assigned to `index` contribute work. Then an exclusive scan
     * computes `collision_offsets`, and `collision_cells` is filled so that each
     * flattened work item can recover its owning cell.
     *
     * @throws std::overflow_error
     * Thrown if the total flattened collision count exceeds the range of `int`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build(int* collision_count_ptr,
          int num_of_cells,
          const int* allocated_solver_ptr = nullptr,
          int index = 0);

    /**
     * @brief Applies DSMC collisions using a flattened collision work queue.
     *
     * @param allocated_solver Optional per-cell solver assignment buffer.
     * @param index Solver index used to filter cells when `allocated_solver` is set.
     * @param dt Simulation time step.
     *
     * @details
     * This override replaces the base solver's cell-major collision loop with a
     * flattened work-item loop. It first calls `build()` using the collision counts
     * stored in the current DSMC probe. Then it launches one device work item for
     * each scheduled collision attempt.
     *
     * Each flattened work item:
     *
     * 1. Looks up its owning cell from `collision_cells`.
     * 2. Computes the local collision index using `collision_offsets`.
     * 3. Samples two distinct local particles from the cell.
     * 4. Calls `collide_pair()` to perform acceptance-rejection and velocity update.
     *
     * @note
     * The `dt` parameter is not used directly in this stage because the time-step
     * dependence is already encoded in the per-cell collision counts.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

/**
 * @brief Builder for validated construction of `DsmcFlattenSolver`.
 *
 * @tparam T Floating-point scalar type used by the DSMC solver.
 *
 * @details
 * The builder collects the required simulation objects and optional kernel type
 * before constructing a `DsmcFlattenSolver`. It validates that universe, fluid,
 * and spatial searcher pointers are all provided before building.
 */
template <typename T>
class DsmcFlattenSolver<T>::Builder final {
public:
    /**
     * @brief Sets the universe used by the solver.
     *
     * @param universe Shared host pointer to the simulation universe.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid storage used by the solver.
     *
     * @param fluid Shared host pointer to the particle fluid storage.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher used by the solver.
     *
     * @param searcher Shared host pointer to the spatial hashing searcher.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the DSMC collision kernel type.
     *
     * @param kernel_type Collision kernel type used by the solver.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Validates that all required construction inputs are present.
     *
     * @throws std::runtime_error
     * Thrown if universe, fluid, or searcher has not been set.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

    /**
     * @brief Builds a `DsmcFlattenSolver` value.
     *
     * @return Constructed `DsmcFlattenSolver`.
     *
     * @throws std::runtime_error
     * Thrown if required construction inputs are missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcFlattenSolver<T>
    build() const;

    /**
     * @brief Builds a host shared pointer to `DsmcFlattenSolver`.
     *
     * @return Host shared pointer owning a constructed `DsmcFlattenSolver`.
     *
     * @throws std::runtime_error
     * Thrown if required construction inputs are missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcFlattenSolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Universe used by the constructed solver.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid storage used by the constructed solver.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher used by the constructed solver.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Collision kernel type used by the constructed solver.
     */
    DsmcKernelType _kernel_type = DsmcKernelType::hard_sphere;
};

}

namespace atlas {

/**
 * @brief Public alias for the system-level flattened DSMC solver.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcFlattenSolver = atlas::system::DsmcFlattenSolver<T>;

/**
 * @brief Host shared-pointer alias for `atlas::system::DsmcFlattenSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcFlattenSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

/**
 * @brief Device shared-pointer alias for `atlas::system::DsmcFlattenSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcFlattenSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_flatten_solver.hpp>