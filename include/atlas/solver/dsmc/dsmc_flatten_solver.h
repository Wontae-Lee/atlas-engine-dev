#pragma once

/**
 * @file dsmc_flatten_solver.h
 * @brief Declares a DSMC solver that flattens per-cell collision counts into a linear work queue.
 *
 * This header defines `atlas::system::DsmcFlattenSolver<T>`, a concrete DSMC
 * solver that converts per-cell collision counts into a flattened collision
 * workload and executes collision trials over that linear work range.
 */

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief DSMC solver that executes collision trials from a flattened work queue.
 *
 * `DsmcFlattenSolver<T>` is a concrete @ref DsmcSolver implementation. The base
 * solver prepares the DSMC runtime context and measures per-cell collision
 * statistics. This class then converts the per-cell collision counts into a
 * compact flattened list where each entry represents one collision trial.
 *
 * The flattened workload construction is:
 *
 * @code
 * collision_offsets = exclusive_scan(collision_count);
 * collision_cells[offset[cell] + local_collision] = cell;
 * @endcode
 *
 * After flattening, `apply_collision` launches one device work item per scheduled
 * collision trial. Each work item resolves its owning cell, samples a candidate
 * particle pair from that cell, evaluates the acceptance probability, and applies
 * the selected DSMC kernel when the collision is accepted.
 *
 * This strategy exposes more parallelism than a cell-sequential implementation
 * when many collision trials are scheduled, because collision trials are mapped
 * directly to the device work range.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
class DsmcFlattenSolver : public DsmcSolver<T> {
public:
    /**
     * @brief Fluent builder for constructing validated `DsmcFlattenSolver` instances.
     */
    class Builder;

    /**
     * @brief Alias for the DSMC probe type provided by the base solver.
     *
     * The probe contains raw pointers to fluid data, universe collision statistics,
     * searcher ranges, material properties, and the runtime DSMC kernel.
     */
    using Probe = typename DsmcSolver<T>::DsmcSolverProbe;

public:
    /**
     * @brief Constructs an empty flattened DSMC solver.
     *
     * Dependencies are initialized by the base/default state. Calling `solve` on
     * an unconfigured solver is safe because the base solver validates required
     * dependencies before invoking collision application.
     */
    DsmcFlattenSolver() = default;

    /**
     * @brief Constructs a flattened DSMC solver from simulation dependencies.
     *
     * The constructor forwards the universe, fluid, searcher, and kernel type to
     * the @ref DsmcSolver base class. Required DSMC universe states are prepared
     * by the base constructor when a universe is available.
     *
     * @param universe Universe containing per-cell DSMC statistic states.
     * @param fluid Fluid containing particle velocity, species, and material data.
     * @param searcher Spatial hashing searcher used to group particles by cell.
     * @param kernel_type DSMC collision kernel model used for accepted collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcFlattenSolver(UniverseHostPtr<T> universe,
                      FluidHostPtr<T> fluid,
                      SpatialHashingSearcherHostPtr<T> searcher,
                      DsmcKernelType kernel_type = DsmcKernelType::hard_sphere,
                      bool prevent_duplicate_pairing = false) noexcept;

    /**
     * @brief Destroys the solver through the base interface.
     */
    ~DsmcFlattenSolver() override = default;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct a solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Returns the per-cell offsets into the flattened collision workload.
     *
     * After @ref build_flattened_collision_workload succeeds, entry `cell`
     * contains the starting offset of that cell's collision trials in the
     * flattened `_collision_cells` buffer.
     *
     * @return Const reference to the collision-offset buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    /**
     * @brief Resets base DSMC statistic states and flattened workload buffers.
     *
     * This override first calls `DsmcSolver<T>::reset_states()` to clear per-cell
     * DSMC statistics, then clears:
     *
     * - `_collision_offsets`,
     * - `_collision_cells`,
     * - `_flattened_collision_count`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_states() override;

    /**
     * @brief Builds a flattened collision workload from per-cell collision counts.
     *
     * This function reads `UniverseCollisionCountState<int>`, computes exclusive
     * prefix offsets into `_collision_offsets`, and fills `_collision_cells` so
     * each scheduled collision trial maps back to its owning cell.
     *
     * If the collision-count state is missing, the solver resets its states and
     * returns `false`. If the total collision count is zero, the flattened buffer
     * is cleared and the function returns `false`.
     *
     * The function checks for integer overflow before computing the total number
     * of flattened collision trials.
     *
     * @retval true A non-empty flattened collision workload was created.
     * @retval false Collision-count state was missing or no collisions were scheduled.
     *
     * @throw std::overflow_error If the flattened workload would exceed `int` range.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload();

    /**
     * @brief Applies DSMC collisions over the flattened collision work queue.
     *
     * This function first calls @ref build_flattened_collision_workload. If no
     * flattened work is available, it returns immediately.
     *
     * For each flattened work item, the implementation:
     *
     * - resolves the owning cell from `_collision_cells`,
     * - computes the local collision ordinal from `_collision_offsets`,
     * - samples two distinct local particle indices in the cell,
     * - maps them to global particle indices with `DsmcSolver<T>::nth_valid_particle`,
     * - computes the pair-specific `sigma_g`,
     * - accepts the candidate with probability `min(sigma_g / max_sigma_g, 1)`,
     * - applies the configured DSMC kernel to accepted velocity pairs,
     * - writes accepted velocity updates back to the fluid velocity buffer.
     *
     * The `allocated_solver`, `index`, and `dt` arguments are accepted for
     * interface compatibility. Cell filtering has already been reflected in the
     * base solver's collision-count measurement, and `dt` is not used directly in
     * this stage.
     *
     * @param allocated_solver Unused in this stage; filtering is already encoded in collision counts.
     * @param index Unused in this stage; filtering is already encoded in collision counts.
     * @param dt Unused time-step value for this collision-application stage.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

private:
    /**
     * @brief Per-cell prefix offsets into `_collision_cells`.
     *
     * Populated by @ref build_flattened_collision_workload using an exclusive scan
     * over per-cell collision counts.
     */
    DeviceBuffer<int> _collision_offsets {};

    /**
     * @brief Flattened list mapping each collision work item to its owning cell.
     *
     * Each cell appears once for each scheduled collision trial in that cell.
     */
    DeviceBuffer<int> _collision_cells {};

    /**
     * @brief Number of active entries in the flattened collision workload.
     */
    int _flattened_collision_count {};
};

/**
 * @brief Fluent builder for `DsmcFlattenSolver`.
 *
 * The builder collects required dependencies and DSMC kernel configuration, then
 * constructs a validated flattened DSMC solver.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * Optional configuration:
 *
 * - DSMC kernel type, defaulting through the default-constructed `DsmcKernel<T>`.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
class DsmcFlattenSolver<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @param universe Universe containing per-cell DSMC statistic states.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @param fluid Fluid containing particle velocity, species, and material data.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @param searcher Searcher used to expose cell-local particle ranges.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the DSMC collision kernel type.
     *
     * The builder stores the selection by constructing an internal
     * `DsmcKernel<T>` and later passing its `type` to the solver constructor.
     *
     * @param kernel_type DSMC kernel type used by the constructed solver.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_prevent_duplicate_pairing(bool enabled) noexcept;

    /**
     * @brief Builds a validated flattened DSMC solver value.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null.
     *
     * @return Constructed solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcFlattenSolver<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared flattened DSMC solver.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null.
     *
     * @return Host-side shared pointer to the constructed solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcFlattenSolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates that all required builder dependencies are configured.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Universe dependency collected by the builder.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency collected by the builder.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency collected by the builder.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief DSMC kernel configuration collected by the builder.
     *
     * The constructed solver receives `_kernel.type`.
     */
    DsmcKernel<T> _kernel {};

    bool _prevent_duplicate_pairing {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcFlattenSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcFlattenSolver = atlas::system::DsmcFlattenSolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::DsmcFlattenSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcFlattenSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::DsmcFlattenSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcFlattenSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcFlattenSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_flatten_solver.hpp>
