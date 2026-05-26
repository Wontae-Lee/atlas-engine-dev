#pragma once

/**
 * @file dsmc_cell_sequential_solver.h
 * @brief Declares a DSMC solver that applies collision trials sequentially inside each cell.
 *
 * This header defines `atlas::system::DsmcCellSequentialSolver<T>`, a concrete
 * DSMC solver that uses the common collision-statistics pipeline provided by
 * `DsmcSolver<T>` and then applies collision trials cell by cell.
 */

#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief DSMC solver that processes collision trials sequentially within each cell.
 *
 * `DsmcCellSequentialSolver<T>` is a concrete @ref DsmcSolver implementation.
 * The base class prepares the DSMC runtime context, measures per-cell collision
 * statistics, and stores the result in the cached DSMC probe. This class then
 * implements @ref apply_collision to perform the actual collision trials.
 *
 * The collision-application strategy is:
 *
 * - launch one device task per universe cell,
 * - skip cells that are not assigned to this solver when an allocation buffer is provided,
 * - read the precomputed collision count and majorant sigma-g for the cell,
 * - sample unordered particle pairs through deterministic hashed sampling,
 * - accept or reject each candidate collision using `sigma_g / max_sigma_g`,
 * - apply the configured DSMC kernel to accepted pairs,
 * - write updated velocities back to the fluid velocity buffer.
 *
 * Collision trials inside one cell are executed in a sequential loop inside the
 * cell's device task. Different cells are processed in parallel.
 *
 * @tparam T Scalar type used by the DSMC solver.
 */
template <typename T>
class DsmcCellSequentialSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Fluent builder for constructing validated `DsmcCellSequentialSolver` instances.
     */
    class Builder;

    /**
     * @brief Alias for the DSMC probe type provided by the base solver.
     *
     * The probe contains raw pointers to particle data, cell ranges, collision
     * statistics, material properties, and the DSMC kernel.
     */
    using Probe = typename DsmcSolver<T>::DsmcSolverProbe;

public:
    /**
     * @brief Constructs an empty cell-sequential DSMC solver.
     *
     * Dependencies are initialized by the base/default state. Calling `solve` on
     * an unconfigured solver is safe because the base solver validates required
     * dependencies before collision application.
     */
    DsmcCellSequentialSolver() = default;

    /**
     * @brief Constructs a cell-sequential DSMC solver from simulation dependencies.
     *
     * The constructor forwards the universe, fluid, searcher, and kernel type to
     * the @ref DsmcSolver base class. The base class creates or resizes required
     * DSMC universe states during construction when a universe is available.
     *
     * @param universe Universe containing per-cell DSMC statistic states.
     * @param fluid Fluid containing particle velocity, species, and material data.
     * @param searcher Spatial hashing searcher used to group particles by cell.
     * @param kernel_type DSMC collision kernel model used for accepted collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcCellSequentialSolver(UniverseHostPtr<T> universe,
                             FluidHostPtr<T> fluid,
                             SpatialHashingSearcherHostPtr<T> searcher,
                             DsmcKernelType kernel_type = DsmcKernelType::hard_sphere,
                             bool pairing_without_replacement = false) noexcept;

    /**
     * @brief Destroys the solver through the base interface.
     */
    ~DsmcCellSequentialSolver() override = default;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct a solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Applies DSMC collisions using a cell-local sequential trial loop.
     *
     * This function is called by the @ref DsmcSolver base class after successful
     * collision-statistics measurement. It reads the cached base probe and returns
     * immediately if required probe pointers or cell counts are invalid.
     *
     * For each selected cell, the implementation:
     *
     * - reads `collision_count[cell]`, `number_particle[cell]`, and `max_sigma_g[cell]`,
     * - skips cells with no scheduled collisions, fewer than two particles, or non-positive majorant,
     * - repeatedly samples two distinct local particle indices,
     * - maps local indices to global particle indices with `DsmcSolver<T>::nth_valid_particle`,
     * - computes the candidate pair's `sigma_g`,
     * - accepts the trial with probability `min(sigma_g / max_sigma_g, 1)`,
     * - applies the configured DSMC kernel to accepted velocity pairs.
     *
     * When `allocated_solver` is non-null, only cells with
     * `allocated_solver[cell] == index` are processed.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Time-step size associated with the current solve step. The current
     *           implementation does not use this value directly.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_random_pairing_collision(Probe probe, const int* allocated_solver_ptr, int index) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_pairing_without_replacement_collision(Probe probe, const int* allocated_solver_ptr, int index) const;
};

/**
 * @brief Fluent builder for `DsmcCellSequentialSolver`.
 *
 * The builder collects required solver dependencies and the DSMC kernel
 * configuration, then constructs a validated cell-sequential solver.
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
class DsmcCellSequentialSolver<T>::Builder final {
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
    with_pairing_without_replacement(bool enabled) noexcept;

    /**
     * @brief Builds a validated cell-sequential DSMC solver value.
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
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcCellSequentialSolver<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared cell-sequential DSMC solver.
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
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcCellSequentialSolver<T>>
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

    bool _pairing_without_replacement {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcCellSequentialSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcCellSequentialSolver = atlas::system::DsmcCellSequentialSolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::DsmcCellSequentialSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcCellSequentialSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcCellSequentialSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::DsmcCellSequentialSolver`.
 *
 * @tparam T Scalar type used by the solver.
 */
template <typename T>
using DsmcCellSequentialSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcCellSequentialSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_cell_sequential_solver.hpp>
