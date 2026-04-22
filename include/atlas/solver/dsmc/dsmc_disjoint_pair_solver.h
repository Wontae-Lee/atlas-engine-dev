#pragma once

/**
 * @file dsmc_disjoint_pair_solver.h
 * @brief Declares a DSMC solver variant that processes only disjoint local particle pairs in each cell.
 *
 * This solver is a specialized DSMC collision solver derived from `DsmcSolver<T>`.
 * Instead of selecting arbitrary collision pairs inside a cell, it processes only
 * non-overlapping local pairs formed from the cell-local particle ordering.
 *
 * Conceptually, particles inside a cell are paired as:
 * - local pair (0, 1)
 * - local pair (2, 3)
 * - local pair (4, 5)
 * - ...
 *
 * Only these disjoint pairs are eligible for collision processing during a single
 * solver pass. The number of actually processed pairs is additionally limited by
 * the per-cell collision count stored in the universe state.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief DSMC solver that applies collisions only to disjoint local particle pairs.
 *
 * This solver implements a restricted collision strategy in which each particle
 * may participate in at most one collision pair during a single pass over a cell.
 * Pairing is derived from the local particle ordering inside each search cell.
 *
 * Relative to more general DSMC pairing strategies, this solver:
 * - avoids overlapping pair reuse within the same pass
 * - is straightforward to evaluate in parallel per cell
 * - limits the number of candidate collisions to `floor(count / 2)` per cell
 *
 * The final number of processed collisions in a cell is the minimum of:
 * - the number of disjoint local pairs available in that cell
 * - the collision count requested by the universe collision-count state
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcDisjointPairSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `DsmcDisjointPairSolver`.
     */
    class Builder;

public:
    /**
     * @brief Construct a default-initialized solver.
     *
     * This constructor leaves the solver unconfigured. A fully usable instance
     * is typically created through the parameterized constructor or the builder.
     */
    DsmcDisjointPairSolver() = default;

    /**
     * @brief Construct a disjoint-pair DSMC solver with all required dependencies.
     *
     * The solver stores references to the target universe, fluid, and searcher,
     * and initializes the base `DsmcSolver<T>` with the selected collision kernel.
     *
     * @param universe Host-side universe containing cell topology and per-cell states.
     * @param fluid Host-side fluid containing particle properties and particle states.
     * @param searcher Spatial hashing searcher that provides cell-local particle ordering.
     * @param kernel_type Collision kernel type used when resolving particle collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcDisjointPairSolver(UniverseHostPtr<T> universe,
                           FluidHostPtr<T> fluid,
                           SpatialHashingSearcherHostPtr<T> searcher,
                           DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroy the solver.
     */
    ~DsmcDisjointPairSolver() override = default;

    /**
     * @brief Create a builder for `DsmcDisjointPairSolver`.
     *
     * @return Default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Apply DSMC collisions to disjoint local particle pairs.
     *
     * This function processes eligible cells and resolves collisions between
     * non-overlapping local particle pairs determined from the searcher's
     * cell-local particle ordering.
     *
     * For a cell with local particle order:
     * - 0, 1, 2, 3, 4, 5, ...
     *
     * the solver attempts collisions on:
     * - (0, 1)
     * - (2, 3)
     * - (4, 5)
     * - ...
     *
     * The number of attempted collisions in each cell is limited by both:
     * - the number of available disjoint pairs, `floor(count / 2)`
     * - the per-cell collision count stored in `UniverseCollisionCountState<int>`
     *
     * If `allocated_solver` is provided, only cells whose assigned solver index
     * matches `index` are processed. This allows multiple solver instances or
     * solver passes to partition work across cells.
     *
     * This function requires the following runtime states to be present:
     * - `FluidVelocityState<T>`
     * - `FluidSpeciesState<T>`
     * - `UniverseNumberParticleState<T>`
     * - `UniverseCollisionCountState<int>`
     *
     * If any of these states are missing, the function returns without performing
     * any collision work.
     *
     * @param allocated_solver Optional device buffer that stores, for each cell,
     *        the solver index assigned to process that cell. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step size associated with the current update. This solver
     *        interface accepts it for polymorphic compatibility, although this
     *        implementation may not use it directly inside the pairing loop.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

/**
 * @brief Builder for `DsmcDisjointPairSolver`.
 *
 * This builder collects the required runtime dependencies and optional solver
 * configuration before constructing a validated solver instance.
 *
 * Required dependencies:
 * - universe
 * - fluid
 * - searcher
 *
 * Optional configuration:
 * - collision kernel type
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcDisjointPairSolver<T>::Builder final {
public:
    /**
     * @brief Construct a default-initialized builder.
     */
    Builder() = default;

    /**
     * @brief Set the universe used by the solver.
     *
     * The universe provides:
     * - the simulation cell topology
     * - the number of cells
     * - per-cell diagnostic and collision-control states
     *
     * @param universe Host-side universe instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Set the fluid used by the solver.
     *
     * The fluid provides:
     * - particle states such as velocity and species
     * - species/material properties used by the collision kernel
     * - the global particle count
     *
     * @param fluid Host-side fluid instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the spatial hashing searcher used by the solver.
     *
     * The searcher provides:
     * - the cell-local particle ordering
     * - per-cell start/end index ranges
     * - the indirection needed to map local cell ordering to particle indices
     *
     * @param searcher Host-side spatial hashing searcher instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Set the DSMC collision kernel type.
     *
     * This selects the particle collision kernel used by the underlying base
     * solver when resolving the velocity update for each accepted pair.
     *
     * @param kernel_type Desired DSMC collision kernel type.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Build a validated `DsmcDisjointPairSolver` instance.
     *
     * This function validates that all required dependencies have been provided.
     *
     * @return Constructed solver instance.
     *
     * @throws std::runtime_error Thrown when one or more required dependencies
     *         have not been configured.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcDisjointPairSolver<T>
    build() const;

    /**
     * @brief Build a validated solver and wrap it in a host-shared pointer.
     *
     * This function is convenient when the solver is intended to be passed into
     * other host-side runtime systems via shared ownership.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown when one or more required dependencies
     *         have not been configured.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the builder configuration.
     *
     * The builder requires non-null values for:
     * - universe
     * - fluid
     * - searcher
     *
     * @throws std::runtime_error Thrown when any required dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Universe dependency used by the solver.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency used by the solver.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency used by the solver.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Collision kernel type to use when building the solver.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcDisjointPairSolver<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcDisjointPairSolver = atlas::system::DsmcDisjointPairSolver<T>;

/**
 * @brief Host-shared-pointer alias for `DsmcDisjointPairSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcDisjointPairSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `DsmcDisjointPairSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcDisjointPairSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.hpp>