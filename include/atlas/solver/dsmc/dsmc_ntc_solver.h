#pragma once

/**
 * @file dsmc_ntc_solver.h
 * @brief Declares the classic NTC DSMC collision solver.
 *
 * This header defines `DsmcNtcSolver<T>`, a DSMC solver implementation based on
 * the classic NTC (No-Time-Counter) collision formulation.
 *
 * The solver is built on top of `DsmcSolver<T>` and relies on:
 * - a universe that stores cell topology and per-cell collision-related states
 * - a fluid that stores particle states and species/material properties
 * - a spatial hashing searcher that provides cell-local particle grouping
 *
 * The concrete collision model applied to each accepted pair is selected through
 * `DsmcKernelType`.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief Classic No-Time-Counter DSMC solver.
 *
 * `DsmcNtcSolver<T>` implements a DSMC collision step using the NTC strategy,
 * where the number of collision attempts per cell is determined independently
 * from the actual pair loop and then resolved through the selected DSMC kernel.
 *
 * Compared with simpler deterministic local-pair schemes, the NTC method is
 * intended to approximate stochastic collision sampling in a cell while still
 * using the shared universe / fluid / searcher infrastructure provided by
 * `DsmcSolver<T>`.
 *
 * The solver itself does not own the simulation state buffers directly. Instead,
 * it operates on:
 * - universe states for cell-level collision statistics
 * - fluid states for particle velocities, species, and other particle data
 * - searcher-produced cell-local particle indexing tables
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcNtcSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `DsmcNtcSolver`.
     */
    class Builder;

public:
    /**
     * @brief Construct a default-initialized NTC solver.
     *
     * This constructor leaves the solver unconfigured. In normal usage, callers
     * should prefer the parameterized constructor or the builder interface.
     */
    DsmcNtcSolver() = default;

    /**
     * @brief Construct an NTC DSMC solver with all required runtime dependencies.
     *
     * This forwards the universe, fluid, searcher, and collision-kernel choice
     * to the base `DsmcSolver<T>` implementation.
     *
     * @param universe Host-side universe containing cell topology and per-cell states.
     * @param fluid Host-side fluid containing particle states and material properties.
     * @param searcher Spatial hashing searcher used to locate particles per cell.
     * @param kernel_type Collision kernel type used when resolving accepted collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcNtcSolver(UniverseHostPtr<T> universe,
                  FluidHostPtr<T> fluid,
                  SpatialHashingSearcherHostPtr<T> searcher,
                  DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroy the solver.
     */
    ~DsmcNtcSolver() override = default;

    /**
     * @brief Create a builder for `DsmcNtcSolver`.
     *
     * @return Default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Apply NTC-based collisions to the currently assigned cell subset.
     *
     * This function performs the collision step for the classic NTC solver.
     * The exact implementation details are provided in the accompanying `.hpp`
     * file, but conceptually the function:
     * - reads cell-local particle information from the searcher
     * - reads per-cell collision scheduling data from the universe
     * - selects or evaluates collision candidates according to the NTC scheme
     * - applies the selected DSMC kernel to accepted particle pairs
     *
     * If `allocated_solver` is not null, only cells whose solver-assignment entry
     * equals `index` are processed by this invocation. This allows work partitioning
     * across multiple solver instances or passes.
     *
     * The function operates in place on the existing particle-state storage. It
     * does not replace the fluid or universe buffers with new allocations.
     *
     * @param allocated_solver Optional device buffer storing the solver assignment
     *        for each cell. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step associated with the current collision update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;
};

/**
 * @brief Builder for `DsmcNtcSolver`.
 *
 * The builder collects all runtime dependencies required to construct a valid
 * solver instance. A solver cannot be built unless:
 * - universe is set
 * - fluid is set
 * - searcher is set
 *
 * The collision kernel type is optional because it has a default value of
 * `DsmcKernelType::hard_sphere`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcNtcSolver<T>::Builder final {
public:
    /**
     * @brief Construct a default-initialized builder.
     */
    Builder() = default;

    /**
     * @brief Set the universe dependency.
     *
     * The universe provides:
     * - cell topology
     * - cell count
     * - per-cell collision-related states used by the solver
     *
     * @param universe Host-side universe instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Set the fluid dependency.
     *
     * The fluid provides:
     * - particle states such as velocity and species
     * - particle/material properties required by the collision kernel
     * - particle-count metadata used during solver execution
     *
     * @param fluid Host-side fluid instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the spatial hashing searcher dependency.
     *
     * The searcher provides:
     * - sorted particle indices
     * - per-cell particle start/end ranges
     * - cell-local particle ordering used during collision processing
     *
     * @param searcher Host-side spatial hashing searcher instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Set the collision kernel type used by the solver.
     *
     * This selects which concrete DSMC collision model is applied when the solver
     * resolves accepted collision pairs.
     *
     * @param kernel_type Desired DSMC collision kernel type.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Build a validated `DsmcNtcSolver` instance.
     *
     * This function validates that all required dependencies have been configured
     * before constructing the solver.
     *
     * @return Constructed solver instance.
     *
     * @throws std::runtime_error Thrown when one or more required dependencies
     *         have not been set.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcNtcSolver<T>
    build() const;

    /**
     * @brief Build a validated solver and wrap it in a host-shared pointer.
     *
     * This is convenient when the solver will be inserted into higher-level runtime
     * systems that use shared ownership.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown when one or more required dependencies
     *         have not been set.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcNtcSolver<T>>
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
     * @brief Collision kernel type selected for the solver.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcNtcSolver<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcNtcSolver = atlas::system::DsmcNtcSolver<T>;

/**
 * @brief Host-shared-pointer alias for `DsmcNtcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcNtcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `DsmcNtcSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using DsmcNtcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_ntc_solver.hpp>