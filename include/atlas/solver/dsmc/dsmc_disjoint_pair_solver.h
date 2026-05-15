#pragma once

/**
 * @file dsmc_disjoint_pair_solver.h
 * @brief Declares a DSMC collision solver that processes disjoint particle pairs in each cell.
 *
 * @details
 * This file defines @ref atlas::system::DsmcDisjointPairSolver, a specialized
 * Direct Simulation Monte Carlo (DSMC) collision solver.
 *
 * DSMC solvers update particle velocities by applying binary collision models to
 * selected pairs of particles. In this implementation, particles are first
 * grouped by spatial search cells through a spatial hashing searcher. Each cell
 * owns a local sorted-particle range, and collision pairs are selected from that
 * local ordering.
 *
 * Unlike more general DSMC collision-pair selection strategies, this solver uses
 * a deterministic disjoint-pair rule. Inside each cell, valid local particles are
 * paired as:
 *
 * @code
 * local particle 0 with local particle 1
 * local particle 2 with local particle 3
 * local particle 4 with local particle 5
 * ...
 * @endcode
 *
 * Therefore, during one collision pass, a particle can appear in at most one
 * candidate pair. This is the meaning of "disjoint pair" in this solver.
 *
 * The number of actually processed pairs is limited by two quantities:
 *
 * - the number of available disjoint local pairs, `floor(number_of_particles / 2)`,
 * - the requested collision count stored in `UniverseCollisionCountState<int>`.
 *
 * The solver processes:
 *
 * @code
 * min(floor(number_of_particles / 2), collision_count[cell])
 * @endcode
 *
 * pairs in each eligible cell.
 *
 * @note
 * This solver does not randomly sample arbitrary particle pairs. Pairing follows
 * the cell-local particle ordering provided by the spatial hashing searcher.
 *
 * @note
 * The actual velocity update for a pair is delegated to the configured DSMC
 * collision kernel stored in the base `DsmcSolver<T>` probe.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief DSMC solver that applies collisions only to disjoint local particle pairs.
 *
 * @details
 * `DsmcDisjointPairSolver<T>` derives from `DsmcSolver<T>` and implements a
 * restricted binary collision strategy.
 *
 * In DSMC, a collision solver typically chooses pairs of particles and updates
 * their velocities according to a collision kernel. This class keeps that basic
 * structure, but restricts pair selection to non-overlapping local pairs inside
 * each search cell.
 *
 * For each cell, the solver uses the searcher-provided sorted particle range:
 *
 * @code
 * begin = probe.cell_start_ptr[cell];
 * end   = probe.cell_end_ptr[cell];
 * @endcode
 *
 * A local pair index is converted into actual particle indices using
 * `DsmcSolver<T>::nth_valid_particle()`. This helper skips invalid entries in
 * the sorted index buffer and returns the N-th valid particle inside the cell
 * range.
 *
 * For a cell with valid local particles:
 *
 * @code
 * p0, p1, p2, p3, p4, p5, ...
 * @endcode
 *
 * this solver attempts collisions between:
 *
 * @code
 * (p0, p1), (p2, p3), (p4, p5), ...
 * @endcode
 *
 * This guarantees that each valid particle is used at most once during this
 * solver pass.
 *
 * @section dsmc_disjoint_pair_solver_cell_filtering Cell filtering
 *
 * The solver can be used as part of a hybrid solver pipeline. If the
 * `DsmcSolverProbe` contains `allocated_solver_ptr`, then a cell is processed
 * only when:
 *
 * @code
 * probe.allocated_solver_ptr[cell] == index
 * @endcode
 *
 * Cells assigned to a different solver index are skipped completely.
 *
 * @section dsmc_disjoint_pair_solver_collision_limit Collision-count limiting
 *
 * The universe collision-count state controls how many collisions are requested
 * in each cell. This solver never processes more pairs than are available from
 * the disjoint-pair rule.
 *
 * For each eligible cell:
 *
 * @code
 * count               = probe.number_particle_ptr[cell]
 * disjoint_pair_count = count / 2
 * requested           = probe.collision_count_ptr[cell]
 * collision_limit     = min(requested, disjoint_pair_count)
 * @endcode
 *
 * The solver then processes local pair IDs:
 *
 * @code
 * 0, 1, 2, ..., collision_limit - 1
 * @endcode
 *
 * where each local collision ID maps to:
 *
 * @code
 * lhs_local = local_collision * 2
 * rhs_local = lhs_local + 1
 * @endcode
 *
 * @section dsmc_disjoint_pair_solver_velocity_update Velocity update
 *
 * For each valid pair, the solver:
 *
 * 1. reads the two particle species,
 * 2. verifies that both species indices refer to valid material properties,
 * 3. copies both particle velocities into local variables,
 * 4. applies the configured DSMC collision kernel,
 * 5. writes the updated velocities back to the fluid velocity state.
 *
 * The collision kernel receives:
 *
 * @code
 * lhs_velocity
 * rhs_velocity
 * properties[species_i]
 * properties[species_j]
 * @endcode
 *
 * and is responsible for applying the physical collision model.
 *
 * @tparam T Floating-point scalar type used by the solver.
 *
 * @note
 * This class only defines the pair-selection policy. Collision physics are
 * provided by the DSMC kernel configured through the base solver.
 *
 * @note
 * The solver updates particle velocities only. Particle positions are not
 * modified by `apply_collisions()`.
 *
 * @see DsmcSolver
 * @see DsmcKernel
 * @see SpatialHashingSearcher
 */
template <typename T>
class DsmcDisjointPairSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `DsmcDisjointPairSolver`.
     *
     * @details
     * The builder collects the required simulation dependencies and optional DSMC
     * kernel type before constructing a validated solver instance.
     */
    class Builder;

public:
    /**
     * @brief Constructs a default-initialized solver.
     *
     * @details
     * The default constructor leaves the solver without configured universe,
     * fluid, or searcher dependencies.
     *
     * A solver created this way is not ready to perform collision work until the
     * required base solver dependencies are configured by another path.
     *
     * Prefer using either the parameterized constructor or the builder for normal
     * runtime construction.
     */
    DsmcDisjointPairSolver() = default;

    /**
     * @brief Constructs a disjoint-pair DSMC solver with all required dependencies.
     *
     * @details
     * This constructor forwards the universe, fluid, spatial searcher, and kernel
     * type to the base `DsmcSolver<T>` constructor.
     *
     * The base solver owns the common DSMC runtime context, including the selected
     * collision kernel and the probe-construction logic used by this derived
     * solver.
     *
     * @param universe Host-side universe containing the cell topology and per-cell
     *        DSMC control states, including particle counts and collision counts.
     * @param fluid Host-side fluid containing particle velocity, species, and
     *        material-property data.
     * @param searcher Spatial hashing searcher that provides cell-local sorted
     *        particle ranges and the indirection from local sorted order to global
     *        particle indices.
     * @param kernel_type Collision kernel type used when resolving each accepted
     *        particle-pair collision.
     *
     * @pre @p universe should be non-null for a usable solver.
     * @pre @p fluid should be non-null for a usable solver.
     * @pre @p searcher should be non-null for a usable solver.
     *
     * @post The base DSMC solver is initialized with the selected collision kernel.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcDisjointPairSolver(UniverseHostPtr<T> universe,
                           FluidHostPtr<T> fluid,
                           SpatialHashingSearcherHostPtr<T> searcher,
                           DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroys the solver.
     *
     * @details
     * Uses default destruction. Shared pointers, buffers, and base-class resources
     * are released according to their own RAII semantics.
     */
    ~DsmcDisjointPairSolver() override = default;

    /**
     * @brief Creates a builder for `DsmcDisjointPairSolver`.
     *
     * @details
     * The returned builder can be used to configure the universe, fluid, searcher,
     * and DSMC collision kernel type before constructing the solver.
     *
     * @return Default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Applies DSMC collisions to disjoint local particle pairs.
     *
     * @details
     * This function is the core collision stage of the disjoint-pair DSMC solver.
     * It is called by the base DSMC solve pipeline after a `DsmcSolverProbe` has
     * been built.
     *
     * The probe provides raw device pointers to:
     *
     * - particle velocities,
     * - particle species,
     * - material properties,
     * - per-cell particle counts,
     * - per-cell requested collision counts,
     * - sorted particle indices,
     * - per-cell sorted-index ranges,
     * - optional per-cell solver allocation data,
     * - the configured DSMC collision kernel.
     *
     * For each cell, the function first applies optional solver filtering:
     *
     * @code
     * if allocated_solver exists and allocated_solver[cell] != index:
     *     skip cell
     * @endcode
     *
     * For each selected cell, it computes:
     *
     * @code
     * count               = number of particles recorded for this cell
     * disjoint_pair_count = count / 2
     * collisions          = requested collision count for this cell
     * collision_limit     = min(collisions, disjoint_pair_count)
     * @endcode
     *
     * It then visits local collision IDs from `0` to `collision_limit - 1`.
     * Each local collision ID selects one disjoint pair:
     *
     * @code
     * lhs_local = local_collision * 2
     * rhs_local = lhs_local + 1
     * @endcode
     *
     * The local indices are converted to actual particle indices through
     * `DsmcSolver<T>::nth_valid_particle()`.
     *
     * Invalid pairs are skipped. A pair is invalid when either particle index
     * cannot be resolved, or when either particle species is outside the material
     * property array.
     *
     * For a valid pair, the current particle velocities are copied into local
     * variables, passed to the collision kernel together with both species
     * properties, and then written back to the velocity buffer.
     *
     * @param probe Raw-pointer runtime data prepared by `DsmcSolver<T>::make_probe`.
     *        The probe must contain all particle, cell, property, and kernel data
     *        required for DSMC collision processing.
     * @param index Solver index used to filter cells when
     *        `probe.allocated_solver_ptr` is non-null.
     * @param dt Time-step size associated with the current update. This parameter
     *        is accepted for interface compatibility with `DsmcSolver<T>`, but the
     *        current disjoint-pair implementation does not use it directly.
     *
     * @pre `probe.velocity_ptr` must be readable and writable.
     * @pre `probe.species_ptr` must be readable.
     * @pre `probe.properties_ptr` must be readable.
     * @pre `probe.number_particle_ptr` must be readable.
     * @pre `probe.collision_count_ptr` must be readable.
     * @pre `probe.cell_start_ptr`, `probe.cell_end_ptr`, and `probe.indices_ptr`
     *      must describe valid searcher-generated cell ranges.
     *
     * @post Valid selected collision pairs have updated particle velocities.
     * @post Particles in skipped cells are unchanged by this function.
     * @post Invalid pairs are ignored without modifying their velocities.
     *
     * @note
     * Each particle can participate in at most one pair in a single call because
     * the local pair sequence is disjoint.
     *
     * @note
     * The number of processed pairs may be smaller than the requested collision
     * count if the cell does not contain enough disjoint valid pairs.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const typename DsmcSolver<T>::DsmcSolverProbe& probe,
                     int index,
                     T dt) override;
};

/**
 * @brief Builder for `DsmcDisjointPairSolver`.
 *
 * @details
 * This builder collects all information required to construct a valid
 * disjoint-pair DSMC solver.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * Optional configuration:
 *
 * - collision kernel type.
 *
 * The default collision kernel type is `DsmcKernelType::hard_sphere`.
 *
 * Use the builder when construction should fail early if any required dependency
 * is missing. The direct constructor accepts the provided shared pointers as-is,
 * while the builder validates them before constructing the solver.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcDisjointPairSolver<T>::Builder final {
public:
    /**
     * @brief Constructs a default-initialized builder.
     *
     * @details
     * Required dependencies are initialized to null. The kernel type defaults to
     * `DsmcKernelType::hard_sphere`.
     */
    Builder() = default;

    /**
     * @brief Sets the universe used by the solver.
     *
     * @details
     * The universe provides the cell topology and per-cell DSMC states used by
     * the solver pipeline.
     *
     * In particular, the disjoint-pair collision stage relies on universe-derived
     * per-cell fields such as:
     *
     * - particle count per cell,
     * - requested collision count per cell,
     * - optional solver allocation per cell.
     *
     * @param universe Host-side universe instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p universe as the universe dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid used by the solver.
     *
     * @details
     * The fluid provides the particle data required by collision processing:
     *
     * - particle velocities,
     * - particle species indices,
     * - species/material properties.
     *
     * The collision kernel reads the material properties associated with the two
     * particles in each accepted pair.
     *
     * @param fluid Host-side fluid instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p fluid as the fluid dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher used by the solver.
     *
     * @details
     * The spatial hashing searcher provides the local particle ordering used to
     * form disjoint pairs.
     *
     * The solver reads from:
     *
     * - sorted particle index buffer,
     * - cell start offsets,
     * - cell end offsets.
     *
     * These buffers define the cell-local ordering from which local pairs
     * `(0, 1)`, `(2, 3)`, `(4, 5)`, and so on are formed.
     *
     * @param searcher Host-side spatial hashing searcher instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p searcher as the searcher dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the DSMC collision kernel type.
     *
     * @details
     * This selects the collision model used to update velocities for each valid
     * particle pair.
     *
     * The pair-selection rule is implemented by `DsmcDisjointPairSolver<T>`, but
     * the actual velocity update is delegated to the DSMC kernel configured here.
     *
     * @param kernel_type Desired DSMC collision kernel type.
     * @return Reference to this builder.
     *
     * @post The builder stores @p kernel_type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Builds a validated `DsmcDisjointPairSolver` instance.
     *
     * @details
     * This function validates that all required dependencies have been provided,
     * then constructs and returns a solver value.
     *
     * Validation requires:
     *
     * - universe is non-null,
     * - fluid is non-null,
     * - searcher is non-null.
     *
     * @return Constructed solver instance.
     *
     * @throws std::runtime_error Thrown if the universe dependency is missing.
     * @throws std::runtime_error Thrown if the fluid dependency is missing.
     * @throws std::runtime_error Thrown if the searcher dependency is missing.
     *
     * @post The returned solver is initialized with the configured dependencies
     *       and kernel type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcDisjointPairSolver<T>
    build() const;

    /**
     * @brief Builds a validated solver and wraps it in a host-shared pointer.
     *
     * @details
     * This function performs the same validation as `build()`, but returns the
     * constructed solver as an `atlas::host_shared_ptr`.
     *
     * Use this function when the solver will be shared by host-side simulation
     * systems or stored in containers that expect Atlas host shared pointers.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown if the universe dependency is missing.
     * @throws std::runtime_error Thrown if the fluid dependency is missing.
     * @throws std::runtime_error Thrown if the searcher dependency is missing.
     *
     * @post The returned solver is initialized with the configured dependencies
     *       and kernel type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder configuration.
     *
     * @details
     * The builder requires non-null values for all runtime dependencies needed by
     * the DSMC solver:
     *
     * - universe,
     * - fluid,
     * - spatial hashing searcher.
     *
     * The kernel type does not require validation because it is always initialized
     * to a valid default value.
     *
     * @throws std::runtime_error Thrown if the universe dependency is missing.
     * @throws std::runtime_error Thrown if the fluid dependency is missing.
     * @throws std::runtime_error Thrown if the searcher dependency is missing.
     *
     * @post No builder state is modified.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Universe dependency used by the solver.
     *
     * @details
     * The universe supplies cell topology and per-cell collision-control states
     * through the base DSMC solver probe.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency used by the solver.
     *
     * @details
     * The fluid supplies particle velocity, species, and material-property data
     * used by the collision kernel.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial hashing searcher dependency used by the solver.
     *
     * @details
     * The searcher supplies sorted particle indices and cell-local ranges. These
     * ranges define the local ordering used to form disjoint pairs.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Collision kernel type used when building the solver.
     *
     * @details
     * This value selects the DSMC collision model used by the base solver. The
     * default is `DsmcKernelType::hard_sphere`.
     */
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::DsmcDisjointPairSolver<T>`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcDisjointPairSolver = atlas::system::DsmcDisjointPairSolver<T>;

/**
 * @brief Host-shared-pointer alias for `DsmcDisjointPairSolver`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcDisjointPairSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `DsmcDisjointPairSolver`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcDisjointPairSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcDisjointPairSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.hpp>