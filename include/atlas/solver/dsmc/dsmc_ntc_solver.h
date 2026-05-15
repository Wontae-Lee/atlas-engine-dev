#pragma once

/**
 * @file dsmc_ntc_solver.h
 * @brief Declares the classic No-Time-Counter DSMC collision solver.
 *
 * @details
 * This file defines @ref atlas::system::DsmcNtcSolver, a Direct Simulation
 * Monte Carlo (DSMC) collision solver that uses a classic
 * No-Time-Counter-style collision stage.
 *
 * DSMC methods represent a gas by simulation particles. During a collision step,
 * particles inside the same spatial cell are considered as possible collision
 * partners, and selected pairs have their velocities updated by a collision
 * model. The collision model itself is represented by the configured
 * `DsmcKernelType`.
 *
 * The NTC formulation separates the idea of:
 *
 * - how many collision attempts should be made in a cell,
 * - which particles are selected as collision partners,
 * - how accepted particle pairs have their velocities updated.
 *
 * In this project, the common DSMC infrastructure is provided by
 * `DsmcSolver<T>`. The base solver prepares cell-local particle ranges,
 * collision-count data, particle-state pointers, material-property pointers, and
 * the selected collision kernel. `DsmcNtcSolver<T>` implements the concrete
 * collision-application stage for the NTC solver variant.
 *
 * The solver relies on three main runtime systems:
 *
 * - a universe, which owns cell topology and per-cell DSMC states,
 * - a fluid, which owns particle states and material/species data,
 * - a spatial hashing searcher, which groups particles into cell-local ranges.
 *
 * @note
 * This header declares the solver interface and documents its runtime behavior.
 * The implementation is provided by `dsmc_ntc_solver.hpp`.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

namespace atlas::system {

/**
 * @brief Classic No-Time-Counter DSMC collision solver.
 *
 * @details
 * `DsmcNtcSolver<T>` derives from `DsmcSolver<T>` and provides a concrete DSMC
 * collision strategy based on the No-Time-Counter approach.
 *
 * In a DSMC collision step, particles are not continuously collided through
 * deterministic force integration. Instead, candidate particle pairs are sampled
 * or selected inside spatial cells, and a binary collision kernel updates the
 * two particle velocities. This makes DSMC suitable for rarefied-gas simulation,
 * where molecular collisions can be modeled statistically.
 *
 * The NTC solver uses per-cell collision scheduling information prepared by the
 * broader DSMC pipeline. For each cell, the collision stage uses:
 *
 * - cell-local particle ordering from the spatial hashing searcher,
 * - per-cell particle counts,
 * - per-cell collision counts,
 * - particle velocity and species arrays,
 * - material/species properties,
 * - the configured DSMC collision kernel.
 *
 * @section dsmc_ntc_solver_runtime_data Runtime data flow
 *
 * The solver does not directly own the main simulation arrays. Instead, it
 * receives a `DsmcSolver<T>::DsmcSolverProbe` from the base solver pipeline.
 * The probe is a compact raw-pointer view over the simulation state and contains
 * data such as:
 *
 * - `velocity_ptr`: writable particle velocity buffer,
 * - `species_ptr`: readable particle species indices,
 * - `properties_ptr`: readable species/material properties,
 * - `number_particle_ptr`: readable per-cell particle counts,
 * - `collision_count_ptr`: readable per-cell requested collision counts,
 * - `indices_ptr`: searcher-generated sorted particle index buffer,
 * - `cell_start_ptr`: start offset of each cell's sorted-particle range,
 * - `cell_end_ptr`: end offset of each cell's sorted-particle range,
 * - `allocated_solver_ptr`: optional per-cell solver assignment buffer,
 * - `kernel`: configured DSMC collision kernel.
 *
 * @section dsmc_ntc_solver_hybrid_filtering Hybrid solver filtering
 *
 * The solver can be used in a hybrid simulation where different cells are
 * assigned to different solvers. If `allocated_solver_ptr` is available in the
 * probe, then this solver processes only cells whose assignment equals the
 * requested solver index:
 *
 * @code
 * probe.allocated_solver_ptr[cell] == index
 * @endcode
 *
 * Cells assigned to another solver are skipped by this collision stage.
 *
 * @section dsmc_ntc_solver_collision_kernel Collision kernel responsibility
 *
 * This class controls the NTC collision-application policy, but the physical
 * velocity update is delegated to the configured DSMC kernel. For each accepted
 * or selected particle pair, the collision kernel receives:
 *
 * @code
 * lhs_velocity
 * rhs_velocity
 * properties[species_i]
 * properties[species_j]
 * @endcode
 *
 * The kernel modifies the two local velocity variables, and the solver writes
 * the updated velocities back to the fluid velocity state.
 *
 * @tparam T Floating-point scalar type used by the solver.
 *
 * @note
 * `DsmcNtcSolver<T>` updates particle velocities during the collision stage.
 * It does not update particle positions.
 *
 * @note
 * The exact pair-selection and acceptance details are implemented in
 * `dsmc_ntc_solver.hpp`. This interface documents the data dependencies and the
 * intended NTC collision-stage role.
 *
 * @see DsmcSolver
 * @see DsmcKernel
 * @see SpatialHashingSearcher
 */
template <typename T>
class DsmcNtcSolver final : public DsmcSolver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `DsmcNtcSolver`.
     *
     * @details
     * The builder collects required runtime dependencies and the optional DSMC
     * kernel type before constructing a validated NTC solver instance.
     */
    class Builder;

public:
    /**
     * @brief Constructs a default-initialized NTC solver.
     *
     * @details
     * The default constructor leaves the solver without configured universe,
     * fluid, or searcher dependencies.
     *
     * A default-constructed solver is mainly useful for containers or delayed
     * initialization. In normal runtime code, prefer the parameterized constructor
     * or the builder so that all required dependencies are supplied explicitly.
     */
    DsmcNtcSolver() = default;

    /**
     * @brief Constructs an NTC DSMC solver with all required runtime dependencies.
     *
     * @details
     * This constructor forwards the universe, fluid, spatial hashing searcher,
     * and collision-kernel choice to the base `DsmcSolver<T>` implementation.
     *
     * The base DSMC solver is responsible for the shared DSMC infrastructure,
     * such as storing the selected kernel and preparing runtime probes. This
     * derived class supplies the NTC-specific collision application step.
     *
     * @param universe Host-side universe containing cell topology and per-cell
     *        DSMC states, including particle counts and collision counts.
     * @param fluid Host-side fluid containing particle velocity, species, and
     *        material-property data used by collision kernels.
     * @param searcher Spatial hashing searcher used to group particles into
     *        cell-local sorted ranges.
     * @param kernel_type Collision kernel type used when resolving selected
     *        particle-pair collisions.
     *
     * @pre @p universe should be non-null for a usable solver.
     * @pre @p fluid should be non-null for a usable solver.
     * @pre @p searcher should be non-null for a usable solver.
     *
     * @post The base DSMC solver is initialized with the configured dependencies
     *       and collision kernel type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcNtcSolver(UniverseHostPtr<T> universe,
                  FluidHostPtr<T> fluid,
                  SpatialHashingSearcherHostPtr<T> searcher,
                  DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    /**
     * @brief Destroys the solver.
     *
     * @details
     * Uses default destruction. Base-class resources, shared pointers, and any
     * owned buffers are released through their normal RAII behavior.
     */
    ~DsmcNtcSolver() override = default;

    /**
     * @brief Creates a builder for `DsmcNtcSolver`.
     *
     * @details
     * The returned builder can be used to configure the universe, fluid, spatial
     * searcher, and collision kernel type before constructing the solver.
     *
     * @return Default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Applies NTC-based DSMC collisions to the assigned cell subset.
     *
     * @details
     * This function implements the collision-application stage for the classic
     * NTC DSMC solver.
     *
     * The function receives a `DsmcSolverProbe` prepared by the base solver. The
     * probe contains raw pointers to all device data needed for collision
     * processing. The solver then processes eligible cells and applies the
     * configured DSMC collision kernel to selected particle pairs.
     *
     * Conceptually, the function performs the following operations:
     *
     * 1. Iterate over simulation cells.
     * 2. If solver-allocation data is available, skip cells whose allocation does
     *    not match @p index.
     * 3. Read the cell-local particle range from the spatial hashing searcher.
     * 4. Read the requested number of collision attempts from the universe
     *    collision-count state.
     * 5. Select or evaluate particle pairs according to the NTC collision rule.
     * 6. Validate particle indices and species/material indices.
     * 7. Apply the configured DSMC collision kernel to each valid pair.
     * 8. Write updated velocities back to the fluid velocity buffer.
     *
     * The function operates in place on particle velocity storage. It does not
     * allocate new particle buffers and does not modify particle positions.
     *
     * @param probe Raw-pointer runtime data prepared by `DsmcSolver<T>::make_probe`.
     *        The probe must contain valid pointers for particle velocities,
     *        particle species, material properties, cell ranges, particle counts,
     *        collision counts, and the configured collision kernel.
     * @param index Solver index used to filter cells when
     *        `probe.allocated_solver_ptr` is non-null.
     * @param dt Time step associated with the current collision update. The NTC
     *        collision count is normally prepared by the surrounding DSMC pipeline;
     *        this function accepts @p dt through the virtual solver interface.
     *
     * @pre `probe.velocity_ptr` must be readable and writable.
     * @pre `probe.species_ptr` must be readable.
     * @pre `probe.properties_ptr` must be readable.
     * @pre `probe.number_particle_ptr` must be readable.
     * @pre `probe.collision_count_ptr` must be readable.
     * @pre `probe.cell_start_ptr`, `probe.cell_end_ptr`, and `probe.indices_ptr`
     *      must describe valid searcher-generated cell-local particle ranges.
     *
     * @post Valid selected collision pairs have updated particle velocities.
     * @post Cells not assigned to @p index are unchanged when solver filtering is
     *       enabled.
     * @post Invalid particle pairs or invalid species references are skipped.
     *
     * @note
     * This function is called by the base DSMC solve pipeline. Users normally call
     * the solver's public solve entry point rather than invoking this method
     * directly.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const typename DsmcSolver<T>::DsmcSolverProbe& probe,
                     int index,
                     T dt) override;
};

/**
 * @brief Builder for `DsmcNtcSolver`.
 *
 * @details
 * This builder collects all runtime dependencies required to construct a valid
 * NTC DSMC solver.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * Optional configuration:
 *
 * - DSMC collision kernel type.
 *
 * The default collision kernel type is `DsmcKernelType::hard_sphere`.
 *
 * Use the builder when construction should fail early if any required dependency
 * is missing. The direct constructor accepts the provided shared pointers as-is,
 * while the builder validates them before construction.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcNtcSolver<T>::Builder final {
public:
    /**
     * @brief Constructs a default-initialized builder.
     *
     * @details
     * Required dependencies are initialized to null. The collision kernel type is
     * initialized to `DsmcKernelType::hard_sphere`.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @details
     * The universe provides the cell topology and cell-level DSMC states used by
     * the solver pipeline.
     *
     * Important universe-side data includes:
     *
     * - number of cells,
     * - per-cell particle counts,
     * - per-cell collision counts,
     * - optional per-cell solver allocation data in hybrid workflows.
     *
     * @param universe Host-side universe instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p universe as the universe dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @details
     * The fluid provides all particle-level data required by collision processing.
     *
     * Important fluid-side data includes:
     *
     * - particle velocities,
     * - particle species indices,
     * - species/material properties used by the collision kernel,
     * - global particle-count metadata.
     *
     * The NTC collision stage updates particle velocities in place.
     *
     * @param fluid Host-side fluid instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p fluid as the fluid dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @details
     * The spatial hashing searcher maps particles to simulation cells and
     * provides the cell-local indexing used during collision processing.
     *
     * The solver relies on searcher-generated buffers such as:
     *
     * - sorted particle indices,
     * - per-cell start offsets,
     * - per-cell end offsets.
     *
     * These buffers allow the collision solver to visit particles cell by cell.
     *
     * @param searcher Host-side spatial hashing searcher instance.
     * @return Reference to this builder.
     *
     * @post The builder stores @p searcher as the searcher dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the collision kernel type used by the solver.
     *
     * @details
     * This selects the concrete DSMC collision model used to update particle-pair
     * velocities.
     *
     * Pair selection and scheduling are handled by the DSMC solver pipeline and
     * this NTC solver. The physical velocity transformation for each accepted pair
     * is handled by the collision kernel selected here.
     *
     * @param kernel_type Desired DSMC collision kernel type.
     * @return Reference to this builder.
     *
     * @post The builder stores @p kernel_type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Builds a validated `DsmcNtcSolver` instance.
     *
     * @details
     * This function validates that all required dependencies have been configured,
     * then constructs and returns a solver value.
     *
     * Validation requires:
     *
     * - universe is non-null,
     * - fluid is non-null,
     * - searcher is non-null.
     *
     * The kernel type is always available because it has a default value.
     *
     * @return Constructed solver instance.
     *
     * @throws std::runtime_error Thrown if the universe dependency is missing.
     * @throws std::runtime_error Thrown if the fluid dependency is missing.
     * @throws std::runtime_error Thrown if the searcher dependency is missing.
     *
     * @post The returned solver is initialized with the configured dependencies
     *       and collision kernel type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DsmcNtcSolver<T>
    build() const;

    /**
     * @brief Builds a validated solver and wraps it in a host-shared pointer.
     *
     * @details
     * This function performs the same validation as `build()`, but returns the
     * constructed solver as an `atlas::host_shared_ptr`.
     *
     * Use this function when the solver will be shared by host-side runtime
     * systems or stored in containers that expect Atlas host shared pointers.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown if the universe dependency is missing.
     * @throws std::runtime_error Thrown if the fluid dependency is missing.
     * @throws std::runtime_error Thrown if the searcher dependency is missing.
     *
     * @post The returned solver is initialized with the configured dependencies
     *       and collision kernel type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcNtcSolver<T>>
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
     * The universe supplies cell topology and per-cell DSMC states through the
     * base solver probe.
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
     * The searcher supplies sorted particle indices and cell-local ranges used by
     * the DSMC collision stage.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Collision kernel type selected for the solver.
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
 * @brief Convenience alias for `atlas::system::DsmcNtcSolver<T>`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcNtcSolver = atlas::system::DsmcNtcSolver<T>;

/**
 * @brief Host-shared-pointer alias for `DsmcNtcSolver`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcNtcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `DsmcNtcSolver`.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
using DsmcNtcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcNtcSolver<T>>;

} // namespace atlas

#include <atlas/solver/dsmc/dsmc_ntc_solver.hpp>