#pragma once

/**
 * @file sph_gateway_solver.h
 * @brief Declares a gateway SPH solver that operates on deterministic per-cell particle groups.
 *
 * This header defines `SphGatewaySolver<T>`, an SPH solver variant that does not
 * directly evaluate particle-particle interactions for every particle in a cell.
 * Instead, it introduces a grouped intermediate representation:
 *
 * 1. particles in each cell are partitioned deterministically into fixed-size groups
 * 2. each group is represented by an aggregated position, velocity, mass, and species
 * 3. SPH density / pressure / motion updates are evaluated between group representatives
 * 4. the updated representative state is scattered back to all particles in the group
 *
 * This grouped formulation is useful when:
 * - a more compact or codec-aware execution path is desired
 * - the solver should operate on a reduced representative set instead of all particles
 * - deterministic grouping is preferred over stochastic local approximation
 *
 * The solver still depends on the standard SPH runtime inputs:
 * - a universe defining the simulation domain and grid
 * - a fluid containing particle states and material properties
 * - a spatial hashing searcher providing cell-local particle ordering
 */

#include <atlas/core/macros.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>

namespace atlas::system {

/**
 * @brief SPH solver variant that aggregates particles into fixed-size groups inside each cell.
 *
 * This solver replaces direct particle-level SPH interaction inside a cell with
 * a two-stage grouped approximation:
 *
 * - first, particles in each selected cell are partitioned into deterministic groups
 * - second, SPH interaction is evaluated between group representatives
 *
 * The group representatives act as reduced-order proxies for the particles in the
 * cell. After representative-level density, pressure, and motion have been updated,
 * the solver propagates those results back to the underlying particles.
 *
 * High-level grouped solve pipeline:
 * - ensure required universe-side field buffers exist
 * - reset temporary universe-side field values
 * - update the per-cell particle counts
 * - build representative groups
 * - estimate representative density and pressure
 * - update representative motion
 * - scatter representative states back to particles
 *
 * The grouped solve path is primarily exposed through:
 * - `solve(const DeviceBuffer<int>* allocated_solver, int index, T dt)`
 *
 * The simpler overload:
 * - `solve(T dt)`
 * simply applies the grouped path to all cells.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class SphGatewaySolver final : public Solver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `SphGatewaySolver`.
     */
    class Builder;

public:
    /**
     * @brief Construct a default-initialized solver.
     *
     * A default-constructed solver is not yet fully configured for use in a real
     * runtime pipeline. In normal usage, the parameterized constructor or builder
     * should be preferred.
     */
    SphGatewaySolver() = default;

    /**
     * @brief Construct a grouped SPH solver with all required runtime dependencies.
     *
     * The solver stores shared references to the universe, fluid, and searcher,
     * selects an SPH kernel model, and records the fixed particle count used when
     * partitioning particles into deterministic groups inside each cell.
     *
     * @param universe Host-side universe describing the simulation domain and grid.
     * @param fluid Host-side fluid containing particle states and material properties.
     * @param searcher Spatial hashing searcher providing cell-local particle ordering.
     * @param kernel_type SPH kernel type used for representative interaction.
     * @param group_particle_count Fixed number of particles targeted per group.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphGatewaySolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher,
                     SphKernelType kernel_type = SphKernelType::standard,
                     int group_particle_count  = 5) noexcept;

    /**
     * @brief Destroy the solver.
     */
    ~SphGatewaySolver() override = default;

    /**
     * @brief Create a builder for `SphGatewaySolver`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return the configured SPH kernel type.
     *
     * @return Active SPH kernel type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    /**
     * @brief Return the configured fixed particle count per group.
     *
     * This value controls how particles are partitioned into deterministic groups
     * inside each cell. The final group in a cell may contain fewer particles if
     * the cell particle count is not divisible by this value.
     *
     * @return Target particle count per group.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    group_particle_count() const noexcept;

    /**
     * @brief Solve the grouped SPH update for all cells.
     *
     * This overload runs the grouped representative path without any solver-index
     * filtering. It is equivalent to applying the grouped solve procedure over the
     * full cell set owned by the current universe/searcher configuration.
     *
     * @param dt Time step used for representative motion update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override final;

    /**
     * @brief Solve the grouped SPH update for a filtered subset of cells.
     *
     * When `allocated_solver` is not null, only cells whose solver assignment
     * matches `index` are processed. This enables external work partitioning or
     * codec-aware scheduling across multiple solver passes.
     *
     * The grouped solve pipeline for selected cells is typically:
     * - initialize context and required fields
     * - update cell particle counts
     * - build group representatives
     * - estimate density and pressure for groups
     * - update representative motion
     * - scatter representative results back to particles
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step used for representative motion update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override final;

    /**
     * @brief Ensure that all required universe-side field states exist.
     *
     * The grouped solver needs a set of universe-side fields to store intermediate
     * or diagnostic quantities used during grouped SPH evaluation. This function
     * creates or prepares those fields before the main grouped solve stages run.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Initialize the grouped SPH execution context.
     *
     * This function prepares all shared state required before grouped processing.
     * Typical responsibilities include:
     * - ensuring required universe states exist
     * - allocating or resizing internal group buffers
     * - resetting universe-side transient fields
     *
     * @return True when initialization succeeded and the grouped solve may continue.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    /**
     * @brief Prepare internal group-field buffers.
     *
     * The grouped solver maintains several device buffers describing group-level
     * representative data such as:
     * - group position
     * - group velocity
     * - updated group position
     * - updated group velocity
     * - group mass
     * - group density
     * - group pressure
     * - group member count
     * - group species
     *
     * This function prepares those buffers so later grouped stages can safely read
     * and write representative-level state.
     *
     * @return True when the group-field buffers are ready for use.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_group_fields();

    /**
     * @brief Reset universe-side temporary SPH fields.
     *
     * This function clears or reinitializes transient universe-side data used during
     * grouped SPH evaluation so that each solve step starts from a clean state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    /**
     * @brief Update per-cell particle counts for the selected cells.
     *
     * This stage refreshes the number of particles stored per cell, which is then
     * used to determine how many deterministic groups must be created in each cell.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Build deterministic group representatives for the selected cells.
     *
     * In each eligible cell, particles are partitioned into groups of size
     * `group_particle_count()` (except possibly the final remainder group).
     *
     * For each group, representative fields are built, typically including:
     * - mean position
     * - mean velocity
     * - total or representative mass
     * - member count
     * - species label
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_group_representatives(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Estimate SPH density and pressure using group representatives.
     *
     * This stage evaluates representative-level SPH interaction instead of direct
     * particle-level interaction. Neighbor searches and kernel evaluations are
     * performed between group representatives, and the resulting density/pressure
     * values are stored in the group buffers.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Update representative group motion over one time step.
     *
     * This stage uses group-level density, pressure, velocity, and other relevant
     * quantities to compute updated representative position and velocity values.
     *
     * The updated states are written into the dedicated "updated group" buffers
     * rather than immediately overwriting the original representative state.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     * @param dt Time step used for motion integration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Scatter updated representative states back to particle storage.
     *
     * After group-level motion has been computed, this stage propagates the updated
     * representative state back to all particles belonging to each group.
     *
     * This is the final stage that reconnects the reduced group representation to
     * the actual per-particle simulation state.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer. May be null.
     * @param index Solver index used to filter cells when `allocated_solver` is provided.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Return the smoothing length used for a material property entry.
     *
     * This helper derives the SPH smoothing length associated with a particle
     * material/property record. If the material does not explicitly provide a
     * smoothing length, the cell size may be used as a fallback scale.
     *
     * @param property Particle/material property record.
     * @param cell_size Universe cell size used as a fallback reference scale.
     * @return Effective smoothing length.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    smoothing_length_for(const MatrialProperties<T>& property, T cell_size) noexcept;

    /**
     * @brief Return the rest density used for a material property entry.
     *
     * @param property Particle/material property record.
     * @return Effective rest density.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density_for(const MatrialProperties<T>& property) noexcept;

    /**
     * @brief Return the pressure coefficient used for a material property entry.
     *
     * @param property Particle/material property record.
     * @return Effective pressure coefficient.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MatrialProperties<T>& property) noexcept;

    /**
     * @brief Return the integer search radius, in cells, required for a smoothing length.
     *
     * This helper converts a physical smoothing length into a neighborhood radius
     * measured in grid cells, which can be used when scanning neighboring cells
     * around a representative group.
     *
     * @param smoothing_length Effective smoothing length.
     * @param cell_size Universe cell size.
     * @return Neighborhood search radius measured in cells.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    search_radius_for(T smoothing_length, T cell_size) noexcept;

    /**
     * @brief Map a particle position to its integer grid cell coordinate.
     *
     * @param position Particle or representative position.
     * @param lower_corner Lower corner of the universe domain.
     * @param inverse_cell_size Inverse of the universe cell size.
     * @param grid_size Integer grid resolution of the universe.
     * @return Grid-cell coordinate containing the position.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<int>
    particle_cell(const Vector3<T>& position,
                  const Vector3<T>& lower_corner,
                  T inverse_cell_size,
                  const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Check whether a grid-cell coordinate lies inside the valid grid domain.
     *
     * @param cell Candidate neighbor cell coordinate.
     * @param grid_size Integer grid resolution of the universe.
     * @return True when the neighbor cell is داخل the valid grid range.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Compute how many groups are needed for a cell.
     *
     * The result is based on:
     * - the number of particles currently in the cell
     * - the configured fixed particle count per group
     *
     * The final group may contain fewer particles than the target group size.
     *
     * @param particle_count Number of particles currently in the cell.
     * @param group_particle_count Target number of particles per group.
     * @return Number of groups required for the cell.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    group_count_for_cell(int particle_count, int group_particle_count) noexcept;

private:
    /**
     * @brief Runtime-selected SPH kernel wrapper used for representative interaction.
     */
    SphKernel<T> _kernel {};

    /**
     * @brief Fixed target particle count per deterministic group.
     */
    int _group_particle_count { 5 };

    /**
     * @brief Per-cell number of groups constructed in the current grouped solve.
     */
    DeviceBuffer<int> _cell_group_count {};

    /**
     * @brief Current representative position for each group.
     */
    DeviceBuffer<Vector3<T>> _group_position {};

    /**
     * @brief Current representative velocity for each group.
     */
    DeviceBuffer<Vector3<T>> _group_velocity {};

    /**
     * @brief Updated representative position after the current motion step.
     */
    DeviceBuffer<Vector3<T>> _group_updated_position {};

    /**
     * @brief Updated representative velocity after the current motion step.
     */
    DeviceBuffer<Vector3<T>> _group_updated_velocity {};

    /**
     * @brief Representative or total group mass.
     */
    DeviceBuffer<T> _group_mass {};

    /**
     * @brief Group-level density estimate.
     */
    DeviceBuffer<T> _group_density {};

    /**
     * @brief Group-level pressure estimate.
     */
    DeviceBuffer<T> _group_pressure {};

    /**
     * @brief Number of particle members assigned to each group.
     */
    DeviceBuffer<int> _group_member_count {};

    /**
     * @brief Representative species id used for each group.
     */
    DeviceBuffer<std::size_t> _group_species {};
};

/**
 * @brief Builder for `SphGatewaySolver`.
 *
 * The builder collects all dependencies and grouped-SPH configuration parameters
 * required to construct a valid solver instance.
 *
 * Required configuration:
 * - universe
 * - fluid
 * - searcher
 *
 * Optional configuration:
 * - SPH kernel type
 * - fixed particle count per group
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class SphGatewaySolver<T>::Builder final {
public:
    /**
     * @brief Construct a default-initialized builder.
     */
    Builder() = default;

    /**
     * @brief Set the universe dependency.
     *
     * @param universe Host-side universe instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Set the fluid dependency.
     *
     * @param fluid Host-side fluid instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the spatial hashing searcher dependency.
     *
     * @param searcher Host-side spatial hashing searcher instance.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Set the SPH kernel type used for representative interaction.
     *
     * @param kernel_type Desired SPH kernel type.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    /**
     * @brief Set the target particle count per deterministic group.
     *
     * @param group_particle_count Desired number of particles per group.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_group_particle_count(int group_particle_count) noexcept;

    /**
     * @brief Build a validated `SphGatewaySolver` instance.
     *
     * @return Constructed solver.
     *
     * @throws std::runtime_error Thrown when required dependencies or parameters
     *         are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SphGatewaySolver<T>
    build() const;

    /**
     * @brief Build a validated solver and wrap it in a host-shared pointer.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown when required dependencies or parameters
     *         are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphGatewaySolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the builder configuration.
     *
     * Required:
     * - universe must not be null
     * - fluid must not be null
     * - searcher must not be null
     * - group_particle_count must be positive
     *
     * @throws std::runtime_error Thrown when configuration is invalid.
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
     * @brief Selected SPH kernel type.
     */
    SphKernelType _kernel_type { SphKernelType::standard };

    /**
     * @brief Target number of particles per deterministic group.
     */
    int _group_particle_count { 5 };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::SphGatewaySolver<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphGatewaySolver = atlas::system::SphGatewaySolver<T>;

/**
 * @brief Host-shared-pointer alias for `SphGatewaySolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphGatewaySolverHostPtr = atlas::host_shared_ptr<atlas::system::SphGatewaySolver<T>>;

/**
 * @brief Device-shared-pointer alias for `SphGatewaySolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphGatewaySolverDevicePtr = atlas::device_shared_ptr<atlas::system::SphGatewaySolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_gateway_solver.hpp>