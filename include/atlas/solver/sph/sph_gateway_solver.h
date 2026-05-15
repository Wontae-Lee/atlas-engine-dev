#pragma once

/**
 * @file sph_gateway_solver.h
 * @brief Declares the grouped SPH gateway solver.
 *
 * This header defines `atlas::system::SphGatewaySolver<T>`, an SPH solver
 * variant that compresses cell-local particles into deterministic group
 * representatives, updates those representatives, and scatters representative
 * velocity back to member particles.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/sph/sph_solver.h>

namespace atlas::system {

/**
 * @brief Grouped SPH solver that updates particles through per-cell representatives.
 *
 * `SphGatewaySolver<T>` derives from @ref Solver and implements a reduced
 * representative-based SPH update. Instead of evaluating SPH interaction for
 * every particle, it partitions particles in each cell into deterministic groups
 * of up to @ref group_particle_count particles.
 *
 * For each group, the solver computes representative data:
 *
 * - mean position,
 * - mean velocity,
 * - total mass,
 * - member count,
 * - representative species.
 *
 * It then estimates density and pressure at the group level, updates
 * representative velocity through SPH pressure and viscosity terms, and finally
 * copies each representative velocity back to all particles in that group.
 *
 * The full solve path is:
 *
 * @code
 * solve(dt)
 *     -> solve(nullptr, 0, dt)
 *
 * solve(allocated_solver, index, dt)
 *     -> initialize_context()
 *     -> validate dt > 0
 *     -> prepare_group_fields()
 *     -> update_cell_particle_counts(allocated_solver, index)
 *     -> build_group_representatives(allocated_solver, index)
 *     -> estimate_group_density_and_pressure(allocated_solver, index)
 *     -> update_group_motion(allocated_solver, index, dt)
 *     -> scatter_group_states_to_particles(allocated_solver, index)
 * @endcode
 *
 * When `allocated_solver` is provided, only cells whose allocation entry equals
 * `index` are processed. Non-matching cells have their particle count and group
 * count reset during @ref update_cell_particle_counts.
 *
 * @tparam T Scalar type used by the simulation.
 */
template <typename T>
class SphGatewaySolver final : public Solver<T> {
public:
    /**
     * @brief Fluent builder for constructing validated `SphGatewaySolver` instances.
     */
    class Builder;

public:
    /**
     * @brief Constructs an empty grouped SPH solver.
     *
     * Dependencies are initialized by the @ref Solver base class default state.
     * Calling @ref solve on an unconfigured solver is safe: context initialization
     * fails, transient buffers are cleared, and no device kernels are launched.
     */
    SphGatewaySolver() = default;

    /**
     * @brief Constructs a grouped SPH solver from simulation dependencies.
     *
     * The constructor forwards universe, fluid, and searcher dependencies to the
     * @ref Solver base class, initializes the runtime SPH kernel from
     * `kernel_type`, clamps non-positive `group_particle_count` to `5`, and calls
     * @ref ensure_universe_states.
     *
     * @param universe Universe containing per-cell SPH output states.
     * @param fluid Fluid containing particle position, velocity, species, and material data.
     * @param searcher Spatial hashing searcher used for cell-local particle ordering.
     * @param kernel_type SPH kernel model used for representative interaction.
     * @param group_particle_count Target number of particles per group. Non-positive
     *                             values are replaced with `5`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphGatewaySolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SpatialHashingSearcherHostPtr<T> searcher,
                     SphKernelType kernel_type = SphKernelType::standard,
                     int group_particle_count  = 5) noexcept;

    /**
     * @brief Destroys the grouped SPH solver through the base interface.
     */
    ~SphGatewaySolver() override = default;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct a grouped SPH solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Returns the configured SPH kernel type.
     *
     * @return Kernel type stored in the runtime SPH kernel wrapper.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    /**
     * @brief Returns the target number of particles represented by each group.
     *
     * Groups are formed per cell from contiguous sorted-particle ranges. The last
     * group in a cell may contain fewer particles than this target.
     *
     * @return Positive target particle count per group.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    group_particle_count() const noexcept;

    /**
     * @brief Runs the grouped SPH update over all cells.
     *
     * This overload forwards to:
     *
     * @code
     * solve(nullptr, 0, dt);
     * @endcode
     *
     * @param dt Positive time-step size used for representative velocity update.
     *
     * @throw std::invalid_argument If `dt` is not positive after context
     *                              initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    /**
     * @brief Runs the grouped SPH update with optional per-cell solver filtering.
     *
     * The function initializes the context, validates `dt`, prepares group
     * buffers, updates per-cell particle/group counts, builds group
     * representatives, estimates group density/pressure, updates group velocity,
     * and scatters updated group velocity to particles.
     *
     * If context initialization fails or group-field preparation fails, the
     * function returns without completing the grouped update.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer. When
     *                         non-null, only cells whose value equals `index`
     *                         are processed.
     * @param index Solver index used when filtering through `allocated_solver`.
     * @param dt Positive time-step size.
     *
     * @throw std::invalid_argument If `dt` is not positive after context
     *                              initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    /**
     * @brief Ensures universe-side output states required by the solver exist.
     *
     * If a universe is configured, this function creates the following missing
     * states with `universe->number_of_cells()` elements:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseFieldForceState<T>`.
     *
     * Existing states are left unchanged. If no universe is configured, the
     * function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Initializes the grouped SPH context for one solve step.
     *
     * The function verifies that universe, fluid, and searcher dependencies exist
     * and that the fluid provides:
     *
     * - `FluidPositionState<T>`,
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`.
     *
     * If dependencies or required fluid states are missing, it resets universe
     * fields, clears all group buffers, and returns `false`.
     *
     * On success, it ensures required universe states exist, rebuilds the searcher
     * by calling `searcher->build()`, and returns `true`.
     *
     * @retval true The grouped SPH context is ready.
     * @retval false Required dependencies or fluid states were missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    /**
     * @brief Allocates and clears group working buffers for the current step.
     *
     * The solver allocates:
     *
     * - `_cell_group_count` with one entry per universe cell,
     * - all group data buffers with one entry per particle.
     *
     * Group data is indexed by a representative index derived from the cell's
     * sorted-particle range:
     *
     * @code
     * representative_index = cell_begin + group_local;
     * @endcode
     *
     * All group buffers are cleared to zero-equivalent values, and universe
     * fields are reset through @ref reset_universe_fields.
     *
     * If particle count or cell count is not positive, all group buffers are
     * cleared, universe fields are reset, and the function returns `false`.
     *
     * @retval true Group buffers were allocated and cleared.
     * @retval false Particle count or cell count was not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_group_fields();

    /**
     * @brief Resets universe-side grouped-SPH output fields to zero.
     *
     * If a universe exists, this function ensures required universe states exist
     * and clears:
     *
     * - `UniverseNumberParticleState<T>` to zero,
     * - `UniverseFieldForceState<T>` to zero vectors.
     *
     * If no universe exists, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    /**
     * @brief Updates per-cell particle count and group count.
     *
     * For each cell, the function counts particles using the searcher range:
     *
     * @code
     * count = end - begin
     * @endcode
     *
     * when the range is valid, then writes:
     *
     * - `UniverseNumberParticleState<T>[cell] = count`,
     * - `_cell_group_count[cell] = group_count_for_cell(count, _group_particle_count)`.
     *
     * If `allocated_solver` is provided and the cell assignment does not match
     * `index`, both values are reset to zero.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Builds deterministic group representatives for selected cells.
     *
     * For each eligible cell, particles are partitioned into contiguous chunks of
     * up to `_group_particle_count` entries from the cell's sorted range. For
     * each group, the function computes:
     *
     * - mean particle position,
     * - mean particle velocity,
     * - total mass over valid species entries,
     * - member count,
     * - representative species taken from the first valid member.
     *
     * It writes the representative data at:
     *
     * @code
     * representative_index = cell_begin + group_local;
     * @endcode
     *
     * The initial updated position and velocity are initialized to the computed
     * mean position and velocity.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_group_representatives(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Estimates group-level density and pressure inside each selected cell.
     *
     * The implementation evaluates group interactions only among representatives
     * belonging to the same cell. For each group representative:
     *
     * - material properties are resolved through its representative species,
     * - smoothing length, rest density, and pressure coefficient are computed,
     * - density is accumulated from all group representatives in that cell:
     *
     * @code
     * density += group_mass[rhs] * kernel.density_weight(radius, smoothing_length);
     * @endcode
     *
     * - non-positive density falls back to rest density,
     * - pressure is computed as:
     *
     * @code
     * pressure = pressure_coefficient * (density - rest_density);
     * @endcode
     *
     * Groups with invalid species receive zero density and pressure.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Updates group velocities and writes per-cell averaged force output.
     *
     * For each selected cell, this stage iterates over group representatives in
     * that cell. For each group, it accumulates pressure-gradient and optional
     * viscosity acceleration from other representatives in the same cell.
     *
     * A viscosity term is included only when the representative material's
     * `dynamic_viscosity` is positive.
     *
     * The representative velocity is updated explicitly:
     *
     * @code
     * updated_velocity = velocity + acceleration * dt;
     * @endcode
     *
     * The updated position is currently assigned to the original group position;
     * no position integration is performed by this implementation.
     *
     * For each cell, `UniverseFieldForceState<T>` receives the average of:
     *
     * @code
     * acceleration[group] * group_mass[group]
     * @endcode
     *
     * over active groups. Empty cells, invalid cells, or cells with no active
     * groups receive a zero vector.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     * @param dt Positive time-step size used for representative velocity update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Copies updated representative velocity back to particles.
     *
     * For each selected cell and each representative group, every valid particle
     * in the group's sorted range receives:
     *
     * @code
     * particle_velocity = group_updated_velocity[representative_index];
     * @endcode
     *
     * This stage does not update particle positions.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when `allocated_solver` is non-null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Returns the effective smoothing length for a material.
     *
     * If `property.smoothing_length` exists and is positive, that value is
     * returned. Otherwise, `cell_size` is used as the fallback.
     *
     * @param property Material property record.
     * @param cell_size Searcher cell size used as fallback.
     *
     * @return Effective smoothing length.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    smoothing_length_for(const MaterialProperties<T>& property, T cell_size) noexcept;

    /**
     * @brief Returns the effective rest density for a material.
     *
     * If `property.rest_density` exists and is positive, that value is returned.
     * Otherwise, `T(1)` is used.
     *
     * @param property Material property record.
     *
     * @return Effective rest density.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density_for(const MaterialProperties<T>& property) noexcept;

    /**
     * @brief Returns the pressure coefficient for a material.
     *
     * If `property.pressure_coefficient` is set, that value is returned.
     * Otherwise, `T(0)` is used.
     *
     * @param property Material property record.
     *
     * @return Pressure coefficient.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MaterialProperties<T>& property) noexcept;

    /**
     * @brief Converts a smoothing length to a cell-neighborhood search radius.
     *
     * The returned radius is:
     *
     * @code
     * ceil(smoothing_length / cell_size)
     * @endcode
     *
     * This helper is defined for API symmetry with @ref SphSolver. The current
     * grouped implementation evaluates representative interactions only inside
     * the same cell and does not call this helper.
     *
     * @param smoothing_length Effective smoothing length.
     * @param cell_size Searcher cell size.
     *
     * @return Integer radius in grid cells.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    search_radius_for(T smoothing_length, T cell_size) noexcept;

    /**
     * @brief Maps a position to a clamped search-grid cell coordinate.
     *
     * The function computes:
     *
     * @code
     * floor((position - lower_corner) * inverse_cell_size)
     * @endcode
     *
     * casts the result to integer coordinates, and clamps it to:
     *
     * @code
     * [Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1)]
     * @endcode
     *
     * This helper is defined for API symmetry with @ref SphSolver. The current
     * grouped implementation does not call it.
     *
     * @param position Position to map.
     * @param lower_corner Lower corner of the search grid.
     * @param inverse_cell_size Reciprocal cell size.
     * @param grid_size Grid resolution.
     *
     * @return Clamped integer grid-cell coordinate.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<int>
    particle_cell(const Vector3<T>& position,
                  const Vector3<T>& lower_corner,
                  T inverse_cell_size,
                  const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Checks whether a grid-cell coordinate lies inside the search grid.
     *
     * The cell is valid when each component is in the half-open interval:
     *
     * @code
     * [0, grid_size.component)
     * @endcode
     *
     * This helper is defined for API symmetry with @ref SphSolver. The current
     * grouped implementation does not call it.
     *
     * @param cell Candidate grid-cell coordinate.
     * @param grid_size Grid resolution.
     *
     * @retval true Cell coordinate is inside the grid.
     * @retval false Cell coordinate is outside the grid.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Computes the number of deterministic groups required for a cell.
     *
     * The result is the ceiling division:
     *
     * @code
     * (particle_count + group_particle_count - 1) / group_particle_count
     * @endcode
     *
     * If either input is not positive, the function returns zero.
     *
     * @param particle_count Number of particles in the cell.
     * @param group_particle_count Target number of particles per group.
     *
     * @return Number of groups required for the cell.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    group_count_for_cell(int particle_count, int group_particle_count) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_particle_counts(const typename SphSolver<T>::SphSolverProbe& probe, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_group_representatives(const typename SphSolver<T>::SphSolverProbe& probe, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure(const typename SphSolver<T>::SphSolverProbe& probe, int index);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(const typename SphSolver<T>::SphSolverProbe& probe, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles(const typename SphSolver<T>::SphSolverProbe& probe, int index);

private:
    /**
     * @brief Runtime-selected SPH kernel used for representative interaction.
     */
    SphKernel<T> _kernel {};

    /**
     * @brief Target number of particles per deterministic group.
     *
     * Constructor input is clamped to `5` when non-positive. Builder validation
     * rejects non-positive values before construction.
     */
    int _group_particle_count { 5 };

    /**
     * @brief Per-cell group count for the current solve step.
     *
     * Resized to the universe cell count by @ref prepare_group_fields.
     */
    DeviceBuffer<int> _cell_group_count {};

    /**
     * @brief Per-representative mean group position.
     *
     * Resized to the particle count by @ref prepare_group_fields. Group entries
     * are indexed by `representative_index = cell_begin + group_local`.
     */
    DeviceBuffer<Vector3<T>> _group_position {};

    /**
     * @brief Per-representative mean group velocity.
     *
     * Resized to the particle count by @ref prepare_group_fields.
     */
    DeviceBuffer<Vector3<T>> _group_velocity {};

    /**
     * @brief Per-representative updated position.
     *
     * Currently initialized to the mean group position and later assigned the
     * original representative position during @ref update_group_motion.
     */
    DeviceBuffer<Vector3<T>> _group_updated_position {};

    /**
     * @brief Per-representative updated velocity.
     *
     * Updated by @ref update_group_motion and scattered to member particles by
     * @ref scatter_group_states_to_particles.
     */
    DeviceBuffer<Vector3<T>> _group_updated_velocity {};

    /**
     * @brief Per-representative total group mass.
     *
     * Computed as the sum of valid member particle masses during
     * @ref build_group_representatives.
     */
    DeviceBuffer<T> _group_mass {};

    /**
     * @brief Per-representative group density.
     *
     * Populated by @ref estimate_group_density_and_pressure.
     */
    DeviceBuffer<T> _group_density {};

    /**
     * @brief Per-representative group pressure.
     *
     * Populated by @ref estimate_group_density_and_pressure.
     */
    DeviceBuffer<T> _group_pressure {};

    /**
     * @brief Per-representative number of valid member particles.
     *
     * Populated by @ref build_group_representatives.
     */
    DeviceBuffer<int> _group_member_count {};

    /**
     * @brief Per-representative species index.
     *
     * The species of the first valid member in the group is used as the
     * representative species.
     */
    DeviceBuffer<std::size_t> _group_species {};
};

/**
 * @brief Fluent builder for `SphGatewaySolver`.
 *
 * The builder collects required dependencies and grouped-SPH configuration.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * Optional configuration:
 *
 * - SPH kernel type, defaulting to `SphKernelType::standard`,
 * - target particles per group, defaulting to `5`.
 *
 * Unlike the direct constructor, the builder rejects non-positive
 * `group_particle_count` values during validation.
 *
 * @tparam T Scalar type used by the grouped SPH solver.
 */
template <typename T>
class SphGatewaySolver<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @param universe Universe used for grouped-SPH per-cell output states.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @param fluid Fluid used for particle positions, velocities, species, and material data.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @param searcher Searcher used for cell-local particle ordering.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the SPH kernel type.
     *
     * @param kernel_type Kernel type used to construct the runtime SPH kernel.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    /**
     * @brief Sets the target number of particles per group.
     *
     * The builder stores the value as-is and validates positivity during
     * @ref build or @ref make_host_shared.
     *
     * @param group_particle_count Target number of particles per group.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_group_particle_count(int group_particle_count) noexcept;

    /**
     * @brief Builds a validated grouped SPH solver value.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null, and `group_particle_count` to be positive.
     *
     * @return Constructed grouped SPH solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     * @throw std::runtime_error If `group_particle_count` is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SphGatewaySolver<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared grouped SPH solver.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null, and `group_particle_count` to be positive.
     *
     * @return Host-side shared pointer to the constructed grouped SPH solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     * @throw std::runtime_error If `group_particle_count` is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphGatewaySolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates that all required builder fields are configured.
     *
     * Required:
     *
     * - universe must not be null,
     * - fluid must not be null,
     * - searcher must not be null,
     * - group particle count must be positive.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     * @throw std::runtime_error If `group_particle_count` is not positive.
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
     * @brief Kernel type collected by the builder.
     */
    SphKernelType _kernel_type { SphKernelType::standard };

    /**
     * @brief Target number of particles per group collected by the builder.
     */
    int _group_particle_count { 5 };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::SphGatewaySolver`.
 *
 * @tparam T Scalar type used by the grouped SPH solver.
 */
template <typename T>
using SphGatewaySolver = atlas::system::SphGatewaySolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::SphGatewaySolver`.
 *
 * @tparam T Scalar type used by the grouped SPH solver.
 */
template <typename T>
using SphGatewaySolverHostPtr = atlas::host_shared_ptr<atlas::system::SphGatewaySolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::SphGatewaySolver`.
 *
 * @tparam T Scalar type used by the grouped SPH solver.
 */
template <typename T>
using SphGatewaySolverDevicePtr = atlas::device_shared_ptr<atlas::system::SphGatewaySolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_gateway_solver.hpp>