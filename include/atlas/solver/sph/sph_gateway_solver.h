#pragma once

/**
 * @file sph_gateway_solver.h
 * @brief Declares the grouped SPH gateway solver.
 *
 * @details
 * This header defines @ref atlas::SphGatewaySolver, a reduced SPH solver
 * variant that operates on deterministic per-cell group representatives instead
 * of directly updating every particle through full particle-neighbor SPH
 * interactions.
 *
 * The solver is intended as a gateway or reduced-order SPH stage. It compresses
 * particles inside each spatial-search cell into small deterministic groups,
 * evaluates SPH-like density, pressure, and velocity update at the group level,
 * and then scatters the updated representative velocity back to the member
 * particles.
 *
 * The grouped update is useful when the simulation wants a cheaper cell-local
 * SPH approximation, for example in hybrid solver workflows where only selected
 * regions are assigned to the grouped SPH model.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/sph/sph_solver.h>

namespace atlas {

/**
 * @brief Grouped SPH solver that updates particles through per-cell representatives.
 *
 * @details
 * `SphGatewaySolver<T>` derives from `Solver<T>` and implements a reduced
 * representative-based SPH update. Instead of computing SPH interactions for
 * every particle against all neighboring particles, this solver first partitions
 * particles inside each search cell into deterministic groups of up to
 * `_group_particle_count` particles.
 *
 * Each group is represented by one representative entry. The representative
 * stores group-averaged and group-aggregated data:
 *
 * - mean position,
 * - mean velocity,
 * - updated representative position,
 * - updated representative velocity,
 * - total group mass,
 * - group density,
 * - group pressure,
 * - number of valid member particles,
 * - representative species.
 *
 * The representative species is selected from the first valid particle in the
 * group. The group mass is accumulated from all valid member particles whose
 * species index resolves to a valid material property entry.
 *
 * @section sph_gateway_solver_pipeline Solve pipeline
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
 *     -> make_probe()
 *     -> update_cell_particle_counts(allocated_solver, index)
 *     -> build_group_representatives(allocated_solver, index)
 *     -> estimate_group_density_and_pressure(allocated_solver, index)
 *     -> update_group_motion(allocated_solver, index, dt)
 *     -> scatter_group_states_to_particles(allocated_solver, index)
 * @endcode
 *
 * @section sph_gateway_solver_group_indexing Representative indexing
 *
 * Group buffers are sized to the particle count, but only selected entries are
 * used as representative slots. For a cell with sorted-particle range:
 *
 * @code
 * [cell_begin, cell_end)
 * @endcode
 *
 * the representative index for a local group is:
 *
 * @code
 * representative_index = cell_begin + group_local;
 * @endcode
 *
 * The corresponding member-particle subrange is:
 *
 * @code
 * sorted_begin = cell_begin + group_local * group_particle_count;
 * sorted_end   = min(sorted_begin + group_particle_count, cell_end);
 * @endcode
 *
 * This indexing scheme makes representative placement deterministic and avoids a
 * separate global prefix-sum allocation for group representatives.
 *
 * @section sph_gateway_solver_filtering Solver filtering
 *
 * If `allocated_solver` is provided, only cells whose allocation entry equals
 * the requested solver `index` are processed. Cells assigned to another solver
 * are ignored by grouped-SPH stages. During particle/group counting, non-matching
 * cells have their cell particle count and group count reset to zero.
 *
 * @section sph_gateway_solver_scope Interaction scope
 *
 * The current grouped implementation evaluates group interactions only among
 * representatives that belong to the same search cell. Unlike the full
 * `SphSolver<T>` implementation, this grouped gateway solver does not currently
 * search neighboring cells for group interactions.
 *
 * @tparam T Floating-point scalar type used by the simulation.
 *
 * @note
 * This solver updates particle velocities by scattering representative velocities
 * back to member particles. It does not update particle positions directly.
 *
 * @note
 * The grouped approximation is deterministic with respect to the sorted-particle
 * order provided by the spatial hashing searcher.
 *
 * @see Solver
 * @see SphSolver
 * @see SpatialHashingSearcher
 */
template <typename T>
class SphGatewaySolver : public Solver<T> {
public:
    /**
     * @brief Runtime SPH probe type shared with the full SPH solver.
     */
    using SphSolverProbe = typename SphSolver<T>::SphSolverProbe;

    /**
     * @brief Fluent builder for constructing validated `SphGatewaySolver` instances.
     *
     * @details
     * The builder collects required runtime dependencies and grouped-SPH
     * configuration before constructing a solver instance.
     */
    class Builder;

    /**
     * @brief Constructs an empty grouped SPH solver.
     *
     * @details
     * The default constructor leaves the solver without configured universe,
     * fluid, or searcher dependencies. Calling `solve()` on such an object is
     * safe because context initialization fails, transient buffers are cleared,
     * and no grouped-SPH device kernels are launched.
     *
     * The default runtime values are:
     *
     * - kernel: default-constructed `SphKernel<T>`,
     * - group particle count: `5`.
     */
    SphGatewaySolver() = default;

    /**
     * @brief Constructs a grouped SPH solver from simulation dependencies.
     *
     * @details
     * The constructor forwards the universe, fluid, and searcher dependencies to
     * the base `Solver<T>` class, initializes the runtime SPH kernel from
     * @p kernel_type, stores the target group size, and ensures required
     * universe-side output states exist.
     *
     * If @p group_particle_count is not positive, the constructor replaces it
     * with `5`. This keeps direct construction tolerant of invalid group-size
     * input. In contrast, the builder validates the value and rejects
     * non-positive group sizes.
     *
     * @param universe Universe containing per-cell SPH output states.
     * @param fluid Fluid containing particle position, velocity, species, and
     *        material-property data.
     * @param searcher Spatial hashing searcher used to build sorted particle
     *        ranges for each cell.
     * @param kernel_type SPH kernel model used for representative density,
     *        pressure-gradient, and viscosity terms.
     * @param group_particle_count Target number of particles per group.
     *        Non-positive values are replaced with `5`.
     *
     * @post Required universe output states are created when @p universe is valid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphGatewaySolver(UniverseHostPtr<T> universe,
                     FluidHostPtr<T> fluid,
                     SearcherHostPtr<T> searcher,
                     SphKernelType kernel_type = SphKernelType::standard,
                     int group_particle_count  = 5) noexcept;

    /**
     * @brief Destroys the grouped SPH solver through the base interface.
     *
     * @details
     * Uses default destruction. Device buffers and shared pointers are released
     * through their own RAII semantics.
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
     * @details
     * The value is read from the runtime `SphKernel<T>` wrapper stored by the
     * solver.
     *
     * @return Kernel type used by grouped representative interactions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    /**
     * @brief Returns the target number of particles represented by each group.
     *
     * @details
     * Groups are formed independently in each search cell from contiguous
     * sorted-particle ranges. The last group in a cell may contain fewer particles
     * than this target.
     *
     * @return Positive target particle count per group.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    group_particle_count() const noexcept;

    /**
     * @brief Runs the grouped SPH update over all cells.
     *
     * @details
     * This overload performs an unfiltered grouped-SPH solve by forwarding to:
     *
     * @code
     * solve(nullptr, 0, dt);
     * @endcode
     *
     * Since no allocated-solver buffer is provided, every valid cell is eligible
     * for grouped-SPH processing.
     *
     * @param dt Positive time-step size used for representative velocity update.
     *
     * @throw std::invalid_argument Thrown if @p dt is not positive after context
     *        initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    /**
     * @brief Runs the grouped SPH update with optional per-cell solver filtering.
     *
     * @details
     * This function executes the complete grouped-SPH pipeline for one time step.
     *
     * The function first initializes the runtime context, validates the time step,
     * prepares transient group buffers, and refreshes `_probe`. The cached probe
     * is then reused across all grouped stages to avoid repeated state lookups
     * and raw-pointer extraction.
     *
     * Processing stages:
     *
     * 1. Count particles and groups per selected cell.
     * 2. Build deterministic group representatives.
     * 3. Estimate group density and pressure.
     * 4. Update representative velocities and per-cell force output.
     * 5. Scatter updated representative velocities back to member particles.
     *
     * If context initialization fails, group-field preparation fails, or probe
     * construction fails, the function returns without completing the grouped
     * update.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer. When
     *        non-null, only cells whose allocation value equals @p index are
     *        processed.
     * @param index Solver index used when filtering through @p allocated_solver.
     * @param dt Positive time-step size used for explicit representative velocity
     *        update.
     *
     * @throw std::invalid_argument Thrown if @p dt is not positive after context
     *        initialization succeeds.
     *
     * @pre The solver should have valid universe, fluid, and searcher dependencies.
     * @pre The fluid should contain position, velocity, and species states.
     *
     * @post Selected valid cells have updated grouped-SPH output fields.
     * @post Particles in selected valid cells receive their representative group
     *       velocity.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    /**
     * @brief Ensures universe-side output states required by the solver exist.
     *
     * @details
     * If a universe is configured, this function creates the following missing
     * states with `universe->number_of_cells()` entries:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseFieldForceState<T>`.
     *
     * Existing states are left unchanged. If no universe is configured, the
     * function is a no-op.
     *
     * @post The required universe-side output states exist when a universe is
     *       attached.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    /**
     * @brief Initializes the grouped SPH context for one solve step.
     *
     * @details
     * This function validates the runtime dependencies and fluid states required
     * by grouped-SPH kernels.
     *
     * Required dependencies:
     *
     * - universe,
     * - fluid,
     * - spatial hashing searcher.
     *
     * Required fluid states:
     *
     * - `FluidPositionState<T>`,
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`.
     *
     * If any dependency or required fluid state is missing, universe fields are
     * reset, all transient group buffers are cleared, and the function returns
     * `false`.
     *
     * On success, the function ensures required universe states exist, rebuilds
     * the spatial searcher by calling `searcher->build()`, and returns `true`.
     *
     * @retval true The grouped-SPH context is ready for the current solve step.
     * @retval false Required dependencies or fluid states were missing.
     *
     * @post On failure, transient group buffers are empty.
     * @post On success, searcher cell ranges are rebuilt for the current particle
     *       configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_context() noexcept;

    /**
     * @brief Allocates grouped-SPH working buffers for the current step.
     *
     * @details
     * The solver allocates:
     *
     * - `_cell_group_count` with one entry per universe cell,
     * - `_group_position` with one entry per particle,
     * - `_group_velocity` with one entry per particle,
     * - `_group_updated_position` with one entry per particle,
     * - `_group_updated_velocity` with one entry per particle,
     * - `_group_mass` with one entry per particle,
     * - `_group_density` with one entry per particle,
     * - `_group_pressure` with one entry per particle,
     * - `_group_member_count` with one entry per particle,
     * - `_group_species` with one entry per particle.
     *
     * Group buffers are sized to the particle count even though only
     * representative slots are used. A representative slot is addressed by:
     *
     * @code
     * representative_index = cell_begin + group_local;
     * @endcode
     *
     * Representative slots are overwritten by the grouped solve stages, so the
     * group buffers are not pre-cleared. Universe-side output fields are reset.
     *
     * If either particle count or cell count is not positive, all group buffers
     * are cleared, universe fields are reset, and the function returns `false`.
     *
     * @retval true Group buffers were allocated.
     * @retval false Particle count or cell count was not positive.
     *
     * @post On success, transient group buffers are sized for the current solve step.
     * @post On failure, transient group buffers are empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_group_fields();

    /**
     * @brief Resets universe-side grouped-SPH output fields to zero.
     *
     * @details
     * If a universe exists, this function ensures required universe states exist
     * and resets:
     *
     * - `UniverseNumberParticleState<T>` to zero,
     * - `UniverseFieldForceState<T>` to zero vectors.
     *
     * If no universe exists, the function is a no-op.
     *
     * @post Per-cell particle counts are zero when the state exists.
     * @post Per-cell force outputs are zero vectors when the state exists.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    /**
     * @brief Refreshes the cached SPH probe from this solver's dependencies.
     *
     * The function resolves required fluid states, universe output states,
     * searcher buffers, scalar metadata, and the runtime kernel into `_probe`.
     *
     * @retval true Required dependencies and states were found.
     * @retval false A required dependency or state was missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    /**
     * @brief Updates per-cell particle counts and representative group counts.
     *
     * @details
     * For each cell, this function reads the sorted-particle range generated by
     * the spatial hashing searcher:
     *
     * @code
     * begin = probe.cell_start_ptr[cell];
     * end   = probe.cell_end_ptr[cell];
     * @endcode
     *
     * If the range is valid, the cell particle count is:
     *
     * @code
     * count = end - begin;
     * @endcode
     *
     * The function writes:
     *
     * @code
     * probe.number_particle_ptr[cell] = count;
     * _cell_group_count[cell] = group_count_for_cell(count, _group_particle_count);
     * @endcode
     *
     * If the probe contains solver allocation data and the cell allocation does
     * not match @p index, both the particle count and group count are reset to
     * zero.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when @p allocated_solver is non-null.
     *
     * @pre `_cell_group_count` must be allocated with one entry per cell.
     * @pre `probe.number_particle_ptr` must be writable.
     * @pre `probe.cell_start_ptr` and `probe.cell_end_ptr` must be readable.
     *
     * @post Selected cells contain updated particle and group counts.
     * @post Non-selected cells contain zero particle and group counts when solver
     *       filtering is enabled.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Builds deterministic group representatives for selected cells.
     *
     * @details
     * This stage partitions the sorted particles of each selected cell into
     * contiguous groups of up to `_group_particle_count` entries.
     *
     * For each group, the representative slot is:
     *
     * @code
     * representative_index = cell_begin + group_local;
     * @endcode
     *
     * and the member range is:
     *
     * @code
     * sorted_begin = cell_begin + group_local * group_particle_count;
     * sorted_end   = min(sorted_begin + group_particle_count, cell_end);
     * @endcode
     *
     * For each representative group, the function computes:
     *
     * - mean position,
     * - mean velocity,
     * - total mass over members with valid species/material data,
     * - number of valid member particles,
     * - representative species from the first valid member.
     *
     * The initial updated representative position and velocity are initialized to
     * the computed mean position and mean velocity.
     *
     * Cells assigned to another solver are skipped when solver filtering is
     * enabled.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when @p allocated_solver is non-null.
     *
     * @pre `_cell_group_count` must contain valid group counts.
     * @pre Group buffers must be allocated with at least one entry per particle.
     * @pre `probe.indices_ptr`, `probe.position_ptr`, `probe.velocity_ptr`, and
     *      `probe.species_ptr` must be readable.
     *
     * @post Selected groups contain representative position, velocity, mass,
     *       member count, and species data.
     * @post Updated representative position and velocity are initialized from the
     *       mean group state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_group_representatives(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Estimates group-level density and pressure inside each selected cell.
     *
     * @details
     * This stage computes density and pressure for each group representative.
     * Interactions are evaluated only among representatives in the same cell.
     *
     * For a left-hand-side group representative, density is accumulated as:
     *
     * @code
     * density += group_mass[rhs] * kernel.density_weight(radius, cell_size);
     * @endcode
     *
     * where `radius` is the distance between the left-hand-side and right-hand-side
     * representative positions.
     *
     * Material properties are resolved from the representative species. The
     * material controls:
     *
     * - rest density,
     * - pressure coefficient.
     *
     * The interaction support is the searcher cell size.
     *
     * If the accumulated density is non-positive, the density falls back to the
     * material rest density. Pressure is then computed as:
     *
     * @code
     * pressure = pressure_coefficient * (density - rest_density);
     * @endcode
     *
     * Groups with invalid representative species receive zero density and zero
     * pressure.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when @p allocated_solver is non-null.
     *
     * @pre Group representatives must already be built.
     * @pre `_group_position`, `_group_mass`, and `_group_species` must be readable.
     * @pre `_group_density` and `_group_pressure` must be writable.
     *
     * @post Selected valid groups contain updated density and pressure values.
     * @post Groups with invalid representative species contain zero density and
     *       zero pressure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Updates group velocities and writes per-cell averaged force output.
     *
     * @details
     * This stage updates each selected group representative using SPH pressure and
     * optional viscosity terms. Interactions are evaluated only against other
     * representatives in the same cell.
     *
     * For each left-hand-side group, the pressure contribution uses:
     *
     * @code
     * pressure_term = (pressure_lhs + pressure_rhs) / (2 * density_rhs);
     * acceleration -= pressure_gradient * (mass_rhs * pressure_term);
     * @endcode
     *
     * If the material dynamic viscosity is positive, a viscosity contribution is
     * also added:
     *
     * @code
     * acceleration +=
     *     (velocity_rhs - velocity_lhs)
     *     * (viscosity * mass_rhs * viscosity_laplacian / density_rhs);
     * @endcode
     *
     * The representative velocity is updated explicitly:
     *
     * @code
     * updated_velocity = velocity + acceleration * dt;
     * @endcode
     *
     * The updated representative position is currently set to the original
     * representative position. This gateway solver updates velocities only and
     * does not integrate representative positions.
     *
     * The function also writes a per-cell averaged force-like output:
     *
     * @code
     * field_force[cell] = average(acceleration[group] * group_mass[group]);
     * @endcode
     *
     * Empty cells, invalid cells, or cells with no active groups receive a zero
     * force vector.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when @p allocated_solver is non-null.
     * @param dt Positive time-step size used for explicit representative velocity
     *        update.
     *
     * @pre Group representatives, density, and pressure must already be computed.
     * @pre `_group_updated_velocity` and `_group_updated_position` must be writable.
     * @pre `probe.field_force_ptr` must be writable.
     * @pre @p dt must be positive.
     *
     * @post Selected valid groups contain updated representative velocities.
     * @post Selected cells contain averaged force output.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_group_motion(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    /**
     * @brief Copies updated representative velocity back to member particles.
     *
     * @details
     * This is the final grouped-SPH stage. For each selected cell and each group,
     * the function reconstructs the same sorted-particle subrange used during
     * representative construction:
     *
     * @code
     * sorted_begin = cell_begin + group_local * group_particle_count;
     * sorted_end   = min(sorted_begin + group_particle_count, cell_end);
     * @endcode
     *
     * Every valid particle in that range receives:
     *
     * @code
     * particle_velocity = group_updated_velocity[representative_index];
     * @endcode
     *
     * This stage does not update particle positions.
     *
     * @param allocated_solver Optional per-cell solver-allocation buffer.
     * @param index Solver index used when @p allocated_solver is non-null.
     *
     * @pre `_group_updated_velocity` must contain valid updated representative
     *      velocities.
     * @pre `_cell_group_count` must contain valid group counts.
     * @pre `probe.velocity_ptr` must be writable.
     *
     * @post Member particles in selected cells receive their group's updated
     *       representative velocity.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, int index);

    /**
     * @brief Returns the effective rest density for a material.
     *
     * @details
     * If `property.rest_density` exists and is positive, that value is used.
     * Otherwise, `T(1)` is used as a safe fallback.
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
     * @details
     * If `property.pressure_coefficient` is set, that value is returned.
     * Otherwise, `T(0)` is returned, which disables pressure response for the
     * material.
     *
     * @param property Material property record.
     *
     * @return Pressure coefficient.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MaterialProperties<T>& property) noexcept;

    /**
     * @brief Computes the number of deterministic groups required for a cell.
     *
     * @details
     * The result is the ceiling division:
     *
     * @code
     * (particle_count + group_particle_count - 1) / group_particle_count
     * @endcode
     *
     * This gives the number of groups required to cover all particles in a cell
     * when each group contains at most @p group_particle_count particles.
     *
     * If @p particle_count or @p group_particle_count is not positive, the
     * function returns zero.
     *
     * @param particle_count Number of particles in the cell.
     * @param group_particle_count Target number of particles per group.
     *
     * @return Number of groups required for the cell.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    group_count_for_cell(int particle_count, int group_particle_count) noexcept;

protected:
    /**
     * @brief Cached probe populated by @ref make_probe.
     *
     * The grouped solver refreshes this once per solve step after the searcher
     * has been rebuilt. Grouped device stages copy it by value before launching
     * kernels so each launch observes a stable raw-pointer snapshot.
     */
    SphSolverProbe _probe {};

    /**
     * @brief Runtime-selected SPH kernel used for representative interaction.
     *
     * @details
     * The kernel provides density weighting, pressure-gradient, and viscosity
     * Laplacian functions used by the grouped representative update.
     */
    SphKernel<T> _kernel {};

    /**
     * @brief Target number of particles per deterministic group.
     *
     * @details
     * Groups are formed independently in each cell from contiguous sorted-particle
     * ranges. The final group in a cell may contain fewer particles than this
     * target.
     *
     * Direct constructor input is clamped to `5` when non-positive. Builder
     * validation rejects non-positive values before construction.
     */
    int _group_particle_count { 5 };

    /**
     * @brief Per-cell group count for the current solve step.
     *
     * @details
     * This buffer has one entry per universe cell. It is resized by
     * `prepare_group_fields()` and overwritten by `update_cell_particle_counts()`.
     */
    DeviceBuffer<int> _cell_group_count {};

    /**
     * @brief Per-representative mean group position.
     *
     * @details
     * This buffer is resized to the particle count. Only representative slots are
     * used. For a group inside a cell, the representative slot is:
     *
     * @code
     * representative_index = cell_begin + group_local;
     * @endcode
     */
    DeviceBuffer<Vector3<T>> _group_position {};

    /**
     * @brief Per-representative mean group velocity.
     *
     * @details
     * This buffer is resized to the particle count and populated by
     * `build_group_representatives()`. Only representative slots contain valid
     * data.
     */
    DeviceBuffer<Vector3<T>> _group_velocity {};

    /**
     * @brief Per-representative updated position.
     *
     * @details
     * This buffer is initialized to the representative mean position. During
     * `update_group_motion()`, it is currently assigned the original
     * representative position because this solver does not integrate
     * representative or particle positions.
     */
    DeviceBuffer<Vector3<T>> _group_updated_position {};

    /**
     * @brief Per-representative updated velocity.
     *
     * @details
     * This buffer is initialized to the representative mean velocity, updated by
     * `update_group_motion()`, and scattered back to member particles by
     * `scatter_group_states_to_particles()`.
     */
    DeviceBuffer<Vector3<T>> _group_updated_velocity {};

    /**
     * @brief Per-representative total group mass.
     *
     * @details
     * Computed as the sum of material masses over valid member particles during
     * `build_group_representatives()`. Groups with invalid or zero total mass are
     * ignored during the representative motion update.
     */
    DeviceBuffer<T> _group_mass {};

    /**
     * @brief Per-representative group density.
     *
     * @details
     * Populated by `estimate_group_density_and_pressure()` from same-cell
     * representative density interactions.
     */
    DeviceBuffer<T> _group_density {};

    /**
     * @brief Per-representative group pressure.
     *
     * @details
     * Populated by `estimate_group_density_and_pressure()` using:
     *
     * @code
     * pressure = pressure_coefficient * (density - rest_density);
     * @endcode
     */
    DeviceBuffer<T> _group_pressure {};

    /**
     * @brief Per-representative number of valid member particles.
     *
     * @details
     * Populated by `build_group_representatives()`. This value records how many
     * valid particles contributed to the representative mean position and
     * velocity.
     */
    DeviceBuffer<int> _group_member_count {};

    /**
     * @brief Per-representative species index.
     *
     * @details
     * The species of the first valid member particle in a group is used as the
     * representative species. It is later used to resolve material properties for
     * density, pressure, and viscosity.
     */
    DeviceBuffer<std::size_t> _group_species {};
};

/**
 * @brief Fluent builder for `SphGatewaySolver`.
 *
 * @details
 * The builder collects required dependencies and grouped-SPH configuration before
 * constructing a validated `SphGatewaySolver<T>`.
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
 * @tparam T Floating-point scalar type used by the grouped SPH solver.
 */
template <typename T>
class SphGatewaySolver<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     *
     * @details
     * Required dependencies are initialized to null. The default kernel type is
     * `SphKernelType::standard`, and the default group particle count is `5`.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @details
     * The universe provides cell topology and per-cell output states such as
     * `UniverseNumberParticleState<T>` and `UniverseFieldForceState<T>`.
     *
     * @param universe Universe used for grouped-SPH per-cell output states.
     * @return Reference to this builder.
     *
     * @post The builder stores @p universe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency.
     *
     * @details
     * The fluid provides particle position, velocity, species, and material
     * properties required by the grouped-SPH update.
     *
     * @param fluid Fluid used for particle and material data.
     * @return Reference to this builder.
     *
     * @post The builder stores @p fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * @details
     * The searcher provides sorted particle indices and per-cell particle ranges
     * used to build deterministic groups.
     *
     * @param searcher Searcher used for cell-local particle ordering.
     * @return Reference to this builder.
     *
     * @post The builder stores @p searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the SPH kernel type.
     *
     * @details
     * The kernel type controls the runtime `SphKernel<T>` used for density,
     * pressure-gradient, and viscosity kernel evaluations.
     *
     * @param kernel_type Kernel type used to construct the runtime SPH kernel.
     * @return Reference to this builder.
     *
     * @post The builder stores @p kernel_type.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    /**
     * @brief Sets the target number of particles per representative group.
     *
     * @details
     * The builder stores the value as-is and validates positivity during
     * `build()` or `make_host_shared()`.
     *
     * @param group_particle_count Target number of particles per group.
     * @return Reference to this builder.
     *
     * @post The builder stores @p group_particle_count.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_group_particle_count(int group_particle_count) noexcept;

    /**
     * @brief Builds a validated grouped SPH solver value.
     *
     * @details
     * Validation requires:
     *
     * - universe dependency is non-null,
     * - fluid dependency is non-null,
     * - searcher dependency is non-null,
     * - group particle count is positive.
     *
     * @return Constructed grouped SPH solver.
     *
     * @throw std::runtime_error Thrown if the universe dependency is missing.
     * @throw std::runtime_error Thrown if the fluid dependency is missing.
     * @throw std::runtime_error Thrown if the searcher dependency is missing.
     * @throw std::runtime_error Thrown if `group_particle_count` is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SphGatewaySolver<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared grouped SPH solver.
     *
     * @details
     * This function performs the same validation as `build()`, but returns the
     * constructed solver as an `atlas::host_shared_ptr`.
     *
     * @return Host-side shared pointer to the constructed grouped SPH solver.
     *
     * @throw std::runtime_error Thrown if the universe dependency is missing.
     * @throw std::runtime_error Thrown if the fluid dependency is missing.
     * @throw std::runtime_error Thrown if the searcher dependency is missing.
     * @throw std::runtime_error Thrown if `group_particle_count` is not positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphGatewaySolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates that all required builder fields are configured.
     *
     * @details
     * Required:
     *
     * - universe must not be null,
     * - fluid must not be null,
     * - searcher must not be null,
     * - group particle count must be positive.
     *
     * @throw std::runtime_error Thrown if the universe dependency is missing.
     * @throw std::runtime_error Thrown if the fluid dependency is missing.
     * @throw std::runtime_error Thrown if the searcher dependency is missing.
     * @throw std::runtime_error Thrown if `group_particle_count` is not positive.
     *
     * @post No builder state is modified.
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
    SearcherHostPtr<T> _searcher {};

    /**
     * @brief Kernel type collected by the builder.
     */
    SphKernelType _kernel_type { SphKernelType::standard };

    /**
     * @brief Target number of particles per group collected by the builder.
     */
    int _group_particle_count { 5 };
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for `atlas::SphGatewaySolver`.
 *
 * @tparam T Floating-point scalar type used by the grouped SPH solver.
 */
/**
 * @brief Host-side shared pointer alias for `atlas::SphGatewaySolver`.
 *
 * @tparam T Floating-point scalar type used by the grouped SPH solver.
 */
template <typename T>
using SphGatewaySolverHostPtr = atlas::host_shared_ptr<atlas::SphGatewaySolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::SphGatewaySolver`.
 *
 * @tparam T Floating-point scalar type used by the grouped SPH solver.
 */
template <typename T>
using SphGatewaySolverDevicePtr = atlas::device_shared_ptr<atlas::SphGatewaySolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_gateway_solver.hpp>
