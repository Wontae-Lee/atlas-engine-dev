#pragma once

/**
 * @file sph_solver.h
 * @brief Declares the SPH solver used to estimate particle fields and update velocities.
 *
 * This header defines `atlas::system::SphSolver<T>`, a smooth-particle-
 * hydrodynamics style solver that computes density, pressure, acceleration, and
 * cell-level force/particle-count fields from local particle neighborhoods.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>

namespace atlas::system {

/**
 * @brief Smooth-particle-hydrodynamics solver using spatial-hash neighbor traversal.
 *
 * `SphSolver<T>` derives from @ref Solver and implements an explicit SPH-style
 * velocity update. It uses the spatial hashing searcher to find particles in
 * neighboring grid cells, computes per-particle density and pressure, then
 * accumulates pressure-gradient and viscosity terms into an acceleration field.
 *
 * The primary execution path is @ref solve(T), which performs:
 *
 * @code
 * if (!initialize_sph_context()) return;
 * if (!(dt > 0)) throw std::invalid_argument(...);
 * if (!prepare_particle_fields()) {
 *     reset_universe_fields();
 *     return;
 * }
 * make_probe();
 * update();
 * accumulate_acceleration(dt);
 * @endcode
 *
 * The internal @ref update stage executes:
 *
 * @code
 * estimate_particle_density_and_pressure(probe);
 * update_cell_number_particles(probe);
 * @endcode
 *
 * The codec-aware `solve(const DeviceBuffer<int>*, int, T)` overload is currently
 * a no-op. The SPH solver therefore does not currently support per-cell solver
 * allocation filtering.
 *
 * @tparam T Scalar type used by the simulation.
 */
template <typename T>
class SphSolver final : public Solver<T> {
public:
    /**
     * @brief Raw-pointer view over common SPH runtime data.
     *
     * `SphSolverProbe` is populated by @ref make_probe, cached in `_probe`, and is
     * intended to be captured by value in device kernels. It contains fluid particle data,
     * universe output buffers, searcher cell ranges, spatial-grid metadata, and
     * the runtime smoothing kernel.
     *
     * Required states for a valid probe are:
     *
     * - `FluidPositionState<T>`,
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`,
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseFieldForceState<T>`.
     */
    struct SphSolverProbe {
        /**
         * @brief Raw pointer to per-particle positions.
         *
         * Points to `FluidPositionState<T>::data()`.
         */
        const Vector3<T>* position_ptr {};

        /**
         * @brief Raw pointer to mutable per-particle velocities.
         *
         * Points to `FluidVelocityState<T>::data()`. The acceleration stage
         * updates this buffer in place.
         */
        Vector3<T>* velocity_ptr {};

        /**
         * @brief Raw pointer to per-particle species indices.
         *
         * Points to `FluidSpeciesState<T>::data()`.
         */
        const std::size_t* species_ptr {};

        /**
         * @brief Raw pointer to per-species material properties.
         *
         * Points to `fluid->particle_properties()`. SPH parameters such as mass,
         * smoothing length, rest density, pressure coefficient, and dynamic
         * viscosity are read from this array.
         */
        const MaterialProperties<T>* properties_ptr {};

        /**
         * @brief Raw pointer to per-cell particle-count output.
         *
         * Points to `UniverseNumberParticleState<T>::data()`.
         */
        T* number_particle_ptr {};

        /**
         * @brief Raw pointer to per-cell averaged force output.
         *
         * Points to `UniverseFieldForceState<T>::data()`. The acceleration stage
         * stores the average force-like value per occupied cell.
         */
        Vector3<T>* field_force_ptr {};

        /**
         * @brief Raw pointer to sorted particle indices produced by the searcher.
         *
         * For each cell, `[cell_start_ptr[cell], cell_end_ptr[cell])` indexes
         * into this array.
         */
        const int* indices_ptr {};

        /**
         * @brief Raw pointer to the first sorted index for each cell.
         */
        const int* cell_start_ptr {};

        /**
         * @brief Raw pointer to one-past-the-last sorted index for each cell.
         */
        const int* cell_end_ptr {};

        /**
         * @brief Lower corner of the searcher grid domain.
         */
        Vector3<T> lower_corner {};

        /**
         * @brief Integer grid resolution used by the searcher.
         */
        Vector3<int> grid_size {};

        /**
         * @brief Reciprocal of the searcher cell size.
         */
        T inverse_cell_size {};

        /**
         * @brief Searcher cell size.
         *
         * Used as the fallback smoothing length when material properties do not
         * define a positive smoothing length.
         */
        T cell_size {};

        /**
         * @brief Number of particles reported by the fluid.
         */
        int particle_count {};

        /**
         * @brief Number of cells reported by the universe.
         */
        int num_of_cells {};

        /**
         * @brief Number of species material-property entries.
         */
        int num_of_properties {};

        /**
         * @brief Runtime SPH kernel wrapper used for density, pressure, and viscosity terms.
         */
        SphKernel<T> kernel {};
    };

    /**
     * @brief Fluent builder for constructing validated `SphSolver` instances.
     */
    class Builder;

public:
    /**
     * @brief Constructs an empty SPH solver.
     *
     * Dependencies are initialized by the @ref Solver base class default state.
     * Calling @ref solve on an unconfigured solver is safe: context initialization
     * fails, transient buffers are cleared, and no device kernels are launched.
     */
    SphSolver() = default;

    /**
     * @brief Constructs an SPH solver with simulation dependencies and a kernel type.
     *
     * The constructor forwards the universe, fluid, and searcher to the
     * @ref Solver base class, initializes the runtime SPH kernel from
     * `kernel_type`, and calls @ref ensure_universe_states.
     *
     * @param universe Universe containing SPH per-cell output states.
     * @param fluid Fluid containing particle position, velocity, species, and material data.
     * @param searcher Spatial hashing searcher used for cell and neighbor traversal.
     * @param kernel_type SPH kernel model used for smoothing operations.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphSolver(UniverseHostPtr<T> universe,
              FluidHostPtr<T> fluid,
              SpatialHashingSearcherHostPtr<T> searcher,
              SphKernelType kernel_type = SphKernelType::standard) noexcept;

    /**
     * @brief Destroys the SPH solver through the base interface.
     */
    ~SphSolver() override = default;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct an SPH solver.
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
     * @brief Executes one explicit SPH velocity-update step.
     *
     * The solve step initializes the runtime context, validates that `dt` is
     * positive, prepares solver-owned particle buffers, estimates density and
     * pressure, updates per-cell particle counts, accumulates acceleration, and
     * writes velocity updates in place.
     *
     * Required dependencies and states are checked by @ref initialize_sph_context
     * and @ref make_probe. If required dependencies or states are unavailable,
     * the solver returns without work.
     *
     * @param dt Positive time-step size used to integrate acceleration into velocity.
     *
     * @throw std::invalid_argument If `dt` is not positive after context
     *                              initialization succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    /**
     * @brief Codec-aware SPH solve overload.
     *
     * This overload is currently intentionally empty. It ignores all arguments
     * and performs no SPH update.
     *
     * @param allocated_solver Unused per-cell solver-allocation buffer.
     * @param index Unused solver index.
     * @param dt Unused time-step size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    /**
     * @brief Ensures SPH universe-side output states exist.
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
     * @brief Initializes the SPH context for one solve step.
     *
     * The function verifies that universe, fluid, and searcher dependencies exist
     * and that the fluid provides:
     *
     * - `FluidPositionState<T>`,
     * - `FluidVelocityState<T>`,
     * - `FluidSpeciesState<T>`.
     *
     * If dependencies or required fluid states are missing, it resets universe
     * fields, clears solver-owned particle buffers, and returns `false`.
     *
     * On success, it ensures required universe states exist, rebuilds the searcher
     * by calling `searcher->build()`, and returns `true`.
     *
     * @retval true The SPH context is ready for the solve step.
     * @retval false Required dependencies or fluid states were missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_sph_context() noexcept;

    /**
     * @brief Populates an SPH probe from this solver's configured dependencies.
     *
     * This function resolves this solver's universe, fluid, searcher, and kernel
     * into the cached `_probe`.
     *
     * @retval true Required dependencies and states were found.
     * @retval false A required dependency or state was missing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    /**
     * @brief Allocates and clears per-particle working fields for the current step.
     *
     * The working buffers are resized to `fluid->particle_count()` and cleared:
     *
     * - `_density` is filled with zero,
     * - `_pressure` is filled with zero,
     * - `_acceleration` is filled with zero vectors.
     *
     * Universe output fields are reset by calling @ref reset_universe_fields.
     *
     * If the particle count is not positive, all working buffers are cleared,
     * universe fields are reset, and the function returns `false`.
     *
     * @retval true Working fields were allocated and cleared.
     * @retval false Particle count was zero or negative.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_particle_fields();

    /**
     * @brief Resets universe-side SPH output fields to zero.
     *
     * If a universe exists, this function ensures required universe states exist
     * and resets:
     *
     * - `UniverseNumberParticleState<T>` to zero,
     * - `UniverseFieldForceState<T>` to zero vectors.
     *
     * If no universe exists, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    /**
     * @brief Returns the effective smoothing length for a material.
     *
     * If `property.smoothing_length` exists and is positive, that value is
     * returned. Otherwise, the searcher cell size is used as the fallback.
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
     * @param smoothing_length Effective smoothing length.
     * @param cell_size Searcher cell size.
     *
     * @return Integer radius in grid cells.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    search_radius_for(T smoothing_length, T cell_size) noexcept;

    /**
     * @brief Maps a particle position to a clamped search-grid cell coordinate.
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
     * @param position Particle position.
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
     * @param cell Candidate grid-cell coordinate.
     * @param grid_size Grid resolution.
     *
     * @retval true Cell coordinate is inside the grid.
     * @retval false Cell coordinate is outside the grid.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Estimates per-particle density and pressure using the cached SPH probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_particle_density_and_pressure();

    /**
     * @brief Updates per-cell particle counts using the cached SPH probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_number_particles();

    /**
     * @brief Accumulates SPH acceleration and updates velocities using the cached SPH probe.
     *
     * @param dt Positive time-step size used for explicit velocity integration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    accumulate_acceleration(T dt);

    /**
     * @brief Executes the density/pressure and cell-count update using the cached SPH probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

private:
    /**
     * @brief Cached probe populated by @ref make_probe.
     *
     * The solver refreshes this once per solve step after the searcher has been
     * rebuilt. Device stages copy it by value before launching kernels so each
     * launch observes a stable raw-pointer snapshot.
     */
    SphSolverProbe _probe {};

    /**
     * @brief Runtime-selected SPH smoothing kernel.
     */
    SphKernel<T> _kernel {};

    /**
     * @brief Solver-owned per-particle density working buffer.
     *
     * Resized and cleared by @ref prepare_particle_fields, then populated by
     * @ref estimate_particle_density_and_pressure.
     */
    DeviceBuffer<T> _density {};

    /**
     * @brief Solver-owned per-particle pressure working buffer.
     *
     * Resized and cleared by @ref prepare_particle_fields, then populated by
     * @ref estimate_particle_density_and_pressure.
     */
    DeviceBuffer<T> _pressure {};

    /**
     * @brief Solver-owned per-particle acceleration working buffer.
     *
     * Resized and cleared by @ref prepare_particle_fields, then populated by
     * @ref accumulate_acceleration.
     */
    DeviceBuffer<Vector3<T>> _acceleration {};
};

/**
 * @brief Fluent builder for `SphSolver`.
 *
 * The builder collects required SPH dependencies and an optional kernel type.
 *
 * Required dependencies:
 *
 * - universe,
 * - fluid,
 * - spatial hashing searcher.
 *
 * The default kernel type is `SphKernelType::standard`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
class SphSolver<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency.
     *
     * @param universe Universe used for SPH per-cell output states.
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
     * @param searcher Searcher used for cell mapping and neighbor traversal.
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
     * @brief Builds a validated SPH solver value.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null.
     *
     * @return Constructed SPH solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SphSolver<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared SPH solver.
     *
     * Validation requires universe, fluid, and searcher dependencies to be
     * non-null.
     *
     * @return Host-side shared pointer to the constructed SPH solver.
     *
     * @throw std::runtime_error If the universe dependency is missing.
     * @throw std::runtime_error If the fluid dependency is missing.
     * @throw std::runtime_error If the searcher dependency is missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphSolver<T>>
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
     * @brief Kernel type collected by the builder.
     */
    SphKernelType _kernel_type { SphKernelType::standard };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
using SphSolver = atlas::system::SphSolver<T>;

/**
 * @brief Host-side shared pointer alias for `atlas::system::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
using SphSolverHostPtr = atlas::host_shared_ptr<atlas::system::SphSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::system::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
using SphSolverDevicePtr = atlas::device_shared_ptr<atlas::system::SphSolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_solver.hpp>
