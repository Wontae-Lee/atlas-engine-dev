#pragma once

/**
 * @file sph_solver.h
 * @brief Declares a smooth-particle-hydrodynamics (SPH) solver that updates particle velocities from local neighborhood interactions.
 *
 * This header defines `SphSolver<T>`, an SPH-style solver that advances particle
 * motion using local smoothing-kernel interactions rather than binary collision
 * events.
 *
 * In contrast to DSMC-based solvers:
 * - no stochastic collision pair selection is performed
 * - no per-cell collision-count scheduling is used
 * - particle evolution is driven by neighborhood-based continuum-style interaction
 *
 * The solver relies on:
 * - a universe defining the spatial grid and domain extents
 * - a fluid containing particle states and material properties
 * - a spatial hashing searcher providing efficient neighborhood lookup
 *
 * At a high level, one SPH step consists of:
 * 1. preparing required field/state buffers
 * 2. rebuilding or reusing the spatial neighborhood structure
 * 3. estimating per-particle density and pressure
 * 4. accumulating SPH accelerations from local neighbors
 * 5. updating particle velocities in place
 */

#include <atlas/core/macros.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Smooth-particle-hydrodynamics style solver driven by local neighbor search.
 *
 * `SphSolver<T>` implements a particle-based continuum interaction model in which
 * each particle is influenced by nearby particles inside a smoothing neighborhood.
 *
 * Instead of discrete binary collisions, the solver computes:
 * - density estimates from smoothing-kernel weights
 * - pressure estimates from the density field and rest material parameters
 * - pressure-gradient and viscosity contributions from neighboring particles
 * - explicit velocity updates from the resulting acceleration field
 *
 * The solver stores temporary per-particle fields such as:
 * - density
 * - pressure
 * - acceleration
 *
 * These fields are solver-owned working buffers and are rebuilt or refreshed during
 * the solve pipeline.
 *
 * The overload `solve(T dt)` is the primary execution entry point.
 * The codec-aware overload taking `allocated_solver` is currently reserved for
 * future implementation and intentionally does not perform any work.
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */
template <typename T>
class SphSolver final : public Solver<T> {
public:
    /**
     * @brief Builder type used to configure and construct `SphSolver`.
     */
    class Builder;

public:
    /**
     * @brief Construct a default-initialized SPH solver.
     *
     * A default-constructed solver is not yet fully configured for a real
     * simulation pipeline. In normal usage, the parameterized constructor or
     * builder interface should be preferred.
     */
    SphSolver() = default;

    /**
     * @brief Construct an SPH solver with all required runtime dependencies.
     *
     * The solver stores shared references to:
     * - the universe
     * - the fluid
     * - the spatial hashing searcher
     *
     * It also initializes the selected runtime SPH smoothing kernel.
     *
     * @param universe Host-side universe describing the domain and grid.
     * @param fluid Host-side fluid containing particle states and material properties.
     * @param searcher Spatial hashing searcher used for particle neighborhood lookup.
     * @param kernel_type SPH kernel type used for density, pressure, and viscosity evaluation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphSolver(UniverseHostPtr<T> universe,
              FluidHostPtr<T> fluid,
              SpatialHashingSearcherHostPtr<T> searcher,
              SphKernelType kernel_type = SphKernelType::standard) noexcept;

    /**
     * @brief Destroy the solver.
     */
    ~SphSolver() override = default;

    /**
     * @brief Create a builder for `SphSolver`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return the active SPH kernel type.
     *
     * @return Currently configured SPH kernel type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphKernelType
    kernel_type() const noexcept;

    /**
     * @brief Execute one SPH solve step using the current spatial neighborhood.
     *
     * This is the main execution entry point of the solver. A typical call performs:
     * - context initialization
     * - field/buffer preparation
     * - per-cell particle-count refresh
     * - density and pressure estimation
     * - acceleration accumulation
     * - in-place particle velocity update
     *
     * The actual neighbor interaction radius is derived from each particle's
     * material properties together with the selected SPH kernel support rule.
     *
     * The update is explicit in time and uses the supplied `dt` directly when
     * converting acceleration into a velocity increment.
     *
     * @param dt Explicit time step for the velocity update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override final;

    /**
     * @brief Reserved codec-aware execution path.
     *
     * This overload is intentionally left blank for future implementation.
     * It exists to preserve interface compatibility with other solver families
     * that support per-cell work partitioning through a solver-assignment buffer.
     *
     * @param allocated_solver Optional per-cell solver-assignment buffer.
     * @param index Solver index used for cell filtering.
     * @param dt Explicit time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override final;

    /**
     * @brief Ensure that all required universe-side states exist.
     *
     * The SPH solver may depend on universe-side fields such as:
     * - cell particle counts
     * - auxiliary scalar/vector fields used by solver diagnostics or coupling
     *
     * This function prepares those states before the main solve pipeline begins.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    /**
     * @brief Initialize the SPH execution context.
     *
     * This function prepares the solver for one SPH step. Typical responsibilities
     * include:
     * - ensuring required universe states exist
     * - allocating or resizing internal particle working buffers
     * - resetting transient universe-side fields
     *
     * @return True when the context is ready and the solve step may proceed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_sph_context() noexcept;

    /**
     * @brief Prepare solver-owned particle working fields.
     *
     * The SPH solver keeps several temporary per-particle buffers:
     * - `_density`
     * - `_pressure`
     * - `_acceleration`
     *
     * This function ensures those buffers are correctly sized and ready for use
     * in the current step.
     *
     * @return True when the particle working fields are available.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_particle_fields();

    /**
     * @brief Reset transient universe-side fields used by the SPH solver.
     *
     * This function clears or reinitializes temporary universe-side state so that
     * the next SPH step starts from a clean baseline.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_universe_fields();

    /**
     * @brief Execute the internal SPH update pipeline.
     *
     * This helper drives the solver-side update stages excluding the final external
     * `solve(dt)` interface. It typically sequences:
     * - cell particle-count update
     * - density and pressure estimation
     * - acceleration accumulation
     *
     * The exact order depends on the implementation in the accompanying `.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    /**
     * @brief Estimate per-particle density and pressure from local neighborhoods.
     *
     * This stage uses the configured smoothing kernel to accumulate density from
     * nearby particles and then derives pressure from:
     * - density
     * - rest density
     * - material pressure coefficient
     *
     * The resulting values are written into `_density` and `_pressure`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_particle_density_and_pressure();

    /**
     * @brief Update the per-cell number-of-particles field in the universe.
     *
     * The SPH solver uses the searcher/universe grid to organize local neighborhoods.
     * This function refreshes the per-cell particle-count information needed for
     * later neighbor iteration and diagnostics.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_cell_number_particles();

    /**
     * @brief Accumulate SPH acceleration and update particle velocities.
     *
     * This stage computes SPH forces from neighboring particles, including terms
     * such as:
     * - pressure force
     * - viscosity force
     *
     * The resulting acceleration is stored in `_acceleration`, then converted
     * into a velocity update using the explicit time step `dt`.
     *
     * @param dt Explicit time step used for velocity integration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    accumulate_acceleration(T dt);

    /**
     * @brief Return the effective smoothing length for a particle/material record.
     *
     * If the material explicitly stores a smoothing length, that value may be used.
     * Otherwise, the universe cell size may act as a fallback reference scale.
     *
     * @param property Particle/material property record.
     * @param cell_size Universe cell size used as a fallback reference.
     * @return Effective smoothing length.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    smoothing_length_for(const MatrialProperties<T>& property, T cell_size) noexcept;

    /**
     * @brief Return the effective rest density for a particle/material record.
     *
     * @param property Particle/material property record.
     * @return Rest density used in pressure evaluation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    rest_density_for(const MatrialProperties<T>& property) noexcept;

    /**
     * @brief Return the effective pressure coefficient for a particle/material record.
     *
     * @param property Particle/material property record.
     * @return Pressure coefficient used in the equation of state or pressure model.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    pressure_coefficient_for(const MatrialProperties<T>& property) noexcept;

    /**
     * @brief Convert a smoothing length into a neighborhood search radius in grid cells.
     *
     * This helper maps a physical smoothing support radius to an integer number of
     * cells that must be scanned around a particle's home cell.
     *
     * @param smoothing_length Effective smoothing length.
     * @param cell_size Universe cell size.
     * @return Neighborhood search radius measured in cells.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    search_radius_for(T smoothing_length, T cell_size) noexcept;

    /**
     * @brief Map a particle position to an integer grid-cell coordinate.
     *
     * @param position Particle position.
     * @param lower_corner Lower corner of the universe domain.
     * @param inverse_cell_size Inverse of the universe cell size.
     * @param grid_size Integer resolution of the universe grid.
     * @return Grid-cell coordinate containing the particle.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<int>
    particle_cell(const Vector3<T>& position,
                  const Vector3<T>& lower_corner,
                  T inverse_cell_size,
                  const Vector3<int>& grid_size) noexcept;

    /**
     * @brief Check whether a neighbor-cell coordinate lies inside the valid grid.
     *
     * @param cell Candidate neighbor-cell coordinate.
     * @param grid_size Integer resolution of the universe grid.
     * @return True when the cell coordinate lies inside the valid grid range.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept;

private:
    /**
     * @brief Runtime-selected SPH smoothing kernel wrapper.
     */
    SphKernel<T> _kernel {};

    /**
     * @brief Solver-owned per-particle density field.
     *
     * This buffer stores the density estimate computed from local smoothing-kernel
     * accumulation during the current solve step.
     */
    DeviceBuffer<T> _density {};

    /**
     * @brief Solver-owned per-particle pressure field.
     *
     * This buffer stores the pressure derived from the density field and material
     * rest-density / pressure-coefficient parameters.
     */
    DeviceBuffer<T> _pressure {};

    /**
     * @brief Solver-owned per-particle acceleration field.
     *
     * This buffer stores the net SPH acceleration accumulated from neighbor
     * interactions before it is integrated into particle velocity.
     */
    DeviceBuffer<Vector3<T>> _acceleration {};
};

/**
 * @brief Builder for `SphSolver`.
 *
 * The builder collects all runtime dependencies required to construct a valid
 * SPH solver instance.
 *
 * Required dependencies:
 * - universe
 * - fluid
 * - searcher
 *
 * Optional configuration:
 * - SPH kernel type
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class SphSolver<T>::Builder final {
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
     * @brief Set the SPH kernel type used by the solver.
     *
     * @param kernel_type Desired SPH kernel type.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(SphKernelType kernel_type) noexcept;

    /**
     * @brief Build a validated `SphSolver` instance.
     *
     * @return Constructed solver.
     *
     * @throws std::runtime_error Thrown when required dependencies are missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SphSolver<T>
    build() const;

    /**
     * @brief Build a validated solver and wrap it in a host-shared pointer.
     *
     * @return Host-shared pointer to the constructed solver.
     *
     * @throws std::runtime_error Thrown when required dependencies are missing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SphSolver<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the builder configuration.
     *
     * Required:
     * - universe must not be null
     * - fluid must not be null
     * - searcher must not be null
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
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for `atlas::system::SphSolver<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphSolver = atlas::system::SphSolver<T>;

/**
 * @brief Host-shared-pointer alias for `SphSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphSolverHostPtr = atlas::host_shared_ptr<atlas::system::SphSolver<T>>;

/**
 * @brief Device-shared-pointer alias for `SphSolver`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphSolverDevicePtr = atlas::device_shared_ptr<atlas::system::SphSolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_solver.hpp>