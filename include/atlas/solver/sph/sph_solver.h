#pragma once

/**
 * @file sph_solver.h
 * @brief Declares the SPH solver used to estimate particle fields and update velocities.
 *
 * This header defines `atlas::SphSolver<T>`, a smooth-particle-
 * hydrodynamics style solver that computes density, pressure, acceleration, and
 * cell-level force/particle-count fields from local particle neighborhoods.
 */

#include <atlas/core/macros.h>
#include <atlas/solver/solver.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>

namespace atlas {

/**
 * @brief Smooth-particle-hydrodynamics solver using searcher-provided neighbors.
 *
 * `SphSolver<T>` derives from @ref Solver and implements an explicit SPH-style
 * velocity update. It uses the searcher cell size as the SPH kernel support,
 * consumes the neighbor list built by the searcher, computes per-particle
 * density and pressure, then accumulates pressure-gradient and viscosity terms
 * into an acceleration field.
 *
 * The primary execution path is @ref solve(T), which performs:
 *
 * @code
 * if (!initialize_context()) return;
 * if (!(dt > 0)) throw std::invalid_argument(...);
 * if (!prepare_fields()) {
 *     reset_fields();
 *     return;
 * }
 * make_probe();
 * update();
 * accelerate(dt);
 * @endcode
 *
 * The internal @ref update stage executes:
 *
 * @code
 * estimate_density(probe);
 * count_particles(probe);
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
    using SphSolverProbe = atlas::SphProbe<T>;

    /**
     * @brief Fluent builder for constructing validated `SphSolver` instances.
     */
    class Builder;

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
     * `kernel_type`, and calls @ref ensure_states.
     *
     * @param universe Universe containing SPH per-cell output states.
     * @param fluid Fluid containing particle position, velocity, species, and material data.
     * @param searcher Spatial hashing searcher used for cell and neighbor traversal.
     * @param kernel_type SPH kernel model used for smoothing operations.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphSolver(UniverseHostPtr<T> universe,
              FluidHostPtr<T> fluid,
              SearcherHostPtr<T> searcher,
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
     * Required dependencies and states are checked by @ref initialize_context
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
     * If a universe is configured, this function creates or resizes the
     * following states to `universe->number_of_cells()` elements:
     *
     * - `UniverseNumberParticleState<T>`,
     * - `UniverseFieldForceState<T>`.
     *
     * If no universe is configured, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

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
    initialize_context() noexcept;

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
     * @brief Allocates per-particle working fields for the current step.
     *
     * The working buffers are resized to `fluid->particle_count()`:
     *
     * - `_density`,
     * - `_pressure`,
     * - `_acceleration`.
     *
     * Active entries are overwritten by the density and acceleration stages.
     * Avoiding an unconditional clear removes three full-buffer device writes per
     * solve step.
     *
     * Universe output fields are reset by calling @ref reset_fields.
     *
     * If the particle count is not positive, all working buffers are cleared,
     * universe fields are reset, and the function returns `false`.
     *
     * @retval true Working fields were allocated.
     * @retval false Particle count was zero or negative.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE bool
    prepare_fields();

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
    reset_fields();

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
    rest_density(const MaterialProperties<T>& property) noexcept;

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
    pressure_coefficient(const MaterialProperties<T>& property) noexcept;

    /**
     * @brief Estimates per-particle density and pressure using the cached SPH probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    estimate_density();

    /**
     * @brief Updates per-cell particle counts using the cached SPH probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    count_particles();

    /**
     * @brief Accumulates SPH acceleration and updates velocities using the cached SPH probe.
     *
     * @param dt Positive time-step size used for explicit velocity integration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    accelerate(T dt);

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
     * Resized by @ref prepare_fields, then populated by
     * @ref estimate_density.
     */
    DeviceBuffer<T> _density {};

    /**
     * @brief Solver-owned per-particle pressure working buffer.
     *
     * Resized by @ref prepare_fields, then populated by
     * @ref estimate_density.
     */
    DeviceBuffer<T> _pressure {};

    /**
     * @brief Solver-owned per-particle acceleration working buffer.
     *
     * Resized by @ref prepare_fields, then populated by
     * @ref accelerate.
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
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

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
    SearcherHostPtr<T> _searcher {};

    /**
     * @brief Kernel type collected by the builder.
     */
    SphKernelType _kernel_type { SphKernelType::standard };
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for `atlas::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
/**
 * @brief Host-side shared pointer alias for `atlas::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
using SphSolverHostPtr = atlas::host_shared_ptr<atlas::SphSolver<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::SphSolver`.
 *
 * @tparam T Scalar type used by the SPH solver.
 */
template <typename T>
using SphSolverDevicePtr = atlas::device_shared_ptr<atlas::SphSolver<T>>;

} // namespace atlas

#include <atlas/solver/sph/sph_solver.hpp>
