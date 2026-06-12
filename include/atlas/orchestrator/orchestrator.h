#pragma once

/**
 * @file orchestrator.h
 * @brief Declares the atlas::Orchestrator class and its fluent Builder.
 *
 * The orchestrator is the high-level simulation-step coordinator. It connects
 * the universe, fluid, spatial searcher, optional codec, optional measurer, and
 * an ordered sequence of solvers into one update pipeline.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/detail/orchestrator_force_applier.h>
#include <atlas/orchestrator/detail/orchestrator_pipeline.h>
#include <atlas/orchestrator/detail/orchestrator_probe_builder.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

#include <optional>

namespace atlas {

/**
 * @brief Coordinates search, classification, measurement, force application, and solver execution.
 *
 * `Orchestrator` owns no simulation data directly. Instead, it stores host-side
 * shared pointers to the simulation systems it coordinates:
 *
 * - a universe, used for cell-level states such as field force and gravity,
 * - a fluid, used for particle velocity, species, and material-property data,
 * - a spatial hashing searcher, used to map particles to cell ranges,
 * - an optional codec, used to enable codec-aware solver dispatch,
 * - an optional measurer, used to collect simulation measurements,
 * - an ordered list of solvers.
 *
 * A full update step is performed by @ref update or @ref orchestrate. The
 * implementation executes the following pipeline:
 *
 * @code
 * _search.run(orchestrator);
 * _classification.run(orchestrator);
 * _measurement.run(orchestrator, dt);
 * _force.run(orchestrator, dt);
 * _solve.run(orchestrator, dt);
 * @endcode
 *
 * Missing optional dependencies are handled defensively. For example, if no
 * searcher is configured, search is skipped; if no codec is configured, solvers
 * are executed through their non-codec path; if force-related states are absent,
 * force application becomes a no-op.
 *
 * @tparam T Scalar type used by the simulation, for example `float` or `double`.
 */
template <typename T>
class Orchestrator final {
    template <typename>
    friend struct detail::OrchestratorForceStage;

public:
    /**
     * @brief Fluent builder for constructing validated `Orchestrator` instances.
     *
     * The builder collects the same dependencies stored by `Orchestrator`.
     * During @ref build and @ref make_host_shared it validates the solver list
     * and materializes staged gravity, if any, into the bound universe.
     */
    class Builder;

public:
    /**
     * @brief Constructs an empty orchestrator.
     *
     * All dependencies are initialized to null-equivalent values and the solver
     * list is empty. Calling @ref update or @ref orchestrate on an empty
     * orchestrator is safe and results in no solver execution or force update.
     */
    Orchestrator() = default;

    /**
     * @brief Destroys the orchestrator.
     */
    ~Orchestrator() = default;

    /**
     * @brief Constructs an orchestrator from explicit pipeline dependencies.
     *
     * This constructor stores the provided host-side shared pointers and solver
     * list by move. It performs no validation and does not install gravity state;
     * those responsibilities belong to @ref Builder when the builder is used.
     *
     * @param universe Universe used to access cell-wise field-force and gravity states.
     * @param fluid Fluid used to access particle velocity, species, and material properties.
     * @param searcher Spatial hashing searcher used to build cell-to-particle ranges.
     * @param codec Optional codec enabling codec-aware solver dispatch.
     * @param measurer Optional measurer invoked during the measurement stage.
     * @param solvers Ordered list of solvers invoked during @ref solve.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Orchestrator(UniverseHostPtr<T> universe,
                 FluidHostPtr<T> fluid,
                 SpatialHashingSearcherHostPtr<T> searcher,
                 CodecHostPtr<T> codec,
                 MeasurerHostPtr<T> measurer,
                 HostBuffer<SolveHostPtr<T>> solvers) noexcept;

    /**
     * @brief Builds spatial search data when a searcher is configured.
     *
     * If `_searcher` is non-null, this function calls `build()` on it. If no
     * searcher is configured, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    search();

    /**
     * @brief Updates codec classification when a codec is configured.
     *
     * If `_codec` is non-null, this function calls `update()` on it. This stage
     * is intended to prepare codec-side allocation or classification data before
     * codec-aware solver dispatch. If no codec is configured, the function is a
     * no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify();

    /**
     * @brief Runs the configured measurer, if present.
     *
     * If `_measurer` is non-null, this function calls `measure()` on it. If no
     * measurer is configured, the function is a no-op.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure();

    /**
     * @brief Runs the configured measurer with the current time step.
     *
     * If `_measurer` is non-null, this function calls `measure(dt)` on it. If no
     * measurer is configured, the function is a no-op.
     *
     * @param dt Simulation time step passed to time-aware measurers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt);

    /**
     * @brief Executes all configured solvers for one time step.
     *
     * Solver dispatch depends on whether a codec is configured:
     *
     * - without a codec, each non-null solver is invoked as `solver->solve(dt)`;
     * - with a codec, each non-null solver is invoked as
     *   `solver->solve(&_codec->allocated_solver(), solver_index, dt)`.
     *
     * Null solver entries are skipped defensively. An empty solver list causes
     * an immediate return.
     *
     * @param dt Time-step size forwarded to each solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt);

    /**
     * @brief Runs one full simulation orchestration step.
     *
     * This is the public high-level entry point used by system-level code. It
     * forwards directly to @ref orchestrate.
     *
     * @param dt Time-step size used by gravity, field-force, and solver stages.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Refreshes the cached raw-pointer probe for force-application kernels.
     *
     * This function resolves the currently configured universe, fluid, and
     * searcher into raw data views used by @ref apply_gravity and
     * @ref apply_field_force.
     *
     * A valid probe requires:
     *
     * - non-null universe, fluid, and searcher dependencies,
     * - a non-null `FluidVelocityState<T>`,
     * - non-empty velocity data,
     * - positive particle and cell counts,
     * - non-null searcher `indices`, `cell_start`, and `cell_end` arrays.
     *
     * Optional fields are populated when their corresponding states/data exist:
     *
     * - `species_ptr`, `properties_ptr`, and `num_of_species` for field force,
     * - `field_force_ptr` and `field_force_cell_count`,
     * - `gravity_ptr` and `gravity_cell_count`.
     *
     * @retval true The common velocity and searcher data required for force
     *              application was found.
     * @retval false Required common data was missing; the caller should skip
     *               force application.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    /**
     * @brief Creates an empty fluent builder.
     *
     * @return Builder object used to configure and construct an orchestrator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Runs the complete orchestration pipeline for one simulation step.
     *
     * The execution order is:
     *
     * @code
     * _search.run(orchestrator);
     * _classification.run(orchestrator);
     * _measurement.run(orchestrator, dt);
     * _force.run(orchestrator, dt);
     * _solve.run(orchestrator, dt);
     * @endcode
     *
     * The searcher is invalidated before rebuilding search data. Gravity and
     * field-force application runs after measurement and before solver execution.
     *
     * @param dt Time-step size used by force-application and solver stages.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate(T dt);

    /**
     * @brief Sets or replaces the universe dependency.
     *
     * The universe supplies cell-level states such as
     * `UniverseFieldForceState<T>` and `UniverseGravityState<T>`.
     *
     * @param universe Host-side shared pointer to the universe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets or replaces the fluid dependency.
     *
     * The fluid supplies particle velocity state, species state, particle count,
     * and material properties used by force application.
     *
     * @param fluid Host-side shared pointer to the fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets or replaces the spatial hashing searcher.
     *
     * The searcher supplies sorted particle indices and cell range boundaries
     * used by force-application passes.
     *
     * @param searcher Host-side shared pointer to the spatial hashing searcher.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets or replaces the codec dependency.
     *
     * When a codec is configured, @ref classify updates it and @ref solve uses
     * codec-aware solver dispatch.
     *
     * @param codec Host-side shared pointer to the codec.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(CodecHostPtr<T> codec) noexcept;

    /**
     * @brief Sets or replaces the measurer dependency.
     *
     * When configured, the measurer is invoked during @ref measure.
     *
     * @param measurer Host-side shared pointer to the measurer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measurer(MeasurerHostPtr<T> measurer) noexcept;

    /**
     * @brief Appends a solver to the solver execution list.
     *
     * This mutator does not validate the pointer. Null solvers appended through
     * this function are skipped by @ref solve. Builder-based construction,
     * however, rejects null solver entries during validation.
     *
     * @param solver Host-side shared pointer to the solver to append.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Returns the configured spatial hashing searcher.
     *
     * @return Const reference to the stored searcher shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SpatialHashingSearcherHostPtr<T>&
    searcher() const noexcept;

    /**
     * @brief Returns the configured universe.
     *
     * @return Const reference to the stored universe shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const UniverseHostPtr<T>&
    universe() const noexcept;

    /**
     * @brief Returns the configured fluid.
     *
     * @return Const reference to the stored fluid shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    /**
     * @brief Returns the configured codec.
     *
     * @return Const reference to the stored codec shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    /**
     * @brief Returns the configured measurer.
     *
     * @return Const reference to the stored measurer shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MeasurerHostPtr<T>&
    measurer() const noexcept;

    /**
     * @brief Returns the configured solver list.
     *
     * @return Const reference to the ordered host buffer of solver pointers.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SolveHostPtr<T>>&
    solvers() const noexcept;

    /**
     * @brief Applies all configured cell forces to probe particles in one pass.
     *
     * Gravity and field-force states share the same cell-to-particle ranges, so
     * this fused path avoids launching separate passes over identical sorted
     * ranges during @ref orchestrate.
     *
     * @param probe Pre-built data view.
     * @param dt Time-step size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_forces(const OrchestratorProbe<T>& probe, T dt);

    /**
     * @brief Applies gravity to all particles in the probe using a pre-built probe.
     *
     * This low-level entry point is kept for callers that need the standalone
     * gravity pass. The normal orchestration pipeline uses @ref apply_forces so
     * gravity and field force can share one traversal.
     *
     * @param probe Pre-built data view. Must have a valid `gravity_ptr`.
     * @param dt Time-step size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_gravity(const OrchestratorProbe<T>& probe, T dt);

    /**
     * @brief Applies field forces to all particles in the probe using a pre-built probe.
     *
     * This low-level entry point is kept for callers that need the standalone
     * field-force pass. The normal orchestration pipeline uses @ref apply_forces
     * so gravity and field force can share one traversal.
     *
     * @param probe Pre-built data view. Must have valid `field_force_ptr`,
     *              `species_ptr`, `properties_ptr`, and positive `num_of_species`.
     * @param dt Time-step size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(const OrchestratorProbe<T>& probe, T dt);

private:
    /**
     * @brief Applies force states using the cached probe produced by make_probe().
     *
     * The probe is copied before launch so device kernels receive a stable value
     * object independent of later host-side probe updates.
     *
     * @param dt Time-step size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_probe_forces(T dt);

private:
    /**
     * @brief Universe dependency used for cell-wise orchestration states.
     *
     * May be null. Required by gravity and field-force application.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Fluid dependency used for particle state and material data.
     *
     * May be null. Required by gravity and field-force application.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Spatial searcher used to build and expose cell-to-particle ranges.
     *
     * May be null. Required by search and force-application stages.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Optional codec used for classification and codec-aware solver dispatch.
     *
     * When null, solvers are dispatched through their non-codec solve path.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Optional measurer invoked during the measurement stage.
     */
    MeasurerHostPtr<T> _measurer {};

    /**
     * @brief Ordered solver list executed during the solve stage.
     *
     * Solvers are invoked in buffer order. Null entries are skipped by @ref solve.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};

    /**
     * @brief Fixed solver-side orchestration pipeline.
     *
     * This stateless object names the search, classification, measurement,
     * force, and solve stages in their execution order.
     */
    detail::OrchestratorPipeline<T> _pipeline {};

    /**
     * @brief Cached probe populated by @ref make_probe.
     */
    OrchestratorProbe<T> _probe {};

    /**
     * @brief Builds raw-pointer probes from runtime dependencies.
     */
    detail::OrchestratorProbeBuilder<T> _probe_builder {};

    /**
     * @brief Applies force-related universe states to fluid velocity.
     */
    detail::OrchestratorForceApplier<T> _force_applier {};
};

/**
 * @brief Fluent builder for `Orchestrator`.
 *
 * The builder collects orchestrator dependencies and solver entries, then
 * constructs either a value object or a host-side shared object.
 *
 * During construction, the builder:
 *
 * - validates that every configured solver pointer is non-null,
 * - installs staged gravity into the universe when @ref with_gravity was used,
 * - returns an orchestrator initialized with the collected dependencies.
 *
 * Staged gravity requires a bound universe with a positive number of cells. If a
 * gravity state already exists, it is resized to match the universe cell count
 * if necessary and filled with the staged gravity vector. If it does not exist,
 * the builder creates `UniverseGravityState<T>` and fills it.
 *
 * @tparam T Scalar type used by the orchestrator.
 */
template <typename T>
class Orchestrator<T>::Builder final {
public:
    /**
     * @brief Constructs an empty builder.
     */
    Builder() = default;

    /**
     * @brief Sets the universe dependency to be stored in the orchestrator.
     *
     * The universe is also used by @ref with_gravity during @ref build or
     * @ref make_host_shared to create or update `UniverseGravityState<T>`.
     *
     * @param universe Host-side shared pointer to the universe.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    /**
     * @brief Sets the fluid dependency to be stored in the orchestrator.
     *
     * @param fluid Host-side shared pointer to the fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Sets the spatial hashing searcher dependency.
     *
     * The searcher is used to build search data and to expose cell ranges during
     * gravity and field-force passes.
     *
     * @param searcher Host-side shared pointer to the spatial hashing searcher.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    /**
     * @brief Sets the optional codec dependency.
     *
     * A configured codec enables classification and codec-aware solver dispatch.
     *
     * @param codec Host-side shared pointer to the codec.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(CodecHostPtr<T> codec) noexcept;

    /**
     * @brief Sets the optional measurer dependency.
     *
     * @param measurer Host-side shared pointer to the measurer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measurer(MeasurerHostPtr<T> measurer) noexcept;

    /**
     * @brief Stages a uniform gravity vector for installation during build.
     *
     * The value is not applied immediately. During @ref build or
     * @ref make_host_shared, the builder ensures that the bound universe contains
     * a `UniverseGravityState<T>` with one entry per universe cell and fills that
     * buffer with this gravity vector.
     *
     * If no universe is configured, or if the universe has zero cells, the staged
     * gravity value has no effect during construction.
     *
     * @param gravity Gravity acceleration vector to assign to each universe cell.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_gravity(const Vector3<T>& gravity) noexcept;

    /**
     * @brief Appends a solver to the orchestrator configuration.
     *
     * Solver order is preserved and used by @ref Orchestrator::solve. Null solver
     * entries are rejected by @ref build and @ref make_host_shared.
     *
     * @param solver Host-side shared pointer to the solver to append.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Builds a validated orchestrator value.
     *
     * This function validates the solver list, materializes staged gravity into
     * the universe when requested, and returns an orchestrator initialized with
     * the collected dependencies.
     *
     * @return Constructed orchestrator.
     *
     * @throw std::runtime_error If any configured solver pointer is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    /**
     * @brief Builds a validated host-side shared orchestrator.
     *
     * This function performs the same validation and staged-gravity installation
     * as @ref build, then constructs the orchestrator with
     * `atlas::make_host_shared`.
     *
     * @return Host-side shared pointer to the constructed orchestrator.
     *
     * @throw std::runtime_error If any configured solver pointer is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the collected builder state.
     *
     * Current validation requires every configured solver pointer to be non-null.
     *
     * @throw std::runtime_error If any solver pointer is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

    /**
     * @brief Creates or updates universe gravity state from staged gravity.
     *
     * If a universe and staged gravity vector are available, this function
     * ensures that `UniverseGravityState<T>` exists, has one element per universe
     * cell, and is filled with the staged gravity vector using device
     * `parallel_fill`.
     *
     * The function is a no-op when no universe is configured, no gravity was
     * staged, or the universe reports zero cells.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_gravity_state() const;

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
     * @brief Spatial hashing searcher collected by the builder.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Optional codec collected by the builder.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Optional measurer collected by the builder.
     */
    MeasurerHostPtr<T> _measurer {};

    /**
     * @brief Optional gravity vector staged for build-time universe state setup.
     */
    std::optional<Vector3<T>> _gravity {};

    /**
     * @brief Ordered solver list collected by the builder.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for `atlas::Orchestrator`.
 *
 * @tparam T Scalar type used by the orchestrator.
 */
/**
 * @brief Host-side shared pointer alias for `atlas::Orchestrator`.
 *
 * @tparam T Scalar type used by the orchestrator.
 */
template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::Orchestrator<T>>;

/**
 * @brief Device-side shared pointer alias for `atlas::Orchestrator`.
 *
 * @tparam T Scalar type used by the orchestrator.
 */
template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::Orchestrator<T>>;

} // namespace atlas

#include <atlas/orchestrator/orchestrator.hpp>
