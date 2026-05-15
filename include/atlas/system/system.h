#pragma once

/**
 * @file system.h
 * @brief Declares the top-level simulation System driver.
 *
 * This file defines atlas::system::System, which coordinates the major runtime
 * subsystems used during one simulation step. The system owns references to
 * fluid state, universe/domain state, particle sources and sinks, optional
 * collider logic, and optional orchestration logic.
 *
 * The System class is intentionally a high-level coordinator. It does not
 * implement source emission, solver logic, collision logic, or sink filtering
 * directly. Instead, it invokes the installed subsystem objects in a fixed
 * update order.
 */

#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Coordinates the major runtime subsystems of a particle simulation.
 *
 * System is the top-level simulation driver. A call to update() advances the
 * simulation by one configured time step @f$\Delta t@f$ using the installed
 * runtime components.
 *
 * The system may coordinate the following subsystems:
 *
 * - Source:
 *   creates, injects, or initializes particles.
 * - Orchestrator:
 *   runs solver-side logic such as spatial search, classification, measurement,
 *   force application, and solver dispatch.
 * - Collider:
 *   advances collider units and performs collider-specific particle advection or
 *   interaction logic.
 * - Sink:
 *   removes particles that satisfy user-defined removal criteria.
 *
 * The only mandatory dependency is the fluid object. The fluid stores particle
 * states such as position and velocity, and therefore all update paths require
 * it. The universe, source, sink, collider, and orchestrator are optional and may
 * be omitted depending on the simulation setup.
 *
 * The default update pipeline is:
 *
 * @f[
 *     \mathrm{emit}
 *     \rightarrow
 *     \mathrm{remove}
 *     \rightarrow
 *     \mathrm{orchestrate}
 *     \rightarrow
 *     \mathrm{advect}.
 * @f]
 *
 * In code, this corresponds to:
 *
 * @code
 * emit();
 * remove();
 * orchestrate();
 * advect();
 * @endcode
 *
 * If a collider is installed, advect() delegates the advection/collision phase
 * to the collider. If no collider is installed, System falls back to direct
 * particle time integration:
 *
 * @f[
 *     \mathbf{x}_{i}^{n+1}
 *     =
 *     \mathbf{x}_{i}^{n}
 *     +
 *     \mathbf{v}_{i}^{n} \Delta t.
 * @f]
 *
 * This fallback path is a first-order explicit Euler position update. It updates
 * particle positions using the current particle velocities and the configured
 * time step. It does not update velocity, apply forces, perform collision
 * handling, or enforce boundary conditions by itself.
 *
 * @tparam T Floating-point scalar type used for simulation quantities.
 *
 * @note T must be a floating-point type.
 * @note System stores shared host pointers to its dependencies. It coordinates
 *       the objects but does not define their internal ownership policy beyond
 *       the pointer aliases used by Atlas.
 * @note Optional subsystems are skipped when their pointer is null.
 * @note The configured time step must be strictly positive when constructed
 *       through Builder.
 *
 * @see Source
 * @see Sink
 * @see Collider
 * @see Orchestrator
 * @see Fluid
 * @see Universe
 */
template <typename T>
class System final {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    /**
     * @brief Builder type used for validated System construction.
     *
     * The builder collects the dependencies required to construct a System and
     * validates the minimum invariants before creating the final object.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty system with no installed runtime dependencies and the
     * default time step value.
     *
     * The resulting object is valid as an empty container, but calling update()
     * will not perform useful simulation work unless the required dependencies
     * are installed later through assignment or construction logic.
     *
     * @note The default time step is initialized to @f$0.01@f$.
     */
    System() = default;

    /**
     * @brief Constructs a simulation system from its runtime dependencies.
     *
     * This constructor installs the fluid, universe, source, sink, collider,
     * orchestrator, and time step used by subsequent calls to update().
     *
     * The fluid pointer is expected to be valid for normal simulation use. Other
     * subsystem pointers may be null, in which case their corresponding update
     * phase is skipped.
     *
     * @param fluid Host shared pointer to the fluid object that stores particle
     *              states.
     * @param universe Host shared pointer to the universe/domain object associated
     *                 with the simulation.
     * @param source Optional source subsystem used during emit().
     * @param sink Optional sink subsystem used during remove().
     * @param collider Optional collider subsystem used during advect().
     * @param orchestrator Optional orchestrator subsystem used during
     *                     orchestrate().
     * @param dt Simulation time step. The default value is @f$0.01@f$.
     *
     * @note This constructor does not perform the same validation as Builder.
     *       Prefer Builder when constructing user-facing systems that must reject
     *       invalid input.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    System(FluidHostPtr<T> fluid,
           UniverseHostPtr<T> universe,
           SourceHostPtr<T> source,
           SinkHostPtr<T> sink,
           ColliderHostPtr<T> collider,
           OrchestratorHostPtr<T> orchestrator,
           T dt = static_cast<T>(0.01));

    /**
     * @brief Destructor.
     *
     * Releases the shared host pointers held by the system according to their
     * normal shared-pointer semantics.
     */
    ~System() = default;

    /**
     * @brief Creates a Builder instance.
     *
     * This function is the preferred entry point for constructing a System with
     * validation.
     *
     * @return Newly created builder object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Executes one complete simulation step.
     *
     * The step is executed in the following order:
     *
     * @f[
     *     \mathrm{source}
     *     \rightarrow
     *     \mathrm{sink}
     *     \rightarrow
     *     \mathrm{orchestrator}
     *     \rightarrow
     *     \mathrm{collider/integration}.
     * @f]
     *
     * More explicitly, update() calls:
     *
     * @code
     * emit();
     * remove();
     * orchestrate();
     * advect();
     * @endcode
     *
     * This ordering allows newly emitted particles to be filtered by the sink
     * before solver orchestration and advection/collision handling run.
     *
     * @note Null optional subsystems are skipped by their corresponding phase
     *       functions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    /**
     * @brief Executes the particle emission phase.
     *
     * If a source subsystem is installed, this function calls:
     *
     * @code
     * source->update(dt)
     * @endcode
     *
     * where @c dt is the configured system time step. If no source is installed,
     * the function returns without modifying the simulation state.
     *
     * @note The exact meaning of emission is defined by the installed Source
     *       implementation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    /**
     * @brief Executes the orchestration phase.
     *
     * If an orchestrator subsystem is installed, this function calls:
     *
     * @code
     * orchestrator->update(dt)
     * @endcode
     *
     * The orchestrator is responsible for solver-side coordination such as
     * spatial search construction, particle classification, measurement,
     * force application, and solver dispatch.
     *
     * If no orchestrator is installed, this function returns without modifying
     * the simulation state.
     *
     * @note The exact orchestration pipeline is defined by Orchestrator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate();

    /**
     * @brief Executes the advection or collider phase.
     *
     * If a collider subsystem is installed, this function delegates the phase to
     * the collider:
     *
     * @code
     * collider->update(dt)
     * @endcode
     *
     * If no collider is installed, this function falls back to direct time
     * integration through time_integration().
     *
     * The fallback integration updates particle positions as:
     *
     * @f[
     *     \mathbf{x}_{i}^{n+1}
     *     =
     *     \mathbf{x}_{i}^{n}
     *     +
     *     \mathbf{v}_{i}^{n} \Delta t.
     * @f]
     *
     * @note Collider-driven advection may include geometry interaction,
     *       collision handling, or custom motion logic depending on the collider
     *       implementation.
     * @note The fallback path only updates positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    advect();

    /**
     * @brief Executes the particle removal phase.
     *
     * If a sink subsystem is installed, this function calls:
     *
     * @code
     * sink->update(dt)
     * @endcode
     *
     * The sink may remove particles based on position, domain membership,
     * lifetime, boundary crossing, or any other criterion implemented by the
     * concrete sink type.
     *
     * If no sink is installed, this function returns without modifying the
     * simulation state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    /**
     * @brief Performs fallback particle advection using explicit Euler integration.
     *
     * This function is used by advect() when no collider subsystem is installed.
     * It directly updates the fluid position state using the fluid velocity state.
     *
     * For each particle @f$i@f$, the update is:
     *
     * @f[
     *     \mathbf{x}_{i}
     *     \leftarrow
     *     \mathbf{x}_{i}
     *     +
     *     \mathbf{v}_{i} \Delta t.
     * @f]
     *
     * where:
     *
     * - @f$\mathbf{x}_{i}@f$ is the particle position,
     * - @f$\mathbf{v}_{i}@f$ is the particle velocity,
     * - @f$\Delta t@f$ is the configured system time step.
     *
     * The function exits without work if:
     *
     * - the fluid pointer is null,
     * - the time step is not strictly positive,
     * - the fluid has no position state,
     * - the fluid has no velocity state,
     * - the position or velocity buffer is empty,
     * - the particle count is zero.
     *
     * @note This function performs only position advection. It does not update
     *       velocity, acceleration, solver state, collision state, or sink state.
     * @note The update is executed with the Atlas device parallel-for backend.
     * @note The function assumes that the position and velocity buffers are valid
     *       for all particle indices in the range @f$[0, N)@f$, where @f$N@f$ is
     *       the fluid particle count.
     *
     * @see advect()
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration();

    /**
     * @brief Returns the configured simulation time step.
     *
     * The time step is the scalar value passed to source, orchestrator, collider,
     * sink, and fallback time integration updates.
     *
     * @return Configured time step value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    dt() const noexcept;

    /**
     * @brief Returns the installed fluid object.
     *
     * The returned reference refers to the internal host shared pointer stored by
     * the system.
     *
     * @return Const reference to the host shared pointer owning the fluid.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

private:
    /**
     * @brief Target fluid object processed by the system.
     *
     * The fluid stores particle state arrays such as positions, velocities, and
     * other simulation quantities. A valid fluid object is required for normal
     * system operation.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Universe/domain object associated with the system.
     *
     * The universe may provide domain information, cell layout, field states,
     * gravity states, or other global simulation data used by installed
     * subsystems.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Optional particle source subsystem.
     *
     * When installed, this subsystem is invoked during emit() to create or inject
     * particles.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Optional particle sink subsystem.
     *
     * When installed, this subsystem is invoked during remove() to delete or
     * deactivate particles according to sink-specific rules.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Optional collider subsystem.
     *
     * When installed, this subsystem is invoked during advect(). If absent, the
     * system uses direct fallback time integration instead.
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Optional orchestrator subsystem.
     *
     * When installed, this subsystem is invoked during orchestrate() to coordinate
     * solver-related simulation work.
     */
    OrchestratorHostPtr<T> _orchestrator {};

    /**
     * @brief Simulation time step.
     *
     * This value is passed to each phase of the simulation update. Builder
     * validation requires it to be strictly positive.
     */
    T _dt { static_cast<T>(0.01) };

    /**
     * @brief Cached raw pointers to the fluid position and velocity states.
     *
     * Resolved once at construction from the fluid state registry to avoid
     * repeated unordered_map lookups during time_integration().  The pointed-to
     * state objects are owned by the fluid and remain valid for the lifetime of
     * the system.
     */
    fluid::FluidPositionState<T>* _cached_position_state {};
    fluid::FluidVelocityState<T>* _cached_velocity_state {};
};

/**
 * @brief Builder for validated System construction.
 *
 * The builder collects the dependencies required to construct a System and
 * validates the minimum invariants before the final object is created.
 *
 * Required invariants:
 *
 * - the fluid pointer must be non-null,
 * - the time step must be strictly positive.
 *
 * Optional dependencies:
 *
 * - universe,
 * - source,
 * - sink,
 * - collider,
 * - orchestrator.
 *
 * A typical construction pattern is:
 *
 * @code
 * auto system = atlas::System<float>::builder()
 *                   .with_fluid(fluid)
 *                   .with_domain(universe)
 *                   .with_source(source)
 *                   .with_sink(sink)
 *                   .with_collider(collider)
 *                   .with_solver(orchestrator)
 *                   .with_dt(1.0e-4f)
 *                   .build();
 * @endcode
 *
 * @tparam T Floating-point scalar type used for simulation quantities.
 *
 * @note Builder performs validation in build() and make_host_shared().
 * @note Optional subsystems may be omitted.
 */
template <typename T>
class System<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * Creates an empty builder with no installed dependencies and a default time
     * step value.
     */
    Builder() = default;

    /**
     * @brief Installs the fluid object.
     *
     * The fluid object is the only mandatory runtime dependency. It stores the
     * particle states operated on by the system and its subsystems.
     *
     * @param fluid Host shared pointer to the fluid object.
     * @return Reference to this builder for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(const FluidHostPtr<T>& fluid) noexcept;

    /**
     * @brief Installs the universe/domain object.
     *
     * The universe may provide domain geometry, cell layout, field states, gravity
     * states, or other global data used by the orchestrator, collider, source, or
     * sink.
     *
     * @param universe Host shared pointer to the universe object.
     * @return Reference to this builder for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const UniverseHostPtr<T>& universe) noexcept;

    /**
     * @brief Installs the source subsystem.
     *
     * The source is invoked during emit() and may create or initialize particles
     * using the configured simulation time step.
     *
     * @param source Host shared pointer to the source subsystem.
     * @return Reference to this builder for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source) noexcept;

    /**
     * @brief Installs the sink subsystem.
     *
     * The sink is invoked during remove() and may remove or deactivate particles
     * according to sink-specific criteria.
     *
     * @param sink Host shared pointer to the sink subsystem.
     * @return Reference to this builder for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink) noexcept;

    /**
     * @brief Installs the collider subsystem.
     *
     * The collider is invoked during advect(). If installed, it replaces the
     * fallback time integration path. If omitted, the system uses direct explicit
     * Euler position integration.
     *
     * @param collider Host shared pointer to the collider subsystem.
     * @return Reference to this builder for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider) noexcept;

    /**
     * @brief Installs the orchestrator subsystem.
     *
     * The orchestrator is invoked during orchestrate() and usually coordinates
     * solver-related operations such as spatial search, classification,
     * measurement, force application, and solver execution.
     *
     * @param orchestrator Host shared pointer to the orchestrator subsystem.
     * @return Reference to this builder for chained configuration.
     *
     * @note The method name is with_solver() for API compatibility, but the
     *       installed object is an Orchestrator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(const OrchestratorHostPtr<T>& orchestrator) noexcept;

    /**
     * @brief Sets the simulation time step.
     *
     * The time step is passed to each update phase and is used in fallback time
     * integration.
     *
     * @param dt Positive time step value.
     * @return Reference to this builder for chained configuration.
     *
     * @note Validation rejects non-positive values during build() and
     *       make_host_shared().
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    /**
     * @brief Validates the builder state and constructs a System object.
     *
     * This function checks the required invariants and returns a System by value.
     *
     * @return Constructed System object.
     *
     * @throws std::runtime_error Thrown if the fluid pointer is null.
     * @throws std::runtime_error Thrown if the time step is not strictly positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build() const;

    /**
     * @brief Builds a System object and wraps it in host-managed shared storage.
     *
     * This function performs the same validation as build(), then constructs the
     * System in an Atlas host shared pointer.
     *
     * @return Host shared pointer to the constructed System.
     *
     * @throws std::runtime_error Thrown if the fluid pointer is null.
     * @throws std::runtime_error Thrown if the time step is not strictly positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder configuration.
     *
     * Validation enforces the minimum invariants required for a usable System:
     *
     * - the fluid pointer must be valid,
     * - the time step must be strictly positive.
     *
     * Optional subsystems are not required.
     *
     * @throws std::runtime_error Thrown if the fluid pointer is null.
     * @throws std::runtime_error Thrown if the time step is not strictly positive.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Builder-owned fluid object.
     *
     * This dependency is required. build() and make_host_shared() reject a null
     * fluid pointer.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Builder-owned universe object.
     *
     * This dependency is optional and may remain null.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Builder-owned source subsystem.
     *
     * This dependency is optional and may remain null.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Builder-owned sink subsystem.
     *
     * This dependency is optional and may remain null.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Builder-owned collider subsystem.
     *
     * This dependency is optional. If omitted, System uses fallback time
     * integration during advect().
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Builder-owned orchestrator subsystem.
     *
     * This dependency is optional and may remain null.
     */
    OrchestratorHostPtr<T> _orchestrator {};

    /**
     * @brief Builder-owned time step.
     *
     * This value defaults to @f$0.01@f$ and must be strictly positive when the
     * final System is built.
     */
    T _dt { static_cast<T>(0.01) };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::System.
 *
 * This alias exposes the system type directly in the atlas namespace.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using System = system::System<T>;

/**
 * @brief Host shared pointer alias for System.
 *
 * This alias represents a host-managed shared pointer to atlas::system::System.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

} // namespace atlas

#include <atlas/system/system.hpp>
