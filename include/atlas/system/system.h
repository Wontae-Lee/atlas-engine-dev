#pragma once

/**
 * @file system.h
 * @brief Declares the top-level simulation runtime object that orchestrates Atlas subsystems.
 *
 * @details
 * This header defines @ref atlas::system::System, the high-level runtime object
 * responsible for tying together the major subsystems involved in an Atlas
 * simulation step.
 *
 * A @ref System typically owns or references:
 * - particle/fluid storage,
 * - simulation domain data,
 * - a spatial search structure,
 * - a codec for representation transforms,
 * - optional source and sink stages,
 * - optional measurement and diagnostics stages,
 * - optional collider handling,
 * - an optional solver/orchestrator stage.
 *
 * ## Core responsibility
 * The main role of @ref System is to provide a single orchestration point for
 * the simulation loop. Rather than requiring application code to manually invoke
 * each subsystem in the correct order, the system exposes:
 * - a full-step entry point via @ref System::update,
 * - individual stage entry points such as @ref emit, @ref search, @ref solve,
 *   and @ref collide,
 * - accessors for the canonical device probes used by backend execution code.
 *
 * ## Canonical probe ownership
 * In addition to holding subsystem objects, the system caches the single
 * authoritative runtime device probes for:
 * - the fluid,
 * - the domain,
 * - the spatial searcher,
 * - the codec.
 *
 * This allows backend code to operate on stable, already-built runtime views
 * instead of rebuilding them repeatedly at each stage.
 *
 * ## Typical staged flow
 * A full simulation step generally follows a staged order similar to:
 * 1. emit new particles or state,
 * 2. rebuild spatial search structures,
 * 3. classify or encode simulation state,
 * 4. measure diagnostics,
 * 5. solve or orchestrate simulation updates,
 * 6. resolve collisions,
 * 7. remove particles or state via sinks,
 * 8. advance time integration.
 *
 * The exact meaning of each stage depends on the implementation in
 * `system.hpp` and on the configured subsystems.
 *
 * ## Optional subsystems
 * Several subsystems are optional. When not provided, the corresponding stage
 * may become a no-op while the rest of the system remains usable.
 *
 * ## Construction
 * A system may be:
 * - constructed directly with only a fluid object,
 * - constructed with a fully specified set of subsystems, or
 * - created through the nested fluent @ref Builder, which stages configuration
 *   and validates it before construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout the simulation runtime.
 */

#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measure.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief High-level runtime object that orchestrates an Atlas simulation step.
 *
 * @details
 * @ref System acts as the main integration point for the major runtime
 * subsystems of a simulation.
 *
 * It combines:
 * - state storage through @ref Fluid,
 * - domain/grid metadata through @ref Domain,
 * - neighborhood queries through a spatial hashing searcher,
 * - optional state transformation through a @ref Codec,
 * - optional particle creation/removal through @ref Source and @ref Sink,
 * - optional diagnostics through @ref Measure,
 * - optional boundary handling through @ref Collider,
 * - optional solver coordination through @ref Orchestrator.
 *
 * ## Execution model
 * The system can execute a full step through @ref update, or expose the
 * individual stages separately so applications can:
 * - run the built-in pipeline as-is,
 * - interleave their own logic between stages,
 * - selectively disable or replace individual pieces of the step flow.
 *
 * ## Canonical runtime views
 * To support backend execution efficiently, the system stores the single
 * authoritative device probes for its core active components:
 * - @ref _particle_probe
 * - @ref _domain_probe
 * - @ref _searcher_probe
 * - @ref _codec_probe
 *
 * These probes are intended to be reused across stage calls so runtime views
 * remain stable and do not need to be recreated repeatedly.
 *
 * ## Ownership model
 * The system primarily stores host-side shared pointers to its subsystems.
 * This allows:
 * - shared ownership across application/runtime layers,
 * - replacement of subsystems at runtime via setter functions,
 * - optional omission of non-essential stages.
 *
 * ## Searcher derivation
 * The spatial searcher is stored as a derived/internal subsystem because it is
 * typically rebuilt or refreshed from the current fluid/domain state and then
 * exposed through @ref searcher_probe().
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class System {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref System.
     *
     * @details
     * The builder stages the core and optional runtime subsystems, validates the
     * configuration, and then materializes either:
     * - a value instance of @ref System, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

public:
    /**
     * @brief Construct a system with only a fluid subsystem.
     *
     * @details
     * This constructor provides the minimal entry point for creating a runtime
     * system around an existing fluid object.
     *
     * Additional subsystems such as the domain, codec, collider, and solver may
     * be attached later through the corresponding setter functions.
     *
     * @param fluid Host-side shared pointer to the fluid subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(FluidHostPtr<T> fluid);

    /**
     * @brief Construct a fully configured system.
     *
     * @details
     * Initializes the system with its core timestep, simulation state, and the
     * set of configured optional subsystems required for staged execution.
     *
     * This constructor is appropriate when the runtime configuration is already
     * known up front and no staged builder workflow is desired.
     *
     * @param fluid Host-side shared pointer to the fluid subsystem.
     * @param dt Simulation timestep.
     * @param domain Host-side shared pointer to the domain subsystem.
     * @param codec Host-side shared pointer to the codec subsystem.
     * @param source Optional host-side shared pointer to the source subsystem.
     * @param sink Optional host-side shared pointer to the sink subsystem.
     * @param measure Optional host-side shared pointer to the measurement subsystem.
     * @param collider Optional host-side shared pointer to the collider subsystem.
     * @param solver Optional host-side shared pointer to the solver/orchestrator subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    System(FluidHostPtr<T> fluid,
           T dt,
           DomainHostPtr<T> domain,
           CodecHostPtr<T> codec,
           SourceHostPtr<T> source,
           SinkHostPtr<T> sink,
           MeasureHostPtr<T> measure,
           ColliderHostPtr<T> collider,
           OrchestratorHostPtr<T> solver);

    /**
     * @brief Default destructor.
     *
     * @details
     * Since subsystem lifetime is managed through RAII member objects and shared
     * pointers, the destructor is defaulted.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ~System() = default;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder that can be used to stage a
     * runtime configuration before constructing the final system object.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Execute one full simulation step.
     *
     * @details
     * Runs the system's staged update loop in its prescribed order.
     *
     * A typical implementation may include some or all of the following stages:
     * - emission,
     * - search structure rebuild,
     * - classification / codec update,
     * - measurement,
     * - solver execution,
     * - collision handling,
     * - sink/removal processing,
     * - time integration.
     *
     * The exact sequence and stage semantics are implementation-defined in
     * `system.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    /**
     * @brief Execute the source/emission stage.
     *
     * @details
     * Invokes the configured source subsystem, if any, to inject particles or
     * other state into the simulation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    /**
     * @brief Execute the spatial search rebuild/update stage.
     *
     * @details
     * Refreshes the active spatial hashing search structure so neighborhood
     * queries remain consistent with the current fluid/domain state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    search();

    /**
     * @brief Execute the classification / codec preparation stage.
     *
     * @details
     * Runs the stage responsible for classifying or transforming simulation state,
     * typically through the configured codec and any associated runtime logic.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify();

    /**
     * @brief Execute the measurement/diagnostics stage.
     *
     * @details
     * Invokes the configured measurement subsystem, if any, to compute runtime
     * statistics, diagnostics, or derived observables.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure();

    /**
     * @brief Execute the solver/orchestration stage.
     *
     * @details
     * Invokes the configured solver/orchestrator subsystem, if any, to advance
     * the simulation state according to the current probes and timestep.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve();

    /**
     * @brief Execute the collision stage.
     *
     * @details
     * Invokes the configured collider subsystem, if any, to resolve
     * particle-surface interactions over the current timestep.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide();

    /**
     * @brief Execute the sink/removal stage.
     *
     * @details
     * Invokes the configured sink subsystem, if any, to remove particles or
     * other state according to the sink policy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    /**
     * @brief Execute the time integration stage.
     *
     * @details
     * Advances the simulation's time-dependent state using the configured
     * timestep and the results of the preceding stages.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration();

    /**
     * @brief Replace the active fluid subsystem.
     *
     * @details
     * Updates the system's fluid handle and typically requires the canonical
     * particle probe to be refreshed accordingly.
     *
     * @param fluid Host-side shared pointer to the new fluid subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid);

    /**
     * @brief Set the simulation timestep.
     *
     * @param dt New timestep value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_dt(T dt);

    /**
     * @brief Replace the active domain subsystem.
     *
     * @param domain Host-side shared pointer to the new domain subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_domain(const DomainHostPtr<T>& domain);

    /**
     * @brief Replace the active codec subsystem.
     *
     * @param codec Host-side shared pointer to the new codec subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(const CodecHostPtr<T>& codec);

    /**
     * @brief Replace the active source subsystem.
     *
     * @param source Host-side shared pointer to the new source subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_source(const SourceHostPtr<T>& source);

    /**
     * @brief Replace the active sink subsystem.
     *
     * @param sink Host-side shared pointer to the new sink subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sink(const SinkHostPtr<T>& sink);

    /**
     * @brief Replace the active measurement subsystem.
     *
     * @param measure Host-side shared pointer to the new measurement subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measure(const MeasureHostPtr<T>& measure);

    /**
     * @brief Replace the active collider subsystem using a shared pointer.
     *
     * @param collider Host-side shared pointer to the new collider subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_collider(const ColliderHostPtr<T>& collider);

    /**
     * @brief Replace the active collider subsystem using a value object.
     *
     * @details
     * This overload allows callers to provide a collider by value, which is then
     * wrapped or stored according to the implementation policy in `system.hpp`.
     *
     * @param collider Collider value to install into the system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_collider(const Collider<T>& collider);

    /**
     * @brief Replace the active solver/orchestrator subsystem.
     *
     * @param solver Host-side shared pointer to the new solver subsystem.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_solver(const OrchestratorHostPtr<T>& solver);

    /**
     * @brief Return the active fluid subsystem.
     *
     * @return Host-side shared pointer to the fluid subsystem.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidHostPtr<T>
    fluid() const noexcept;

    /**
     * @brief Return mutable access to the canonical fluid device probe.
     *
     * @details
     * This probe provides the authoritative runtime device-facing view of the
     * active fluid state.
     *
     * @return Mutable reference to the fluid device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>&
    particle_probe() noexcept;

    /**
     * @brief Return const access to the canonical fluid device probe.
     *
     * @return Const reference to the fluid device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const FluidDeviceProbe<T>&
    particle_probe() const noexcept;

    /**
     * @brief Return mutable access to the canonical domain device probe.
     *
     * @return Mutable reference to the domain device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>&
    domain_probe() noexcept;

    /**
     * @brief Return const access to the canonical domain device probe.
     *
     * @return Const reference to the domain device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const DomainDeviceProbe<T>&
    domain_probe() const noexcept;

    /**
     * @brief Return mutable access to the canonical spatial-searcher device probe.
     *
     * @return Mutable reference to the searcher device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE SpatialHashingProbe<T>&
    searcher_probe() noexcept;

    /**
     * @brief Return const access to the canonical spatial-searcher device probe.
     *
     * @return Const reference to the searcher device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const SpatialHashingProbe<T>&
    searcher_probe() const noexcept;

    /**
     * @brief Return mutable access to the canonical codec device probe.
     *
     * @return Mutable reference to the codec device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE CodecDeviceProbe<T>&
    codec_probe() noexcept;

    /**
     * @brief Return const access to the canonical codec device probe.
     *
     * @return Const reference to the codec device probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE const CodecDeviceProbe<T>&
    codec_probe() const noexcept;

    /**
     * @brief Return the simulation timestep.
     *
     * @return Current timestep value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    dt() const noexcept;

    /**
     * @brief Return the active domain subsystem.
     *
     * @return Const reference to the active domain shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DomainHostPtr<T>&
    domain() const noexcept;

    /**
     * @brief Return the active codec subsystem.
     *
     * @return Const reference to the active codec shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    /**
     * @brief Return the active source subsystem.
     *
     * @return Const reference to the active source shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SourceHostPtr<T>&
    source() const noexcept;

    /**
     * @brief Return the active sink subsystem.
     *
     * @return Const reference to the active sink shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SinkHostPtr<T>&
    sink() const noexcept;

    /**
     * @brief Return the active measurement subsystem.
     *
     * @return Const reference to the active measurement shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MeasureHostPtr<T>&
    measure() const noexcept;

    /**
     * @brief Return the active collider subsystem.
     *
     * @return Const reference to the active collider shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const ColliderHostPtr<T>&
    collider() const noexcept;

    /**
     * @brief Return the active solver/orchestrator subsystem.
     *
     * @return Const reference to the active solver shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const OrchestratorHostPtr<T>&
    solver() const noexcept;

    /**
     * @brief Clear the source subsystem.
     *
     * @details
     * Removes the optional source stage from the system, after which emission
     * typically becomes a no-op until a new source is installed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_source() noexcept;

    /**
     * @brief Clear the sink subsystem.
     *
     * @details
     * Removes the optional sink stage from the system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_sink() noexcept;

    /**
     * @brief Clear the measurement subsystem.
     *
     * @details
     * Removes the optional measurement stage from the system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_measure() noexcept;

    /**
     * @brief Clear the collider subsystem.
     *
     * @details
     * Removes the optional collision stage from the system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_collider() noexcept;

    /**
     * @brief Clear the solver/orchestrator subsystem.
     *
     * @details
     * Removes the optional solver stage from the system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_solver() noexcept;

private:
    /**
     * @brief Simulation timestep.
     *
     * @details
     * Stores the canonical timestep used by staged update routines.
     */
    T _dt { static_cast<T>(0.01) };

    /**
     * @brief Active fluid subsystem.
     *
     * @details
     * Stores the host-side shared pointer to the canonical particle/fluid state.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Active domain subsystem.
     *
     * @details
     * Stores the host-side shared pointer to the domain/grid state.
     */
    DomainHostPtr<T> _domain {};

    /**
     * @brief Active codec subsystem.
     *
     * @details
     * Stores the host-side shared pointer to the runtime codec.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Derived spatial search subsystem.
     *
     * @details
     * Stores the active spatial hashing searcher used to build neighborhood
     * queries for solver, codec, or collision stages.
     */
    SpatialHashingSearcherHostPtr<T> _searcher {};

    /**
     * @brief Canonical fluid device probe.
     *
     * @details
     * Device-facing runtime view of the active fluid subsystem.
     */
    FluidDeviceProbe<T> _particle_probe {};

    /**
     * @brief Canonical domain device probe.
     *
     * @details
     * Device-facing runtime view of the active domain subsystem.
     */
    DomainDeviceProbe<T> _domain_probe {};

    /**
     * @brief Canonical spatial-searcher device probe.
     *
     * @details
     * Device-facing runtime view of the active searcher subsystem.
     */
    SpatialHashingProbe<T> _searcher_probe {};

    /**
     * @brief Canonical codec device probe.
     *
     * @details
     * Device-facing runtime view of the active codec subsystem.
     */
    CodecDeviceProbe<T> _codec_probe {};

    /**
     * @brief Optional source subsystem.
     *
     * @details
     * Provides particle or state emission during the source stage when configured.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Optional sink subsystem.
     *
     * @details
     * Provides particle or state removal during the sink stage when configured.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Optional measurement subsystem.
     *
     * @details
     * Provides diagnostic or observable computation during the measure stage.
     */
    MeasureHostPtr<T> _measure {};

    /**
     * @brief Optional collider subsystem.
     *
     * @details
     * Provides particle-surface collision handling during the collision stage.
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Optional solver/orchestrator subsystem.
     *
     * @details
     * Provides the main simulation advancement logic during the solve stage.
     */
    OrchestratorHostPtr<T> _solver {};
};

/**
 * @brief Fluent builder for @ref System.
 *
 * @details
 * The builder provides a controlled construction path for assembling a runtime
 * system from its core and optional subsystems.
 *
 * It stages:
 * - the fluid subsystem,
 * - the timestep,
 * - the domain subsystem,
 * - the codec subsystem,
 * - optional source, sink, measure, collider, and solver subsystems.
 *
 * After validation, it can construct either:
 * - a @ref System value via @ref build, or
 * - a host-owned shared pointer via @ref make_host_shared.
 *
 * ## Typical usage
 * @code
 * auto system = atlas::System<float>::builder()
 *     .with_fluid(fluid)
 *     .with_dt(0.001f)
 *     .with_domain(domain)
 *     .with_codec(codec)
 *     .with_solver(solver)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * The exact validation logic is implementation-defined in `system.hpp`, but it
 * typically checks that the required core subsystems are present and that the
 * timestep/configuration are usable.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class System<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with default-initialized staged state and a default
     * timestep of `0.01`.
     */
    Builder() = default;

    /**
     * @brief Set the fluid subsystem.
     *
     * @param fluid Host-side shared pointer to the fluid subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the simulation timestep.
     *
     * @param dt Timestep to stage for the final system.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    /**
     * @brief Set the domain subsystem.
     *
     * @param domain Host-side shared pointer to the domain subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const DomainHostPtr<T>& domain);

    /**
     * @brief Set the codec subsystem.
     *
     * @param codec Host-side shared pointer to the codec subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(const CodecHostPtr<T>& codec);

    /**
     * @brief Set the source subsystem.
     *
     * @param source Host-side shared pointer to the source subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source);

    /**
     * @brief Set the sink subsystem.
     *
     * @param sink Host-side shared pointer to the sink subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink);

    /**
     * @brief Set the measurement subsystem.
     *
     * @param measure Host-side shared pointer to the measurement subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measure(const MeasureHostPtr<T>& measure);

    /**
     * @brief Set the collider subsystem using a shared pointer.
     *
     * @param collider Host-side shared pointer to the collider subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider);

    /**
     * @brief Set the collider subsystem using a value object.
     *
     * @param collider Collider value to stage for the final system.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const Collider<T>& collider);

    /**
     * @brief Set the solver/orchestrator subsystem.
     *
     * @param solver Host-side shared pointer to the solver subsystem.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(const OrchestratorHostPtr<T>& solver);

    /**
     * @brief Build a configured @ref System by value after validation.
     *
     * @details
     * Validates the staged builder state and constructs the final system object.
     *
     * @return Fully constructed system value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build();

    /**
     * @brief Build a configured @ref System in a host_shared_ptr after validation.
     *
     * @details
     * Validates the staged builder state, constructs the system, and returns it
     * in a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<System<T>>` owning the constructed system.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared();

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the currently staged configuration.
     *
     * Typical checks may include:
     * - required fluid/domain/codec subsystems are present,
     * - the timestep is valid,
     * - optional subsystem dependencies are mutually compatible.
     *
     * @note
     * The exact validation policy is implementation-defined in `system.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending timestep.
     */
    T _dt { static_cast<T>(0.01) };

    /**
     * @brief Pending fluid subsystem.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Pending domain subsystem.
     */
    DomainHostPtr<T> _domain {};

    /**
     * @brief Pending codec subsystem.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Pending source subsystem.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Pending sink subsystem.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Pending measure subsystem.
     */
    MeasureHostPtr<T> _measure {};

    /**
     * @brief Pending collider subsystem.
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Pending solver subsystem.
     */
    OrchestratorHostPtr<T> _solver {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::System.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using System = system::System<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::System.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

} // namespace atlas

#include <atlas/system/system.hpp>