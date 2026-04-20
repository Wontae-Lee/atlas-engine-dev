#pragma once

/**
 * @file system.h
 * @brief Declares the top-level simulation System that coordinates emission,
 *        orchestration, advection/collision, and particle removal.
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
 * @brief High-level simulation system that coordinates the major runtime subsystems.
 *
 * A System object acts as the top-level driver for one simulation step.
 * It may coordinate the following optional subsystems:
 * - a source that emits or initializes particles,
 * - an orchestrator that applies solver or update logic,
 * - a collider that advances collider units and resolves collisions,
 * - a sink that removes particles.
 *
 * The system always requires a valid fluid object because all step logic
 * eventually operates on fluid state. Other subsystems are optional.
 *
 * The default update sequence is:
 * 1. emit particles through the source,
 * 2. run the orchestrator,
 * 3. advect particles either through the collider or through plain time integration,
 * 4. remove particles through the sink.
 *
 * If no collider is present, the system falls back to direct time integration:
 *   position += velocity * dt
 *
 * @tparam T Floating-point scalar type used for simulation quantities.
 */
template <typename T>
class System final {
    static_assert(std::is_floating_point_v<T>, "System requires a floating-point T");

public:
    /**
     * @brief Builder type used for validated System construction.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty system with no installed dependencies and a default time step.
     */
    System() = default;

    /**
     * @brief Constructs a simulation system from its runtime dependencies.
     *
     * @param fluid Host shared pointer to the fluid object.
     * @param universe Host shared pointer to the universe/domain object.
     * @param source Optional source subsystem.
     * @param sink Optional sink subsystem.
     * @param collider Optional collider subsystem.
     * @param orchestrator Optional orchestrator subsystem.
     * @param dt Simulation time step.
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
     */
    ~System() = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Newly created builder object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Executes one full simulation step.
     *
     * The default step order is:
     * 1. source update,
     * 2. orchestrator update,
     * 3. collider update or fallback time integration,
     * 4. sink update.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update();

    /**
     * @brief Executes only the emission phase.
     *
     * If no source is installed, this function does nothing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    /**
     * @brief Executes only the orchestration phase.
     *
     * If no orchestrator is installed, this function does nothing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate();

    /**
     * @brief Executes only the advection/collision phase.
     *
     * If a collider is installed, collider-driven update is used.
     * Otherwise, direct time integration is performed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    advect();

    /**
     * @brief Executes only the removal phase.
     *
     * If no sink is installed, this function does nothing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove();

    /**
     * @brief Performs fallback particle advection by direct time integration.
     *
     * This function updates particle positions as:
     *   position += velocity * dt
     *
     * It is used when no collider subsystem is installed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration();

    /**
     * @brief Returns the configured simulation time step.
     *
     * @return Time step value.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    dt() const noexcept;

    /**
     * @brief Returns the installed fluid object.
     *
     * @return Const reference to the host shared pointer owning the fluid.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

private:
    /**
     * @brief Target fluid object processed by the system.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Universe/domain object associated with the system.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Optional particle source subsystem.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Optional particle sink subsystem.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Optional collider subsystem.
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Optional orchestrator subsystem.
     */
    OrchestratorHostPtr<T> _orchestrator {};

    /**
     * @brief Simulation time step.
     */
    T _dt { static_cast<T>(0.01) };
};

/**
 * @brief Builder for validated System construction.
 *
 * The builder collects the runtime dependencies used by System and validates
 * the minimum required invariants before construction.
 *
 * Required invariants:
 * - a valid fluid object must be installed,
 * - the time step must be strictly positive.
 *
 * Other subsystems are optional and may be omitted.
 *
 * @tparam T Floating-point scalar type used for simulation quantities.
 */
template <typename T>
class System<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Installs the fluid object.
     *
     * @param fluid Host shared pointer to the fluid object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(const FluidHostPtr<T>& fluid) noexcept;

    /**
     * @brief Installs the universe/domain object.
     *
     * @param universe Host shared pointer to the universe object.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_domain(const UniverseHostPtr<T>& universe) noexcept;

    /**
     * @brief Installs the source subsystem.
     *
     * @param source Host shared pointer to the source subsystem.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_source(const SourceHostPtr<T>& source) noexcept;

    /**
     * @brief Installs the sink subsystem.
     *
     * @param sink Host shared pointer to the sink subsystem.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sink(const SinkHostPtr<T>& sink) noexcept;

    /**
     * @brief Installs the collider subsystem.
     *
     * @param collider Host shared pointer to the collider subsystem.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider) noexcept;

    /**
     * @brief Installs the orchestrator subsystem.
     *
     * @param orchestrator Host shared pointer to the orchestrator subsystem.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(const OrchestratorHostPtr<T>& orchestrator) noexcept;

    /**
     * @brief Sets the simulation time step.
     *
     * @param dt Positive time step value.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    /**
     * @brief Validates the builder state and constructs a System object.
     *
     * @return Constructed System object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE System<T>
    build() const;

    /**
     * @brief Builds a System object and wraps it in host-managed shared storage.
     *
     * @return Host shared pointer to the constructed System.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<System<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the builder configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Builder-owned fluid object.
     */
    FluidHostPtr<T> _fluid {};

    /**
     * @brief Builder-owned universe object.
     */
    UniverseHostPtr<T> _universe {};

    /**
     * @brief Builder-owned source subsystem.
     */
    SourceHostPtr<T> _source {};

    /**
     * @brief Builder-owned sink subsystem.
     */
    SinkHostPtr<T> _sink {};

    /**
     * @brief Builder-owned collider subsystem.
     */
    ColliderHostPtr<T> _collider {};

    /**
     * @brief Builder-owned orchestrator subsystem.
     */
    OrchestratorHostPtr<T> _orchestrator {};

    /**
     * @brief Builder-owned time step.
     */
    T _dt { static_cast<T>(0.01) };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::System.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using System = system::System<T>;

/**
 * @brief Host shared pointer alias for System.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

} // namespace atlas

#include <atlas/system/system.hpp>