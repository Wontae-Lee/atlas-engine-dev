#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
System<T>::System(FluidHostPtr<T> fluid,
                  UniverseHostPtr<T> universe,
                  SourceHostPtr<T> source,
                  SinkHostPtr<T> sink,
                  ColliderHostPtr<T> collider,
                  OrchestratorHostPtr<T> orchestrator,
                  const T dt)
    : _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _source(std::move(source))
    , _sink(std::move(sink))
    , _collider(std::move(collider))
    , _orchestrator(std::move(orchestrator))
    , _dt(dt) {

    // Store the runtime dependencies as provided.
    //
    // This constructor does not perform validation directly.
    // The intended validated construction path is through Builder::build()
    // or Builder::make_host_shared().
}

template <typename T>
typename System<T>::Builder
System<T>::builder() noexcept {

    // Return a default-initialized builder for fluent System construction.
    return Builder {};
}

template <typename T>
void
System<T>::update() {

    // Execute one full simulation step in the standard subsystem order.
    //
    // Current order:
    // 1. source emission/update
    // 2. orchestrator update
    // 3. collider-driven advection or fallback time integration
    // 4. sink removal/update
    //
    // This order implies that:
    // - newly emitted particles may be processed by the orchestrator immediately,
    // - orchestrator results affect the same-step advection/collision phase,
    // - sinks observe the post-advection/post-collision particle state.

    if (_source) {
        _source->update(_dt);
    }

    if (_orchestrator) {
        _orchestrator->update(_dt);
    }

    if (_collider) {
        _collider->update(_dt);
    } else {
        time_integration();
    }

    if (_sink) {
        _sink->update(_dt);
    }
}

template <typename T>
void
System<T>::emit() {

    // Execute only the source phase.
    //
    // This is useful when external control code wants to split the full step
    // into manually scheduled sub-phases.
    if (_source) {
        _source->update(_dt);
    }
}

template <typename T>
void
System<T>::orchestrate() {

    // Execute only the orchestrator phase.
    if (_orchestrator) {
        _orchestrator->update(_dt);
    }
}

template <typename T>
void
System<T>::advect() {

    // Execute only the motion phase.
    //
    // If a collider is installed, it becomes the authoritative motion handler
    // because it may both advance collider units and resolve fluid collisions.
    //
    // Otherwise, perform plain velocity-based position integration.
    if (_collider) {
        _collider->update(_dt);
    } else {
        time_integration();
    }
}

template <typename T>
void
System<T>::remove() {

    // Execute only the sink/removal phase.
    if (_sink) {
        _sink->update(_dt);
    }
}

template <typename T>
void
System<T>::time_integration() {

    // Fallback advection path used when no collider subsystem is installed.
    //
    // This path performs simple explicit Euler-style position integration:
    //   position += velocity * dt

    // A valid fluid object and a strictly positive time step are required.
    if (!_fluid || !(_dt > T(0))) {
        return;
    }

    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Position integration requires both position and velocity state buffers.
    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }

    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();

    // Reject degenerate cases before launching a device kernel.
    if (positions.empty() || velocities.empty() || _fluid->particle_count() == 0) {
        return;
    }

    auto* positions_ptr        = atlas::raw_pointer_cast(positions.data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(velocities.data());
    const int particle_count   = static_cast<int>(_fluid->particle_count());
    const T dt                 = _dt;

    // Update each particle independently on the device.
    //
    // This assumes that:
    // - positions and velocities share the same valid active prefix length,
    // - particle_count reflects the number of active particles in that prefix.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt] ATLAS_DEVICE(const int i) {
            positions_ptr[i] += velocities_ptr[i] * dt;
        });
}

template <typename T>
T
System<T>::dt() const noexcept {

    // Return the configured simulation time step.
    return _dt;
}

template <typename T>
const FluidHostPtr<T>&
System<T>::fluid() const noexcept {

    // Return the installed fluid object.
    return _fluid;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_fluid(const FluidHostPtr<T>& fluid) noexcept {

    // Install the required fluid dependency.
    _fluid = fluid;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_domain(const UniverseHostPtr<T>& universe) noexcept {

    // Install the optional universe/domain dependency.
    _universe = universe;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_source(const SourceHostPtr<T>& source) noexcept {

    // Install the optional source subsystem.
    _source = source;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sink(const SinkHostPtr<T>& sink) noexcept {

    // Install the optional sink subsystem.
    _sink = sink;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) noexcept {

    // Install the optional collider subsystem.
    _collider = collider;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solver(const OrchestratorHostPtr<T>& orchestrator) noexcept {

    // Install the optional orchestrator subsystem.
    _orchestrator = orchestrator;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_dt(const T dt) noexcept {

    // Install the time step value.
    _dt = dt;
    return *this;
}

template <typename T>
void
System<T>::Builder::validate() const {

    // The system always requires a fluid object because all simulation phases
    // ultimately operate on fluid state.
    if (!_fluid) {
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }

    // Time step must be strictly positive.
    if (!(_dt > T(0))) {
        throw std::runtime_error("System::Builder: dt must be positive.");
    }
}

template <typename T>
System<T>
System<T>::Builder::build() const {

    // Refuse construction when required invariants are violated.
    validate();

    // Construct the final System value object from the validated dependencies.
    return System<T>(_fluid, _universe, _source, _sink, _collider, _orchestrator, _dt);
}

template <typename T>
atlas::host_shared_ptr<System<T>>
System<T>::Builder::make_host_shared() const {

    // Refuse shared-object construction when required invariants are violated.
    validate();

    // Construct the System directly inside host-managed shared storage.
    return atlas::make_host_shared<System<T>>(
        _fluid,
        _universe,
        _source,
        _sink,
        _collider,
        _orchestrator,
        _dt);
}

} // namespace atlas::system