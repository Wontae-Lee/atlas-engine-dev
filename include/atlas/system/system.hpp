#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas {

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
    // Cache state pointers once so time_integration() can skip the per-step
    // unordered_map lookup that state<>() would otherwise perform.
    if (_fluid) {
        _cached_position_state = _fluid->template state<FluidPositionState<T>>();
        _cached_velocity_state = _fluid->template state<FluidVelocityState<T>>();
    }
}

template <typename T>
typename System<T>::Builder
System<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
System<T>::update() {
    // Execute one complete simulation step.
    emit();
    remove();
    orchestrate();
    advect();
}

template <typename T>
void
System<T>::emit() {
    // Inject new particles when a source is available.
    if (_source) {
        _source->update(_dt);
    }
}

template <typename T>
void
System<T>::orchestrate() {
    // Apply solver-side updates such as search, measurement, force, and collision setup.
    if (_orchestrator) {
        _orchestrator->update(_dt);
    }
}

template <typename T>
void
System<T>::advect() {
    // Use the collider-specific update path when a collider is available.
    if (_collider) {
        _collider->update(_dt);
    } else {
        // Fall back to simple position integration.
        time_integration();
    }
}

template <typename T>
void
System<T>::remove() {
    // Remove particles through the sink when one is available.
    if (_sink) {
        _sink->update(_dt);
    }
}

template <typename T>
void
System<T>::time_integration() {
    // Integration requires a valid fluid object and positive time step.
    if (!_fluid || !(_dt > T(0))) {
        return;
    }

    // Use the cached state pointers resolved at construction to avoid a
    // per-step unordered_map lookup through state<>().
    if (!_cached_position_state || !_cached_velocity_state) {
        return;
    }

    // Read particle_count once; it is used both as the early-exit guard and
    // as the kernel range, eliminating the previous double call.
    const int particle_count = static_cast<int>(_fluid->particle_count());
    if (particle_count == 0) {
        return;
    }

    // Extract raw device-accessible pointers for the parallel kernel.
    auto* positions_ptr        = atlas::raw_pointer_cast(_cached_position_state->data().data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(_cached_velocity_state->data().data());
    const T dt                 = _dt;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt] ATLAS_DEVICE(const int i) {
            // Explicit Euler position update: x += v * dt.
            positions_ptr[i] += velocities_ptr[i] * dt;
        });
}

template <typename T>
T
System<T>::dt() const noexcept {
    return _dt;
}

template <typename T>
const FluidHostPtr<T>&
System<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_fluid(const FluidHostPtr<T>& fluid) noexcept {
    _fluid = fluid;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_domain(const UniverseHostPtr<T>& universe) noexcept {
    _universe = universe;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_source(const SourceHostPtr<T>& source) noexcept {
    _source = source;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sink(const SinkHostPtr<T>& sink) noexcept {
    _sink = sink;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) noexcept {
    _collider = collider;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solver(const OrchestratorHostPtr<T>& orchestrator) noexcept {
    _orchestrator = orchestrator;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_dt(const T dt) noexcept {
    _dt = dt;
    return *this;
}

template <typename T>
void
System<T>::Builder::validate() const {
    // A system cannot advance particles without a fluid container.
    if (!_fluid) {
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }

    // The simulation time step must be strictly positive.
    if (!(_dt > T(0))) {
        throw std::runtime_error("System::Builder: dt must be positive.");
    }
}

template <typename T>
System<T>
System<T>::Builder::build() const {
    validate();

    return System<T>(_fluid, _universe, _source, _sink, _collider, _orchestrator, _dt);
}

template <typename T>
atlas::host_shared_ptr<System<T>>
System<T>::Builder::make_host_shared() const {
    validate();

    return atlas::make_host_shared<System<T>>(
        _fluid,
        _universe,
        _source,
        _sink,
        _collider,
        _orchestrator,
        _dt);
}

} // namespace atlas
