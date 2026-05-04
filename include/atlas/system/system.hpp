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
}

template <typename T>
typename System<T>::Builder
System<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
System<T>::update() {
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
    if (_source) {
        _source->update(_dt);
    }
}

template <typename T>
void
System<T>::orchestrate() {
    if (_orchestrator) {
        _orchestrator->update(_dt);
    }
}

template <typename T>
void
System<T>::advect() {
    if (_collider) {
        _collider->update(_dt);
    } else {
        time_integration();
    }
}

template <typename T>
void
System<T>::remove() {
    if (_sink) {
        _sink->update(_dt);
    }
}

template <typename T>
void
System<T>::time_integration() {
    if (!_fluid || !(_dt > T(0))) {
        return;
    }
    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }
    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();
    if (positions.empty() || velocities.empty() || _fluid->particle_count() == 0) {
        return;
    }
    auto* positions_ptr        = atlas::raw_pointer_cast(positions.data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(velocities.data());
    const int particle_count   = static_cast<int>(_fluid->particle_count());
    const T dt                 = _dt;
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
    if (!_fluid) {
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }
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

}