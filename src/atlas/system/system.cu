#include <atlas/system/system.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas {

System::System(FluidHostPtr fluid,
               UniverseHostPtr universe,
               SourceHostPtr source,
               SinkHostPtr sink,
               ColliderHostPtr collider,
               OrchestratorHostPtr orchestrator,
               const float dt)
    : _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _source(std::move(source))
    , _sink(std::move(sink))
    , _collider(std::move(collider))
    , _orchestrator(std::move(orchestrator))
    , _dt(dt) {

    if (_fluid) {
        _cached_position_state = _fluid->state<FluidPositionState>();
        _cached_velocity_state = _fluid->state<FluidVelocityState>();
    }
}

System::Builder
System::builder() noexcept {
    return Builder {};
}

void
System::update() {
    emit();
    orchestrate();
    advect();
    remove();
}

void
System::emit() {

    if (_source) {
        _source->update(_dt);
    }
}

void
System::orchestrate() {

    if (_orchestrator) {
        _orchestrator->update(_dt);
    }
}

void
System::advect() {

    if (_collider) {
        _collider->update(_dt);
    } else {

        time_integration();
    }
}

void
System::remove() {

    if (_sink) {
        _sink->update(_dt);
    }
}

void
System::time_integration() {

    if (!_fluid || !(_dt > 0.0f)) {
        return;
    }

    if (!_cached_position_state || !_cached_velocity_state) {
        return;
    }

    const int particle_count = static_cast<int>(_fluid->particle_count());
    if (particle_count == 0) {
        return;
    }

    auto* positions_ptr        = atlas::raw_pointer_cast(_cached_position_state->data().data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(_cached_velocity_state->data().data());
    const float dt             = _dt;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt] ATLAS_ALL_DEVICE(const int i) {
            positions_ptr[i] += velocities_ptr[i] * dt;
        });
}

float
System::dt() const noexcept {
    return _dt;
}

const FluidHostPtr&
System::fluid() const noexcept {
    return _fluid;
}

System::Builder&
System::Builder::with_fluid(const FluidHostPtr& fluid) noexcept {
    _fluid = fluid;
    return *this;
}

System::Builder&
System::Builder::with_domain(const UniverseHostPtr& universe) noexcept {
    _universe = universe;
    return *this;
}

System::Builder&
System::Builder::with_source(const SourceHostPtr& source) noexcept {
    _source = source;
    return *this;
}

System::Builder&
System::Builder::with_sink(const SinkHostPtr& sink) noexcept {
    _sink = sink;
    return *this;
}

System::Builder&
System::Builder::with_collider(const ColliderHostPtr& collider) noexcept {
    _collider = collider;
    return *this;
}

System::Builder&
System::Builder::with_solver(const OrchestratorHostPtr& orchestrator) noexcept {
    _orchestrator = orchestrator;
    return *this;
}

System::Builder&
System::Builder::with_dt(const float dt) noexcept {
    _dt = dt;
    return *this;
}

void
System::Builder::validate() const {

    if (!_fluid) {
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }

    if (!(_dt > 0.0f)) {
        throw std::runtime_error("System::Builder: dt must be positive.");
    }
}

System
System::Builder::build() const {
    validate();

    return System(_fluid, _universe, _source, _sink, _collider, _orchestrator, _dt);
}

atlas::host_shared_ptr<System>
System::Builder::make_host_shared() const {
    validate();

    return atlas::make_host_shared<System>(
        _fluid,
        _universe,
        _source,
        _sink,
        _collider,
        _orchestrator,
        _dt);
}

}
