#pragma once

#include <atlas/codec/single_codec.h>
#include <atlas/logging/logging.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename System<T>::Builder
System<T>::builder() noexcept {
    // Return a fresh builder object for staged `System<T>` construction.
    //
    // This entry point is useful when the simulation components
    // (fluid, domain, codec, source, sink, measure, collider, solver)
    // are assembled incrementally before creating the final system object.
    return Builder {};
}

template <typename T>
System<T>::System(FluidHostPtr<T> fluid)
    : _fluid(std::move(fluid)) {
    // A system cannot exist without a valid fluid object because the fluid owns
    // the particle state that nearly every simulation stage operates on.
    if (!_fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    // Build the canonical device-side particle probe immediately.
    //
    // Rationale:
    // - most runtime stages consume particle data through this probe,
    // - creating it during construction guarantees that the system starts
    //   in a usable state even when optional subsystems are absent.
    _particle_probe = _fluid->make_device_probe();
}

template <typename T>
System<T>::System(FluidHostPtr<T> fluid,
                  const T dt,
                  DomainHostPtr<T> domain,
                  CodecHostPtr<T> codec,
                  SourceHostPtr<T> source,
                  SinkHostPtr<T> sink,
                  MeasureHostPtr<T> measure,
                  ColliderHostPtr<T> collider,
                  OrchestratorHostPtr<T> solver)
    : _dt(dt)
    , _fluid(std::move(fluid))
    , _domain(std::move(domain))
    , _codec(std::move(codec))
    , _source(std::move(source))
    , _sink(std::move(sink))
    , _measure(std::move(measure))
    , _collider(std::move(collider))
    , _solver(std::move(solver)) {
    // Fluid is the only mandatory subsystem.
    //
    // Without a fluid object, there is no particle state to evolve,
    // classify, measure, collide, emit into, or remove from.
    if (!_fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    // Create the device-side particle probe up front.
    //
    // This probe is required regardless of whether optional subsystems such as
    // domain, codec, solver, or collider are present.
    _particle_probe = _fluid->make_device_probe();

    if (_domain) {
        // When a domain is available, the system can also support:
        // - spatial neighbor search via the searcher,
        // - domain-aware measurement,
        // - codec-based classification/encoding paths,
        // - solver stages that depend on domain structure.
        //
        // Build the searcher immediately so its probe can be prepared once
        // and reused across update stages.
        _searcher = atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);

        // Cache device-side views of the domain and searcher so runtime steps
        // can pass lightweight probes instead of repeatedly rebuilding them.
        _domain_probe   = _domain->make_device_probe();
        _searcher_probe = _searcher->make_device_probe();

        if (!_codec) {
            // If the caller did not provide a codec, install the default
            // pass-through codec associated with the current domain.
            //
            // This guarantees that classification/solver code paths still have
            // a valid codec object when a domain exists.
            _codec = atlas::make_host_shared<SingleCodec<T>>(_domain);
        }
    }

    if (_codec) {
        // If a codec exists, build its device-side probe immediately so it can
        // participate in classification and solver stages without extra setup.
        _codec_probe = _codec->make_device_probe();
    }
}

template <typename T>
void
System<T>::update() {
    atlas::logger::info() << "System::update: begin";

    // Execute one full system step in the canonical order.
    //
    // Stage ordering:
    // 1. emit      : inject new particles or sources into the fluid
    // 2. search    : rebuild spatial neighbor structure
    // 3. classify  : run codec encode/decode style classification
    // 4. measure   : evaluate domain- or particle-based observables
    // 5. solve     : run orchestrator/solver stage
    // 6. collide   : resolve collider interaction or fallback advection
    // 7. remove    : remove/sink particles after motion/update
    //
    // Keeping the ordering centralized here ensures all update semantics
    // remain explicit and easy to audit.
    emit();
    search();
    classify();
    measure();
    solve();
    collide();
    remove();

    atlas::logger::info() << "System::update: end";
}

template <typename T>
void
System<T>::emit() {
    atlas::logger::info() << "System::emit: begin";

    if (_source) {
        // Advance the source's internal state by the system time step first.
        //
        // This lets time-dependent emitters update clocks, transforms,
        // rates, or schedules before producing particles.
        _source->update(_dt);

        // Inject newly emitted particles into the fluid through the canonical
        // device-side particle probe.
        _source->emit(_particle_probe);
    }

    atlas::logger::info() << "System::emit: end";
}

template <typename T>
void
System<T>::search() {
    atlas::logger::info() << "System::search: begin";

    if (!_searcher) {
        // Spatial search is optional.
        //
        // If no searcher is available, simply skip the stage while preserving
        // the logging structure of the update pipeline.
        atlas::logger::info() << "System::search: end";
        return;
    }

    // Rebuild the particle search structure from current particle state.
    //
    // This prepares neighbor and locality information for downstream stages
    // such as classification, measurement, or solving.
    _searcher->build(_particle_probe);

    atlas::logger::info() << "System::search: end";
}

template <typename T>
void
System<T>::classify() {
    atlas::logger::info() << "System::classify: begin";

    if (!_codec || !_domain || !_searcher) {
        // Classification requires all three components:
        // - codec    : defines encode/decode behavior
        // - domain   : supplies domain-level structure/probe
        // - searcher : supplies neighborhood/spatial indexing information
        atlas::logger::info() << "System::classify: end";
        return;
    }

    // Let the codec perform its update path.
    //
    // For codecs that support encode/decode workflows, this is the stage
    // where particle/domain/search data is interpreted, classified,
    // compressed, reconstructed, or otherwise transformed.
    _codec->update(_particle_probe, _domain_probe, _searcher_probe, _codec_probe);

    atlas::logger::info() << "System::classify: end";
}

template <typename T>
void
System<T>::measure() {
    atlas::logger::info() << "System::measure: begin";

    if (!_domain || !_searcher || !_measure) {
        // Measurement requires:
        // - a domain for domain-aware aggregation,
        // - a searcher for spatial lookup when needed,
        // - a measure implementation defining what is being measured.
        atlas::logger::info() << "System::measure: end";
        return;
    }

    // Execute the measurement stage using prebuilt probes.
    //
    // This typically computes diagnostics, field estimates, statistics,
    // or other observables derived from the current system state.
    _measure->measure(_domain_probe, _searcher_probe, _particle_probe);

    atlas::logger::info() << "System::measure: end";
}

template <typename T>
void
System<T>::solve() {
    atlas::logger::info() << "System::solve: begin";

    if (!_domain || !_searcher || !_codec || !_solver) {
        // Solver execution requires the full supporting context:
        // - domain   : structural simulation context
        // - searcher : spatial neighborhood information
        // - codec    : classification/state encoding information
        // - solver   : the actual orchestrator implementation
        atlas::logger::info() << "System::solve: end";
        return;
    }

    // Run the configured solver/orchestrator.
    //
    // This stage is where domain-aware simulation logic can update particle
    // state, fields, or auxiliary structures using the latest probes.
    _solver->solve(_domain_probe, _searcher_probe, _particle_probe, _codec_probe);

    atlas::logger::info() << "System::solve: end";
}

template <typename T>
void
System<T>::collide() {
    atlas::logger::info() << "System::collide: begin";

    if (_particle_probe.particle_count <= 0 || !(_dt > T(0))) {
        // Skip collision processing when there are no particles or when the
        // time step is invalid/non-positive.
        atlas::logger::info() << "System::collide: end";
        return;
    }

    if (_collider && !_collider->empty()) {
        // When a collider exists, first advance its internal per-unit state
        // for the current time step, then resolve particle collisions.
        _collider->update(_dt);
        _collider->collide(_particle_probe, _dt);
    } else {
        // If there is no active collider, fall back to pure free-flight motion.
        //
        // This preserves basic particle advection even in systems that do not
        // include collision geometry.
        time_integration();
    }

    atlas::logger::info() << "System::collide: end";
}

template <typename T>
void
System<T>::remove() {
    atlas::logger::info() << "System::remove: begin";

    if (_sink) {
        // Advance the sink's internal state by the system time step first.
        //
        // This allows time-dependent removal logic to update schedules,
        // regions, thresholds, or bookkeeping before processing particles.
        _sink->update(_dt);

        // Remove, consume, or otherwise process particles through the sink.
        _sink->sink(_particle_probe);
    }

    atlas::logger::info() << "System::remove: end";
}

template <typename T>
void
System<T>::time_integration() {
    if (_particle_probe.particle_count <= 0 || !(_dt > T(0))) {
        // No advection is possible without particles or a positive time step.
        return;
    }

    // Copy the probe and time step into local values for kernel capture.
    const auto probe = _particle_probe;
    const T dt       = _dt;

    // Perform minimal explicit advection:
    //     p(t + dt) = p(t) + v(t) * dt
    //
    // This is the fallback motion update used when no collision stage
    // modifies the particle trajectories.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [probe, dt] ATLAS_DEVICE(const int i) {
            const Vector3<T> p0       = probe.pos[i];
            const Vector3<T> velocity = probe.vel[i];
            probe.pos[i]              = p0 + velocity * dt;
        });
}

template <typename T>
void
System<T>::set_fluid(FluidHostPtr<T> fluid) {
    // Fluid is mandatory, so null replacement is rejected.
    if (!fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    // Replace the fluid object and immediately rebuild the canonical
    // device-side particle probe so all later stages observe the new fluid.
    _fluid          = std::move(fluid);
    _particle_probe = _fluid->make_device_probe();
}

template <typename T>
void
System<T>::set_dt(const T dt) {
    // The system time step must remain strictly positive.
    if (!(dt > T(0))) {
        atlas::logger::error()
            << "System: dt must be positive.";
        throw std::runtime_error("System: dt must be positive.");
    }

    // Store the validated time step.
    _dt = dt;
}

template <typename T>
void
System<T>::set_domain(const DomainHostPtr<T>& domain) {
    // Domain replacement must provide a valid domain object.
    if (!domain) {
        atlas::logger::error()
            << "System: domain must not be null.";
        throw std::runtime_error("System: domain must not be null.");
    }

    // Install the new domain.
    _domain = domain;

    // Rebuild domain-dependent runtime infrastructure because changing the
    // domain invalidates old search structures and their associated probes.
    _searcher       = atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);
    _domain_probe   = _domain->make_device_probe();
    _searcher_probe = _searcher->make_device_probe();

    if (!_codec || _codec->type() == CodecType::single) {
        // If no codec exists, or if the current codec is the default pass-through
        // codec, rebuild it against the new domain.
        //
        // Custom codecs are preserved because they may carry specialized state
        // or semantics chosen by the caller.
        _codec       = atlas::make_host_shared<SingleCodec<T>>(_domain);
        _codec_probe = _codec->make_device_probe();
    }
}

template <typename T>
void
System<T>::set_codec(const CodecHostPtr<T>& codec) {
    // Codec replacement requires a valid object.
    if (!codec) {
        atlas::logger::error()
            << "System: codec must not be null.";
        throw std::runtime_error("System: codec must not be null.");
    }

    // Install the new codec and rebuild its device-side probe immediately.
    _codec       = codec;
    _codec_probe = _codec->make_device_probe();
}

template <typename T>
void
System<T>::set_source(const SourceHostPtr<T>& source) {
    // Install or replace the source subsystem.
    //
    // Null is allowed here because the source stage is optional.
    _source = source;
}

template <typename T>
void
System<T>::set_sink(const SinkHostPtr<T>& sink) {
    // Install or replace the sink subsystem.
    //
    // Null is allowed here because the remove stage is optional.
    _sink = sink;
}

template <typename T>
void
System<T>::set_measure(const MeasureHostPtr<T>& measure) {
    // Install or replace the measurement subsystem.
    //
    // Null is allowed here because the measure stage is optional.
    _measure = measure;
}

template <typename T>
void
System<T>::set_collider(const ColliderHostPtr<T>& collider) {
    // Install or replace the collider subsystem.
    //
    // Null is allowed here because the collide stage has a free-flight fallback.
    _collider = collider;
}

template <typename T>
void
System<T>::set_collider(const Collider<T>& collider) {
    // Convenience overload:
    // wrap a collider value into host-shared storage and install it
    // as the active collider subsystem.
    _collider = atlas::make_host_shared<Collider<T>>(collider);
}

template <typename T>
void
System<T>::set_solver(const OrchestratorHostPtr<T>& solver) {
    // Install or replace the solver/orchestrator subsystem.
    //
    // Null is allowed here because the solve stage is optional.
    _solver = solver;
}

template <typename T>
FluidHostPtr<T>
System<T>::fluid() const noexcept {
    // Return the current fluid object.
    return _fluid;
}

template <typename T>
FluidDeviceProbe<T>&
System<T>::particle_probe() noexcept {
    // Return mutable access to the cached particle device probe.
    return _particle_probe;
}

template <typename T>
const FluidDeviceProbe<T>&
System<T>::particle_probe() const noexcept {
    // Return read-only access to the cached particle device probe.
    return _particle_probe;
}

template <typename T>
DomainDeviceProbe<T>&
System<T>::domain_probe() noexcept {
    // Return mutable access to the cached domain device probe.
    return _domain_probe;
}

template <typename T>
const DomainDeviceProbe<T>&
System<T>::domain_probe() const noexcept {
    // Return read-only access to the cached domain device probe.
    return _domain_probe;
}

template <typename T>
SpatialHashingProbe<T>&
System<T>::searcher_probe() noexcept {
    // Return mutable access to the cached spatial-search device probe.
    return _searcher_probe;
}

template <typename T>
const SpatialHashingProbe<T>&
System<T>::searcher_probe() const noexcept {
    // Return read-only access to the cached spatial-search device probe.
    return _searcher_probe;
}

template <typename T>
CodecDeviceProbe<T>&
System<T>::codec_probe() noexcept {
    // Return mutable access to the cached codec device probe.
    return _codec_probe;
}

template <typename T>
const CodecDeviceProbe<T>&
System<T>::codec_probe() const noexcept {
    // Return read-only access to the cached codec device probe.
    return _codec_probe;
}

template <typename T>
T
System<T>::dt() const noexcept {
    // Return the current simulation time step.
    return _dt;
}

template <typename T>
const DomainHostPtr<T>&
System<T>::domain() const noexcept {
    // Return the current domain subsystem handle.
    return _domain;
}

template <typename T>
const CodecHostPtr<T>&
System<T>::codec() const noexcept {
    // Return the current codec subsystem handle.
    return _codec;
}

template <typename T>
const SourceHostPtr<T>&
System<T>::source() const noexcept {
    // Return the current source subsystem handle.
    return _source;
}

template <typename T>
const SinkHostPtr<T>&
System<T>::sink() const noexcept {
    // Return the current sink subsystem handle.
    return _sink;
}

template <typename T>
const MeasureHostPtr<T>&
System<T>::measure() const noexcept {
    // Return the current measurement subsystem handle.
    return _measure;
}

template <typename T>
const ColliderHostPtr<T>&
System<T>::collider() const noexcept {
    // Return the current collider subsystem handle.
    return _collider;
}

template <typename T>
const OrchestratorHostPtr<T>&
System<T>::solver() const noexcept {
    // Return the current solver/orchestrator subsystem handle.
    return _solver;
}

template <typename T>
void
System<T>::clear_source() noexcept {
    // Remove the optional source subsystem from the system.
    _source.reset();
}

template <typename T>
void
System<T>::clear_sink() noexcept {
    // Remove the optional sink subsystem from the system.
    _sink.reset();
}

template <typename T>
void
System<T>::clear_measure() noexcept {
    // Remove the optional measurement subsystem from the system.
    _measure.reset();
}

template <typename T>
void
System<T>::clear_collider() noexcept {
    // Remove the optional collider subsystem from the system.
    _collider.reset();
}

template <typename T>
void
System<T>::clear_solver() noexcept {
    // Remove the optional solver/orchestrator subsystem from the system.
    _solver.reset();
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid dependency in the builder.
    //
    // Final validation is deferred to `validate()`.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_dt(const T dt) noexcept {
    // Store the time step in the builder.
    //
    // Final validation is deferred to `validate()`.
    _dt = dt;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_domain(const DomainHostPtr<T>& domain) {
    // Store the optional domain subsystem in the builder.
    _domain = domain;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_codec(const CodecHostPtr<T>& codec) {
    // Store the optional codec subsystem in the builder.
    _codec = codec;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_source(const SourceHostPtr<T>& source) {
    // Store the optional source subsystem in the builder.
    _source = source;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sink(const SinkHostPtr<T>& sink) {
    // Store the optional sink subsystem in the builder.
    _sink = sink;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_measure(const MeasureHostPtr<T>& measure) {
    // Store the optional measurement subsystem in the builder.
    _measure = measure;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) {
    // Store the optional collider subsystem in the builder.
    _collider = collider;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const Collider<T>& collider) {
    // Convenience overload:
    // wrap a collider value into host-shared storage and store it
    // as the builder's collider subsystem.
    _collider = atlas::make_host_shared<Collider<T>>(collider);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solver(const OrchestratorHostPtr<T>& solver) {
    // Store the optional solver/orchestrator subsystem in the builder.
    _solver = solver;
    return *this;
}

template <typename T>
System<T>
System<T>::Builder::build() {
    // Validate builder state before constructing the final system object.
    validate();

    // Delegate construction to the full `System<T>` constructor so all probe
    // setup and default subsystem initialization stay centralized in one place.
    return System<T>(
        _fluid,
        _dt,
        _domain,
        _codec,
        _source,
        _sink,
        _measure,
        _collider,
        _solver);
}

template <typename T>
atlas::host_shared_ptr<System<T>>
System<T>::Builder::make_host_shared() {
    // Build the validated system by value and place it into host-shared storage.
    return atlas::make_host_shared<System<T>>(build());
}

template <typename T>
void
System<T>::Builder::validate() const {
    // Fluid is mandatory for any valid system configuration.
    if (!_fluid) {
        atlas::logger::error()
            << "System::Builder: fluid must be provided.";
        throw std::runtime_error("System::Builder: fluid must be provided.");
    }

    // Time step must be strictly positive.
    if (!(_dt > T(0))) {
        atlas::logger::error()
            << "System::Builder: dt must be positive.";
        throw std::runtime_error("System::Builder: dt must be positive.");
    }
}

} // namespace atlas::system