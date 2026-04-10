#pragma once

#include <atlas/codec/single_codec.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename System<T>::Builder
System<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
System<T>::System(FluidHostPtr<T> fluid)

    : _fluid(std::move(fluid)) {
    if (!_fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    _particle_probe = _fluid->make_device_probe();
}

template <typename T>
System<T>::System(FluidHostPtr<T> fluid,
                  const T dt,
                  DomainHostPtr<T> domain,
                  CodecHostPtr<T> codec,
                  HostBuffer<SourceHostPtr<T>> sources,
                  HostBuffer<SinkHostPtr<T>> sinks,
                  HostBuffer<MeasureHostPtr<T>> measures,
                  HostBuffer<ColliderHostPtr<T>> colliders,
                  HostBuffer<SolverHostPtr<T>> solvers)

    : _fluid(std::move(fluid))
    , _dt(dt)
    , _domain(std::move(domain))
    , _codec(std::move(codec))
    , _particle_probe(_fluid->make_device_probe())
    , _sources(std::move(sources))
    , _sinks(std::move(sinks))
    , _measures(std::move(measures))
    , _colliders(std::move(colliders))
    , _solvers(std::move(solvers)) {
    if (!_fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    if (_domain) {
        _searcher       = atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);
        _domain_probe   = _domain->make_device_probe();
        _searcher_probe = _searcher->make_device_probe();
        if (!_codec) {
            _codec = atlas::make_host_shared<SingleCodec<T>>(_domain);
        }
    }

    if (_codec) {
        _codec_probe = _codec->make_device_probe();
    }
}

template <typename T>
void
System<T>::update() {
    atlas::logger::info() << "System::update: begin";

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

    for (const auto& source : _sources) {
        if (source) {
            source->emit(_particle_probe);
        }
    }

    atlas::logger::info() << "System::emit: end";
}

template <typename T>
void
System<T>::search() {
    atlas::logger::info() << "System::search: begin";

    if (!_searcher) {
        atlas::logger::info() << "System::search: end";
        return;
    }

    _searcher->build(_particle_probe);

    atlas::logger::info() << "System::search: end";
}

template <typename T>
void
System<T>::classify() {
    atlas::logger::info() << "System::classify: begin";

    if (!_codec || !_domain || !_searcher) {
        atlas::logger::info() << "System::classify: end";
        return;
    }

    _codec->update(_particle_probe, _domain_probe, _searcher_probe, _codec_probe);

    atlas::logger::info() << "System::classify: end";
}

template <typename T>
void
System<T>::measure() {
    atlas::logger::info() << "System::measure: begin";

    if (!_domain || !_searcher) {
        atlas::logger::info() << "System::measure: end";
        return;
    }

    for (const auto& measure : _measures) {
        if (measure) {
            measure->measure(_domain_probe, _searcher_probe, _particle_probe);
        }
    }

    atlas::logger::info() << "System::measure: end";
}

template <typename T>
void
System<T>::solve() {
    atlas::logger::info() << "System::solve: begin";

    if (!_domain || !_searcher || !_codec) {
        atlas::logger::info() << "System::solve: end";
        return;
    }

    for (const auto& solver : _solvers) {
        if (solver) {
            solver->solve(_domain_probe, _searcher_probe, _particle_probe, _codec_probe);
        }
    }

    atlas::logger::info() << "System::solve: end";
}

template <typename T>
void
System<T>::collide() const {
    atlas::logger::info() << "System::collide: begin";

    if (_particle_probe.empty() || !(_dt > T(0))) {
        atlas::logger::info() << "System::collide: end";
        return;
    }

    bool has_active_collider = false;

    for (const auto& collider : _colliders) {
        if (collider && !collider->empty()) {
            has_active_collider = true;
            collider->update(_dt);
            collider->collide(_particle_probe, _dt);
        }
    }

    if (!has_active_collider) {
        time_integration();
    }

    atlas::logger::info() << "System::collide: end";
}

template <typename T>
void
System<T>::remove() {
    atlas::logger::info() << "System::remove: begin";

    for (const auto& sink : _sinks) {
        if (sink) {
            sink->sink(_particle_probe);
        }
    }

    atlas::logger::info() << "System::remove: end";
}

template <typename T>
void
System<T>::time_integration() const {
    if (_particle_probe.empty() || !(_dt > T(0))) return;

    const auto probe = _particle_probe;
    const T dt       = _dt;

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
    if (!fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    _fluid          = std::move(fluid);
    _particle_probe = _fluid->make_device_probe();
}

template <typename T>
void
System<T>::set_dt(const T dt) {
    if (!(dt > T(0))) {
        atlas::logger::error()
            << "System: dt must be positive.";
        throw std::runtime_error("System: dt must be positive.");
    }

    _dt = dt;
}

template <typename T>
void
System<T>::set_domain(const DomainHostPtr<T>& domain) {
    if (!domain) {
        atlas::logger::error()
            << "System: domain must not be null.";
        throw std::runtime_error("System: domain must not be null.");
    }

    _domain         = domain;
    _searcher       = atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);
    _domain_probe   = _domain->make_device_probe();
    _searcher_probe = _searcher->make_device_probe();
    if (!_codec || _codec->type() == CodecType::single) {
        _codec       = atlas::make_host_shared<SingleCodec<T>>(_domain);
        _codec_probe = _codec->make_device_probe();
    }
}

template <typename T>
void
System<T>::set_codec(const CodecHostPtr<T>& codec) {
    if (!codec) {
        atlas::logger::error()
            << "System: codec must not be null.";
        throw std::runtime_error("System: codec must not be null.");
    }

    _codec       = codec;
    _codec_probe = _codec->make_device_probe();
}

template <typename T>
void
System<T>::add_source(const SourceHostPtr<T>& source) {
    _sources.push_back(source);
}

template <typename T>
void
System<T>::set_sources(const HostBuffer<SourceHostPtr<T>>& sources) {
    _sources = sources;
}

template <typename T>
void
System<T>::add_sink(const SinkHostPtr<T>& sink) {
    _sinks.push_back(sink);
}

template <typename T>
void
System<T>::set_sinks(const HostBuffer<SinkHostPtr<T>>& sinks) {
    _sinks = sinks;
}

template <typename T>
void
System<T>::add_measure(const MeasureHostPtr<T>& measure) {
    _measures.push_back(measure);
}

template <typename T>
void
System<T>::set_measures(const HostBuffer<MeasureHostPtr<T>>& measures) {
    _measures = measures;
}

template <typename T>
void
System<T>::add_collider(const ColliderHostPtr<T>& collider) {
    _colliders.push_back(collider);
}

template <typename T>
void
System<T>::add_collider(const Collider<T>& collider) {
    _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
}

template <typename T>
void
System<T>::set_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders) {
    _colliders = colliders;
}

template <typename T>
void
System<T>::set_colliders(const HostBuffer<Collider<T>>& colliders) {
    _colliders.clear();
    _colliders.reserve(colliders.size());

    for (const auto& collider : colliders) {
        _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
    }
}

template <typename T>
void
System<T>::add_solver(const SolverHostPtr<T>& solver) {
    _solvers.push_back(solver);
}

template <typename T>
void
System<T>::set_solvers(const HostBuffer<SolverHostPtr<T>>& solvers) {
    _solvers = solvers;
}

template <typename T>
FluidHostPtr<T>
System<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
FluidDeviceProbe<T>&
System<T>::particle_probe() noexcept {
    return _particle_probe;
}

template <typename T>
const FluidDeviceProbe<T>&
System<T>::particle_probe() const noexcept {
    return _particle_probe;
}

template <typename T>
DomainDeviceProbe<T>&
System<T>::domain_probe() noexcept {
    return _domain_probe;
}

template <typename T>
const DomainDeviceProbe<T>&
System<T>::domain_probe() const noexcept {
    return _domain_probe;
}

template <typename T>
SpatialHashingProbe<T>&
System<T>::searcher_probe() noexcept {
    return _searcher_probe;
}

template <typename T>
const SpatialHashingProbe<T>&
System<T>::searcher_probe() const noexcept {
    return _searcher_probe;
}

template <typename T>
CodecDeviceProbe<T>&
System<T>::codec_probe() noexcept {
    return _codec_probe;
}

template <typename T>
const CodecDeviceProbe<T>&
System<T>::codec_probe() const noexcept {
    return _codec_probe;
}

template <typename T>
T
System<T>::dt() const noexcept {
    return _dt;
}

template <typename T>
const DomainHostPtr<T>&
System<T>::domain() const noexcept {
    return _domain;
}

template <typename T>
const CodecHostPtr<T>&
System<T>::codec() const noexcept {
    return _codec;
}

template <typename T>
const HostBuffer<SourceHostPtr<T>>&
System<T>::sources() const noexcept {
    return _sources;
}

template <typename T>
const HostBuffer<SinkHostPtr<T>>&
System<T>::sinks() const noexcept {
    return _sinks;
}

template <typename T>
const HostBuffer<MeasureHostPtr<T>>&
System<T>::measures() const noexcept {
    return _measures;
}

template <typename T>
const HostBuffer<ColliderHostPtr<T>>&
System<T>::colliders() const noexcept {
    return _colliders;
}

template <typename T>
const HostBuffer<SolverHostPtr<T>>&
System<T>::solvers() const noexcept {
    return _solvers;
}

template <typename T>
void
System<T>::clear_sources() noexcept {
    _sources.clear();
}

template <typename T>
void
System<T>::clear_sinks() noexcept {
    _sinks.clear();
}

template <typename T>
void
System<T>::clear_measures() noexcept {
    _measures.clear();
}

template <typename T>
void
System<T>::clear_colliders() noexcept {
    _colliders.clear();
}

template <typename T>
void
System<T>::clear_solvers() noexcept {
    _solvers.clear();
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_dt(const T dt) noexcept {
    _dt = dt;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_domain(const DomainHostPtr<T>& domain) {
    if (!domain) {
        atlas::logger::error()
            << "System::Builder: domain must not be null.";
        throw std::runtime_error("System::Builder: domain must not be null.");
    }

    _domain = domain;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_codec(const CodecHostPtr<T>& codec) {
    if (!codec) {
        atlas::logger::error()
            << "System::Builder: codec must not be null.";
        throw std::runtime_error("System::Builder: codec must not be null.");
    }

    if (codec->type() != CodecType::single) {
        atlas::logger::error()
            << "System::Builder: only SingleCodec is supported by the builder at this time.";
        throw std::runtime_error(
            "System::Builder: only SingleCodec is supported by the builder at this time.");
    }

    _codec = codec;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_source(const SourceHostPtr<T>& source) {
    _sources.push_back(source);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sources(const HostBuffer<SourceHostPtr<T>>& sources) {
    _sources = sources;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sink(const SinkHostPtr<T>& sink) {
    _sinks.push_back(sink);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sinks(const HostBuffer<SinkHostPtr<T>>& sinks) {
    _sinks = sinks;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_measure(const MeasureHostPtr<T>& measure) {
    _measures.push_back(measure);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_measures(const HostBuffer<MeasureHostPtr<T>>& measures) {
    _measures = measures;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) {
    _colliders.push_back(collider);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const Collider<T>& collider) {
    _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders) {
    _colliders = colliders;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_colliders(const HostBuffer<Collider<T>>& colliders) {
    _colliders.clear();
    _colliders.reserve(colliders.size());

    for (const auto& collider : colliders) {
        _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
    }

    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solver(const SolverHostPtr<T>& solver) {
    _solvers.push_back(solver);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solvers(const HostBuffer<SolverHostPtr<T>>& solvers) {
    _solvers = solvers;
    return *this;
}

template <typename T>
System<T>
System<T>::Builder::build() {
    validate();

    System<T> system(_fluid, _dt, _domain, _codec, _sources, _sinks, _measures, _colliders, _solvers);
    _fluid  = nullptr;
    _dt     = static_cast<T>(0.01);
    _domain = nullptr;
    _codec  = nullptr;
    _sources.clear();
    _sinks.clear();
    _measures.clear();
    _colliders.clear();
    _solvers.clear();
    return system;
}

template <typename T>
atlas::host_shared_ptr<System<T>>
System<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<System<T>>(build());
}

template <typename T>
void
System<T>::Builder::validate() const {
    if (!_fluid) {
        atlas::logger::error()
            << "System::Builder: fluid must not be null.";
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }

    if (!(_dt > T(0))) {
        atlas::logger::error()
            << "System::Builder: dt must be positive.";
        throw std::runtime_error("System::Builder: dt must be positive.");
    }

    if (_codec && _codec->type() != CodecType::single) {
        atlas::logger::error()
            << "System::Builder: only SingleCodec is supported by the builder at this time.";
        throw std::runtime_error(
            "System::Builder: only SingleCodec is supported by the builder at this time.");
    }

    if (_domain && _domain->type() == DomainType::isothermal && !_measures.empty()) {
        atlas::logger::error()
            << "System::Builder: measures must be empty when the domain type is isothermal.";
        throw std::runtime_error("System::Builder: measures must be empty when the domain type is isothermal.");
    }
}

}
