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

    if (!_fluid) {
        atlas::logger::error()
            << "System: fluid must not be null.";
        throw std::runtime_error("System: fluid must not be null.");
    }

    _particle_probe = _fluid->make_device_probe();

    if (_domain) {

        _searcher = atlas::make_host_shared<SpatialHashingSearcher<T>>(_domain);

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

    if (_source) {

        _source->update(_dt);

        _source->emit(_particle_probe);
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

    if (!_domain || !_searcher || !_measure) {

        atlas::logger::info() << "System::measure: end";
        return;
    }

    _measure->measure(_domain_probe, _searcher_probe, _particle_probe);

    atlas::logger::info() << "System::measure: end";
}

template <typename T>
void
System<T>::solve() {
    atlas::logger::info() << "System::solve: begin";

    if (!_domain || !_searcher || !_codec || !_solver) {

        atlas::logger::info() << "System::solve: end";
        return;
    }

    _solver->solve(_domain_probe, _searcher_probe, _particle_probe, _codec_probe);

    atlas::logger::info() << "System::solve: end";
}

template <typename T>
void
System<T>::collide() {
    atlas::logger::info() << "System::collide: begin";

    if (_particle_probe.particle_count <= 0 || !(_dt > T(0))) {

        atlas::logger::info() << "System::collide: end";
        return;
    }

    if (_collider && !_collider->empty()) {

        _collider->update(_dt);
        _collider->collide(_particle_probe, _dt);
    } else {

        time_integration();
    }

    atlas::logger::info() << "System::collide: end";
}

template <typename T>
void
System<T>::remove() {
    atlas::logger::info() << "System::remove: begin";

    if (_sink) {

        _sink->update(_dt);

        _sink->sink(_particle_probe);
    }

    atlas::logger::info() << "System::remove: end";
}

template <typename T>
void
System<T>::time_integration() {
    if (_particle_probe.particle_count <= 0 || !(_dt > T(0))) {

        return;
    }

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
            << "System: universe must not be null.";
        throw std::runtime_error("System: universe must not be null.");
    }

    _domain = domain;

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
System<T>::set_source(const SourceHostPtr<T>& source) {

    _source = source;
}

template <typename T>
void
System<T>::set_sink(const SinkHostPtr<T>& sink) {

    _sink = sink;
}

template <typename T>
void
System<T>::set_measure(const MeasureHostPtr<T>& measure) {

    _measure = measure;
}

template <typename T>
void
System<T>::set_collider(const ColliderHostPtr<T>& collider) {

    _collider = collider;
}

template <typename T>
void
System<T>::set_collider(const Collider<T>& collider) {

    _collider = atlas::make_host_shared<Collider<T>>(collider);
}

template <typename T>
void
System<T>::set_solver(const OrchestratorHostPtr<T>& solver) {

    _solver = solver;
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
Universe<T>&
System<T>::domain_probe() noexcept {

    return _domain_probe;
}

template <typename T>
const Universe<T>&
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
const SourceHostPtr<T>&
System<T>::source() const noexcept {

    return _source;
}

template <typename T>
const SinkHostPtr<T>&
System<T>::sink() const noexcept {

    return _sink;
}

template <typename T>
const MeasureHostPtr<T>&
System<T>::measure() const noexcept {

    return _measure;
}

template <typename T>
const ColliderHostPtr<T>&
System<T>::collider() const noexcept {

    return _collider;
}

template <typename T>
const OrchestratorHostPtr<T>&
System<T>::solver() const noexcept {

    return _solver;
}

template <typename T>
void
System<T>::clear_source() noexcept {

    _source.reset();
}

template <typename T>
void
System<T>::clear_sink() noexcept {

    _sink.reset();
}

template <typename T>
void
System<T>::clear_measure() noexcept {

    _measure.reset();
}

template <typename T>
void
System<T>::clear_collider() noexcept {

    _collider.reset();
}

template <typename T>
void
System<T>::clear_solver() noexcept {

    _solver.reset();
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

    _domain = domain;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_codec(const CodecHostPtr<T>& codec) {

    _codec = codec;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_source(const SourceHostPtr<T>& source) {

    _source = source;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_sink(const SinkHostPtr<T>& sink) {

    _sink = sink;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_measure(const MeasureHostPtr<T>& measure) {

    _measure = measure;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) {

    _collider = collider;
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_collider(const Collider<T>& collider) {

    _collider = atlas::make_host_shared<Collider<T>>(collider);
    return *this;
}

template <typename T>
typename System<T>::Builder&
System<T>::Builder::with_solver(const OrchestratorHostPtr<T>& solver) {

    _solver = solver;
    return *this;
}

template <typename T>
System<T>
System<T>::Builder::build() {

    validate();

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

    return atlas::make_host_shared<System<T>>(build());
}

template <typename T>
void
System<T>::Builder::validate() const {

    if (!_fluid) {
        atlas::logger::error()
            << "System::Builder: fluid must be provided.";
        throw std::runtime_error("System::Builder: fluid must be provided.");
    }

    if (!(_dt > T(0))) {
        atlas::logger::error()
            << "System::Builder: dt must be positive.";
        throw std::runtime_error("System::Builder: dt must be positive.");
    }
}

}