#ifndef INCLUDE_ATLAS_SYSTEM_PARTICLE_SYSTEM_HPP
#define INCLUDE_ATLAS_SYSTEM_PARTICLE_SYSTEM_HPP

#include <atlas/logging/logging.h>
#include <atlas/memory/memory.h>
#include <atlas/system/particle_system.h>

namespace atlas::system {

template <typename T>
ParticleSystem<T>::ParticleSystem()
    : ParticleSystem<T>(1000) {
}

template <typename T>
ParticleSystem<T>::ParticleSystem(int capacity) {
    _particle_data = atlas::make_host_shared<ParticleData<T>>(capacity);
    _device_probe  = _particle_data->make_device_probe();
    ATLAS_INFO << "ParticleSystem created with capacity: " << capacity;
}

template <typename T>
const ParticleDataHostPtr<T>&
ParticleSystem<T>::particles() const {
    return _particle_data;
}

template <typename T>
void
ParticleSystem<T>::set_emitter(const Emitter<T>& emitter_) {
    _emitter = atlas::make_host_shared<Emitter<T>>(emitter_);
}

template <typename T>
void
ParticleSystem<T>::set_remover(const Remover<T>& remover_) {
    _remover = atlas::make_host_shared<Remover<T>>(remover_);
}

template <typename T>
void
ParticleSystem<T>::set_advector(const Advector<T>& advector_) {
    _advector = atlas::make_host_shared<Advector<T>>(advector_);
}

template <typename T>
void
ParticleSystem<T>::set_emitter(const EmitterHostPtr<T>& emitter_) {
    _emitter = emitter_;
}

template <typename T>
void
ParticleSystem<T>::set_remover(const RemoverHostPtr<T>& remover_) {
    _remover = remover_;
}

template <typename T>
void
ParticleSystem<T>::set_advector(const AdvectorHostPtr<T>& advector_) {
    _advector = advector_;
}

template <typename T>
void
ParticleSystem<T>::set_solver(const SolverHostPtr<T>& solver_) {
    _solver = std::move(solver_);
}

template <typename T>
const EmitterHostPtr<T>&
ParticleSystem<T>::emitter() const {
    return _emitter;
}

template <typename T>
const RemoverHostPtr<T>&
ParticleSystem<T>::remover() const {
    return _remover;
}

template <typename T>
const AdvectorHostPtr<T>&
ParticleSystem<T>::advector() const {
    return _advector;
}

template <typename T>
const SolverHostPtr<T>&
ParticleSystem<T>::solver() const {
    return _solver;
}

template <typename T>
void
ParticleSystem<T>::update() {
    this->emit(_device_probe, _active);
    ATLAS_INFO << "ParticleSystem emitted particles"
               << " Active: " << _active;

    this->advect(_device_probe, _dt, _active);
    ATLAS_INFO << "ParticleSystem collided particles"
               << " Active: " << _active;

    this->solve(_device_probe, _dt, _active);
    ATLAS_INFO << "ParticleSystem solved particles"
               << " Active: " << _active;

    this->remove(_device_probe, _active);
    ATLAS_INFO << "ParticleSystem removed particles"
               << " Active: " << _active;
}

template <typename T>
void
ParticleSystem<T>::emit(const ParticleDeviceProbe<T>& data, int& active) const {
    if (_emitter) {
        (*_emitter)(data, active);
    }
}

template <typename T>
void
ParticleSystem<T>::advect(const ParticleDeviceProbe<T>& data, T dt, int& active) const {
    if (_advector) {
        (*_advector)(data, dt, active);
    }
}

template <typename T>
void
ParticleSystem<T>::solve(const ParticleDeviceProbe<T>& data, T dt, int& active) const {
    if (_solver) {
        (*_solver)(data, dt, active);
    }
}

template <typename T>
void
ParticleSystem<T>::remove(const ParticleDeviceProbe<T>& data, int& active) const {
    if (_remover) {
        (*_remover)(data, active);
    }
}

template <typename T>
int
ParticleSystem<T>::capacity() const {
    return _capacity;
}

template <typename T>
int
ParticleSystem<T>::active() const {
    return _active;
}

}

#endif