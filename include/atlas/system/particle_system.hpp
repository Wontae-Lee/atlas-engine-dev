#pragma once
#include <atlas/logging/logging.h>
#include <atlas/memory/memory.h>
#include <atlas/system/particle_system.h>

namespace atlas::system {

template <typename T>
ParticleSystem<T>::ParticleSystem()
    : ParticleSystem<T>(1000) {}

template <typename T>
ParticleSystem<T>::ParticleSystem(
    T dt,
    const EmitterHostPtr<T>& emitter,
    const RemoverHostPtr<T>& remover,
    const AdvectorHostPtr<T>& advector,
    const SolverHostPtr<T>& solver,
    int capacity)
    : _dt(dt)
      , _capacity(capacity)
      , _particle_data(atlas::make_host_shared<ParticleData<T>>(capacity))
      , _device_probe(_particle_data->make_device_probe())
      , _solver(solver)
      , _emitter(emitter)
      , _remover(remover)
      , _advector(advector) {

    ATLAS_INFO << "ParticleSystem created with capacity: " << capacity;
}

template <typename T>
const ParticleDataHostPtr<T>&
ParticleSystem<T>::particles() const {
    return _particle_data;
}

template <typename T>
void
ParticleSystem<T>::update() {
    if (_emitter) {
        (*_emitter)(_device_probe);
    }
    if (_solver) {
        (*_solver)(_device_probe, _dt);
    }
    if (_advector) {
        (*_advector)(_device_probe, _dt);
    }

    if (_remover) {
        (*_remover)(_device_probe);
    }
}

template <typename T>
int
ParticleSystem<T>::alive() const {
    return _device_probe.alive;
}

}