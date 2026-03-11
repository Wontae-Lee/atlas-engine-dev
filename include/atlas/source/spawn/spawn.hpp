#pragma once

namespace atlas::system {
template <typename T>
void
Spawn<T>::operator()(const DomainDeviceProbe<T>& domain,
                     const ParticleDeviceProbe<T>& particles,
                     const float dt) {
    if (!this->active) {
        return;
    }
    this->spawn(domain, particles, dt);
}

template <typename T>
void
Spawn<T>::activate() {
    this->active = true;
}

template <typename T>
void
Spawn<T>::deactivate() {
    this->active = false;
}

}