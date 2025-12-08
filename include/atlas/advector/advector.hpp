#pragma once
#include <atlas/parallel/parallel_for.h>

namespace atlas::system {
template <typename T>
void
Advector<T>::set_collider(const ColliderHostPtr<T>& collider) {
    _collider = std::move(collider);
}

template <typename T>
void
Advector<T>::set_collider(const Collider<T>& collider) {
    _collider = atlas::make_host_shared<Collider<T>>(collider);
}

template <typename T>
void
Advector<T>::time_integration(const ParticleDeviceProbe<T>& probe, T dt, int& active) const {
    auto device_position = probe.pos;
    auto device_velocity = probe.vel;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        active,
        [=] ATLAS_DEVICE(int i) {
            Vector3F p0        = device_position[i];
            Vector3F velocity  = device_velocity[i];
            device_position[i] = p0 + velocity * dt;
        });
    return;
}

template <typename T>
void
Advector<T>::operator()(const ParticleDeviceProbe<T>& probe, T dt, int& active) const {
    int n_surfaces = _collider->number_of_surfaces();
    if (n_surfaces == 0) {
        time_integration(probe, dt, active);
        return;
    }
    _collider->collide(probe, dt, active);
}
}