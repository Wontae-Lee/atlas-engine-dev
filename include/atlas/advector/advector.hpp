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
Advector<T>::time_integration(const ParticleDeviceProbe<T>& probe, T dt) const {
    auto alive = probe.alive;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        alive,
        [probe, dt] ATLAS_DEVICE(int i) {
            Vector3F p0       = probe.pos[i];
            Vector3F velocity = probe.vel[i];
            probe.pos[i]      = p0 + velocity * dt;
        }
        );
    return;
}

template <typename T>
void
Advector<T>::operator()(const ParticleDeviceProbe<T>& probe, T dt) const {
    int n_surfaces = _collider->number_of_surfaces();
    if (n_surfaces == 0) {
        time_integration(probe, dt);
        return;
    }
    _collider->collide(probe, dt);
}
}