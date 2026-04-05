#pragma once

#include <atlas/logging/logging.h>
#include <atlas/parallel/parallel_for.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename Advector<T>::Builder
Advector<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Advector<T>::add_collider(const ColliderHostPtr<T>& collider) {
    if (!collider) {
        atlas::logger::error()
            << "Advector: collider must not be null.";
        throw std::runtime_error("Advector: collider must not be null.");
    }

    _colliders.push_back(collider);
}

template <typename T>
void
Advector<T>::add_collider(const Collider<T>& collider) {
    _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
}

template <typename T>
void
Advector<T>::set_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders) {
    for (const auto& collider : colliders) {
        if (!collider) {
            atlas::logger::error()
                << "Advector: collider list must not contain null pointers.";
            throw std::runtime_error("Advector: collider list must not contain null pointers.");
        }
    }

    _colliders = colliders;
}

template <typename T>
void
Advector<T>::clear_colliders() noexcept {
    _colliders.clear();
}

template <typename T>
const HostBuffer<ColliderHostPtr<T>>&
Advector<T>::colliders() const noexcept {
    return _colliders;
}

template <typename T>
void
Advector<T>::time_integration(const ParticleDeviceProbe<T>& probe, const T dt) const {
    if (probe.empty() || !(dt > T(0))) return;

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
Advector<T>::operator()(const ParticleDeviceProbe<T>& probe, const T dt) const {
    if (probe.empty() || !(dt > T(0))) return;

    if (_colliders.empty()) {
        time_integration(probe, dt);
        return;
    }

    const auto* collider_ptrs = _colliders.data();
    const int collider_count  = static_cast<int>(_colliders.size());
    const T far               = static_cast<T>(atlas::far);
    const T epsilon           = static_cast<T>(atlas::eps);
    const T tolerance         = epsilon;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [probe, dt, collider_ptrs, collider_count, far, epsilon, tolerance] ATLAS_DEVICE(const int i) {
            const Vector3<T> p0        = probe.pos[i];
            const Vector3<T> velocity  = probe.vel[i];
            const Vector3<T> direction = velocity * dt;
            const T segment_length     = direction.length();

            if (segment_length <= tolerance) {
                return;
            }

            bool any_hit = false;
            T best_t     = far;
            Vector3<T> best_pos {};
            Vector3<T> best_norm {};
            int best_index = -1;

            for (int j = 0; j < collider_count; ++j) {
                const auto& collider = collider_ptrs[j];
                if (!collider || !collider->unit() || !collider->surface_interaction()) continue;

                const auto& unit    = *collider->unit();
                const auto& sync_op = unit.sync_operator();
                const auto& geom_op = unit.geometry_operator();

                const atlas::spatial::Ray<T> world_ray(p0, direction);
                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);
                const HitSurface<T> local_hit          = geom_op(local_ray);

                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                any_hit    = true;
                best_t     = local_hit.distance;
                best_pos   = sync_op.sync_to_world(local_hit.point);
                best_norm  = sync_op.sync_dir_to_world(local_hit.normal);
                best_index = j;
            }

            if (!any_hit || best_index < 0) {
                probe.pos[i] = p0 + direction;
                probe.vel[i] = velocity;
                return;
            }

            const auto& interaction = *collider_ptrs[best_index]->surface_interaction();
            probe.pos[i]            = best_pos + best_norm * epsilon;
            probe.vel[i]            = interaction(velocity, best_norm);
        });
}

template <typename T>
typename Advector<T>::Builder&
Advector<T>::Builder::with_collider(const ColliderHostPtr<T>& collider) {
    if (!collider) {
        atlas::logger::error()
            << "Advector::Builder: collider must not be null.";
        throw std::runtime_error("Advector::Builder: collider must not be null.");
    }

    _colliders.push_back(collider);
    return *this;
}

template <typename T>
typename Advector<T>::Builder&
Advector<T>::Builder::with_collider(const Collider<T>& collider) {
    _colliders.push_back(atlas::make_host_shared<Collider<T>>(collider));
    return *this;
}

template <typename T>
typename Advector<T>::Builder&
Advector<T>::Builder::with_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders) {
    for (const auto& collider : colliders) {
        if (!collider) {
            atlas::logger::error()
                << "Advector::Builder: collider list must not contain null pointers.";
            throw std::runtime_error("Advector::Builder: collider list must not contain null pointers.");
        }
    }

    _colliders = colliders;
    return *this;
}

template <typename T>
Advector<T>
Advector<T>::Builder::build() {
    validate();

    Advector<T> advector {};
    advector.set_colliders(_colliders);
    _colliders.clear();
    return advector;
}

template <typename T>
atlas::host_shared_ptr<Advector<T>>
Advector<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Advector<T>>(build());
}

template <typename T>
void
Advector<T>::Builder::validate() const {
    for (const auto& collider : _colliders) {
        if (!collider) {
            atlas::logger::error()
                << "Advector::Builder: collider list must not contain null pointers.";
            throw std::runtime_error("Advector::Builder: collider list must not contain null pointers.");
        }
    }
}

}
