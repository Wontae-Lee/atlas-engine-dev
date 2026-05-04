#pragma once
#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <stdexcept>
#include <utility>
namespace atlas::system {
template <typename T>
Collider<T>::Collider(DeviceBuffer<Unit<T>> units,
                      DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions,
                      DeviceBuffer<std::uint8_t> flips,
                      atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept
    : _units(std::move(units))
    , _fluid(std::move(fluid))
    , _surface_interactions(std::move(surface_interactions))
    , _flips(std::move(flips)) {
}

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Collider<T>::update(const T dt) {
    if (_units.empty() || !(dt > T(0))) {
        return;
    }
    auto* units = atlas::raw_pointer_cast(_units.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
    collide(dt);
}

template <typename T>
void
Collider<T>::collide(const T dt) const {
    if (!(dt > T(0)) || empty()) {
        return;
    }
    ColliderProbe probe;
    if (!make_probe(probe)) {
        return;
    }
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<T> p0        = probe.positions[i];
            const Vector3<T> velocity  = probe.velocities[i];
            const Vector3<T> direction = velocity * dt;
            const T segment_length     = direction.length();
            if (segment_length <= atlas::eps) {
                return;
            }
            bool any_hit = false;
            T best_t     = atlas::far;
            Vector3<T> best_pos {};
            Vector3<T> best_norm {};
            int best_index = -1;
            for (int j = 0; j < probe.unit_count; ++j) {
                const auto& unit    = probe.units[j];
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
                probe.positions[i]  = p0 + direction;
                probe.velocities[i] = velocity;
                return;
            }
            const int interaction_index = (probe.interaction_count == 1 || best_index >= probe.interaction_count) ? 0 : best_index;
            const int flip_index        = (probe.flip_count == 1 || best_index >= probe.flip_count) ? 0 : best_index;
            const bool flip_normal      = probe.flip_count > 0 && probe.flips[flip_index] != std::uint8_t { 0 };
            const Vector3<T> hit_normal = flip_normal ? -best_norm : best_norm;
            const auto& hit_unit        = probe.units[best_index];
            Vector3<T> surface_velocity(T(0), T(0), T(0));
            if (hit_unit.velocity().has_value()) {
                surface_velocity += *hit_unit.velocity();
            }
            if (hit_unit.angular_velocity().has_value()) {
                const Vector3<T> radius = best_pos - hit_unit.sync_operator().translation;
                surface_velocity += atlas::math::cross(*hit_unit.angular_velocity(), radius);
            }
            const Vector3<T> relative_incident = velocity - surface_velocity;
            probe.positions[i]                 = best_pos + hit_normal * static_cast<T>(atlas::tol);
            probe.velocities[i]                = probe.surface_interactions[interaction_index](relative_incident, hit_normal) + surface_velocity;
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {
    return _units.empty() || _surface_interactions.empty() || !_fluid;
}

template <typename T>
bool
Collider<T>::make_probe(ColliderProbe& probe) const noexcept {
    if (empty()) {
        return false;
    }
    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    if (position_state == nullptr || velocity_state == nullptr) {
        return false;
    }
    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();
    if (positions.empty() || velocities.empty() || _fluid->particle_count() <= 0) {
        return false;
    }
    probe.units                = atlas::raw_pointer_cast(_units.data());
    probe.surface_interactions = atlas::raw_pointer_cast(_surface_interactions.data());
    probe.flips                = atlas::raw_pointer_cast(_flips.data());
    probe.positions            = atlas::raw_pointer_cast(positions.data());
    probe.velocities           = atlas::raw_pointer_cast(velocities.data());
    probe.unit_count           = static_cast<int>(_units.size());
    probe.interaction_count    = static_cast<int>(_surface_interactions.size());
    probe.flip_count           = static_cast<int>(_flips.size());
    probe.particle_count       = static_cast<int>(_fluid->particle_count());
    return true;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        throw std::runtime_error("Collider::Builder: units must not be empty.");
    }
    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }
    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_flip(const bool flip) noexcept {
    _flips.assign(1, flip ? std::uint8_t { 1 } : std::uint8_t { 0 });
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_flips(const HostBuffer<std::uint8_t>& flips) {
    if (flips.empty()) {
        throw std::runtime_error("Collider::Builder: flip flags must not be empty.");
    }
    _flips = flips;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    validate();
    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }
    if (_flips.empty()) {
        _flips.push_back(std::uint8_t { 0 });
    }
    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        DeviceBuffer<std::uint8_t>(_flips.begin(), _flips.end()),
        _fluid);
    _units.clear();
    _fluid.reset();
    _surface_interactions.clear();
    _flips.clear();
    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Collider<T>>(build());
}

template <typename T>
void
Collider<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Collider::Builder: fluid must not be null.");
    }
    if (_units.empty()) {
        throw std::runtime_error("Collider::Builder: at least one unit must be provided.");
    }
    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != _units.size()) {
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }
    if (!_flips.empty() && _flips.size() != 1 && _flips.size() != _units.size()) {
        throw std::runtime_error("Collider::Builder: flip count must be 1 or match unit count.");
    }
}

}