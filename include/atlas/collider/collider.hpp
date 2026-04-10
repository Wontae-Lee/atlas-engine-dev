#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/constants.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Collider<T>::Collider(Unit<T> unit,
                      ColliderSurfaceInteraction<T> surface_interaction) noexcept
    : _units(1, std::move(unit))
    , _surface_interactions(1, std::move(surface_interaction)) { }

template <typename T>
Collider<T>::Collider(
    UnitHostPtr<T> unit,
    atlas::host_shared_ptr<ColliderSurfaceInteraction<T>> surface_interaction)
    : Collider(unit ? *unit : Unit<T> {},
               surface_interaction ? *surface_interaction : ColliderSurfaceInteraction<T> {}) { }

template <typename T>
Collider<T>::Collider(DeviceBuffer<Unit<T>> units,
                      DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept
    : _units(std::move(units))
    , _surface_interactions(std::move(surface_interactions)) { }

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Collider<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    _units = std::move(units);
}

template <typename T>
void
Collider<T>::set_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        atlas::logger::error()
            << "Collider: units must not be empty.";
        throw std::runtime_error("Collider: units must not be empty.");
    }

    _units = DeviceBuffer<Unit<T>>(units.begin(), units.end());
}

template <typename T>
void
Collider<T>::set_unit(const Unit<T>& unit) {
    set_units(DeviceBuffer<Unit<T>>(1, unit));
}

template <typename T>
void
Collider<T>::set_unit(const UnitHostPtr<T>& unit) {
    if (!unit) {
        atlas::logger::error()
            << "Collider: unit must not be null.";
        throw std::runtime_error("Collider: unit must not be null.");
    }

    set_unit(*unit);
}

template <typename T>
void
Collider<T>::set_surface_interactions(DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept {
    _surface_interactions = std::move(surface_interactions);
}

template <typename T>
void
Collider<T>::set_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    if (surface_interactions.empty()) {
        atlas::logger::error()
            << "Collider: surface interactions must not be empty.";
        throw std::runtime_error("Collider: surface interactions must not be empty.");
    }

    _surface_interactions = DeviceBuffer<ColliderSurfaceInteraction<T>>(
        surface_interactions.begin(),
        surface_interactions.end());
}

template <typename T>
void
Collider<T>::set_surface_interaction(const ColliderSurfaceInteraction<T>& surface_interaction) {
    set_surface_interactions(DeviceBuffer<ColliderSurfaceInteraction<T>>(1, surface_interaction));
}

template <typename T>
void
Collider<T>::set_surface_interaction(
    const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>& surface_interaction) {
    if (!surface_interaction) {
        atlas::logger::error()
            << "Collider: surface interaction must not be null.";
        throw std::runtime_error("Collider: surface interaction must not be null.");
    }

    set_surface_interaction(*surface_interaction);
}

template <typename T>
DeviceBuffer<Unit<T>>&
Collider<T>::units() noexcept {
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Collider<T>::units() const noexcept {
    return _units;
}

template <typename T>
DeviceBuffer<ColliderSurfaceInteraction<T>>&
Collider<T>::surface_interactions() noexcept {
    return _surface_interactions;
}

template <typename T>
const DeviceBuffer<ColliderSurfaceInteraction<T>>&
Collider<T>::surface_interactions() const noexcept {
    return _surface_interactions;
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
}

template <typename T>
void
Collider<T>::collide(FluidDeviceProbe<T>& particle_probe, const T dt) const {
    if (particle_probe.empty() || !(dt > T(0)) || empty()) {
        return;
    }

    const auto* units                = atlas::raw_pointer_cast(_units.data());
    const auto* surface_interactions = atlas::raw_pointer_cast(_surface_interactions.data());
    const int unit_count             = static_cast<int>(_units.size());
    const int interaction_count      = static_cast<int>(_surface_interactions.size());
    const T far                      = static_cast<T>(atlas::far);
    const T epsilon                  = static_cast<T>(atlas::eps);
    const T tolerance                = epsilon;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_probe.particle_count,
        [particle_probe, dt, units, surface_interactions, unit_count, interaction_count, far, epsilon, tolerance] ATLAS_DEVICE(const int i) {
            const Vector3<T> p0        = particle_probe.pos[i];
            const Vector3<T> velocity  = particle_probe.vel[i];
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

            for (int j = 0; j < unit_count; ++j) {
                const auto& unit    = units[j];
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
                particle_probe.pos[i] = p0 + direction;
                particle_probe.vel[i] = velocity;
                return;
            }

            const int interaction_index = (interaction_count == 1 || best_index >= interaction_count) ? 0 : best_index;

            particle_probe.pos[i] = best_pos + best_norm * epsilon;
            particle_probe.vel[i] = surface_interactions[interaction_index](velocity, best_norm);
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {
    return _units.empty() || _surface_interactions.empty();
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_unit(const Unit<T>& unit) {
    _units.push_back(unit);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_unit(Unit<T>&& unit) {
    _units.push_back(std::move(unit));
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_unit(const UnitHostPtr<T>& unit) {
    if (!unit) {
        atlas::logger::error()
            << "Collider::Builder: unit must not be null.";
        throw std::runtime_error("Collider::Builder: unit must not be null.");
    }

    _units.push_back(*unit);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        atlas::logger::error()
            << "Collider::Builder: units must not be empty.";
        throw std::runtime_error("Collider::Builder: units must not be empty.");
    }

    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction(const ColliderSurfaceInteraction<T>& surface_interaction) {
    _surface_interactions.push_back(surface_interaction);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction(ColliderSurfaceInteraction<T>&& surface_interaction) {
    _surface_interactions.push_back(std::move(surface_interaction));
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction(
    const atlas::host_shared_ptr<ColliderSurfaceInteraction<T>>& surface_interaction) {
    if (!surface_interaction) {
        atlas::logger::error()
            << "Collider::Builder: surface interaction must not be null.";
        throw std::runtime_error("Collider::Builder: surface interaction must not be null.");
    }

    _surface_interactions.push_back(*surface_interaction);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    if (surface_interactions.empty()) {
        atlas::logger::error()
            << "Collider::Builder: surface interactions must not be empty.";
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    validate();

    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }

    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()));
    _units.clear();
    _surface_interactions.clear();
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
    if (_units.empty()) {
        atlas::logger::error()
            << "Collider::Builder: at least one unit must be provided.";
        throw std::runtime_error("Collider::Builder: at least one unit must be provided.");
    }

    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != _units.size()) {
        atlas::logger::error()
            << "Collider::Builder: surface interaction count must be 1 or match unit count.";
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }
}

}
