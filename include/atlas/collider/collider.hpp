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
                      atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept
    : _units(std::move(units))
    , _fluid(std::move(fluid))
    , _surface_interactions(std::move(surface_interactions)) {
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
}

template <typename T>
void
Collider<T>::collide(const T dt) const {

    if (!_fluid || !(dt > T(0)) || empty()) {
        return;
    }

    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }

    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();

    if (positions.empty() || velocities.empty() || _fluid->particle_count() <= 0) {
        return;
    }

    const auto* units                = atlas::raw_pointer_cast(_units.data());
    const auto* surface_interactions = atlas::raw_pointer_cast(_surface_interactions.data());
    auto* positions_ptr              = atlas::raw_pointer_cast(positions.data());
    auto* velocities_ptr             = atlas::raw_pointer_cast(velocities.data());

    const int unit_count        = static_cast<int>(_units.size());
    const int interaction_count = static_cast<int>(_surface_interactions.size());
    const int particle_count    = static_cast<int>(_fluid->particle_count());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt, units, surface_interactions, unit_count, interaction_count] ATLAS_DEVICE(
            const int i) {
            const Vector3<T> p0 = positions_ptr[i];

            const Vector3<T> velocity = velocities_ptr[i];

            const Vector3<T> direction = velocity * dt;

            const T segment_length = direction.length();

            if (segment_length <= atlas::tol) {
                return;
            }

            bool any_hit = false;

            T best_t = atlas::far;

            Vector3<T> best_pos {};

            Vector3<T> best_norm {};

            int best_index = -1;

            for (int j = 0; j < unit_count; ++j) {
                const auto& unit = units[j];

                const auto& sync_op = unit.sync_operator();

                const auto& geom_op = unit.geometry_operator();

                const atlas::spatial::Ray<T> world_ray(p0, direction);

                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);

                const HitSurface<T> local_hit = geom_op(local_ray);

                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                any_hit = true;
                best_t  = local_hit.distance;

                best_pos = sync_op.sync_to_world(local_hit.point);

                best_norm = sync_op.sync_dir_to_world(local_hit.normal);

                best_index = j;
            }

            if (!any_hit || best_index < 0) {

                positions_ptr[i]  = p0 + direction;
                velocities_ptr[i] = velocity;
                return;
            }

            const int interaction_index = (interaction_count == 1 || best_index >= interaction_count) ? 0 : best_index;

            positions_ptr[i] = best_pos + best_norm * static_cast<T>(atlas::eps);

            velocities_ptr[i] = surface_interactions[interaction_index](velocity, best_norm);
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {

    return _units.empty() || _surface_interactions.empty() || !_fluid;
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
Collider<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    _fluid = std::move(fluid);
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
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        _fluid);

    _units.clear();
    _fluid.reset();
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

    if (!_fluid) {
        atlas::logger::error()
            << "Collider::Builder: fluid must not be null.";
        throw std::runtime_error("Collider::Builder: fluid must not be null.");
    }

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
