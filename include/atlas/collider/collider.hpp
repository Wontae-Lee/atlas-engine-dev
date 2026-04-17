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
    // Store collider units, surface interaction models, and the target fluid.
}

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Collider construction.
    return Builder {};
}

template <typename T>
void
Collider<T>::update(const T dt) {

    // Nothing to update when there are no units or the time step is invalid.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance all collider units independently on the device.
    //
    // Units may internally update animated transforms, moving geometry state,
    // or other time-dependent properties relevant to collision queries.
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

    // Collision processing requires:
    // - a valid target fluid,
    // - a positive time step,
    // - non-empty collider configuration.
    if (!_fluid || !(dt > T(0)) || empty()) {
        return;
    }

    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Collision response requires both position and velocity state buffers.
    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }

    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();

    // Nothing to do if there is no particle storage or no active particles.
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

    // Process each particle independently on the device.
    //
    // The particle path over the current time step is approximated as the
    // segment from p0 to p0 + velocity * dt.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt, units, surface_interactions, unit_count, interaction_count] ATLAS_DEVICE(
            const int i) {
            const Vector3<T> p0 = positions_ptr[i];

            const Vector3<T> velocity = velocities_ptr[i];

            const Vector3<T> direction = velocity * dt;

            const T segment_length = direction.length();

            // Ignore particles whose motion over this step is too small to
            // produce a meaningful collision segment.
            if (segment_length <= atlas::tol) {
                return;
            }

            bool any_hit = false;

            T best_t = atlas::far;

            Vector3<T> best_pos {};

            Vector3<T> best_norm {};

            int best_index = -1;

            // Test the particle motion segment against every collider unit and
            // keep the closest valid hit.
            for (int j = 0; j < unit_count; ++j) {
                const auto& unit = units[j];

                const auto& sync_op = unit.sync_operator();

                const auto& geom_op = unit.geometry_operator();

                // Build a world-space ray from the current particle position
                // along the motion direction over this time step.
                const atlas::spatial::Ray<T> world_ray(p0, direction);

                // Convert the ray to the unit's local space so the unit's local
                // geometry operator can be queried consistently.
                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);

                const HitSurface<T> local_hit = geom_op(local_ray);

                // Reject:
                // - non-intersections
                // - hits beyond the particle motion segment
                // - hits farther away than the current best hit
                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                any_hit = true;
                best_t  = local_hit.distance;

                // Transform the hit point and surface normal back to world space.
                best_pos = sync_op.sync_to_world(local_hit.point);

                best_norm = sync_op.sync_dir_to_world(local_hit.normal);

                best_index = j;
            }

            // If no collision occurred, simply advance the particle.
            if (!any_hit || best_index < 0) {

                positions_ptr[i]  = p0 + direction;
                velocities_ptr[i] = velocity;
                return;
            }

            // Surface interaction models may be configured either:
            // - as a single shared interaction for all units, or
            // - one interaction per collider unit
            const int interaction_index = (interaction_count == 1 || best_index >= interaction_count) ? 0 : best_index;

            // Place the particle slightly outside the hit surface to reduce the
            // risk of immediate re-penetration due to numerical precision.
            positions_ptr[i] = best_pos + best_norm * static_cast<T>(atlas::eps);

            // Compute the post-collision velocity using the selected surface
            // interaction model.
            velocities_ptr[i] = surface_interactions[interaction_index](velocity, best_norm);
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {

    // A collider is effectively unusable when any essential dependency is missing.
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

    // Replace the current unit set with the provided collider units.
    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    // Store the target fluid to be processed by the collider.
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

    // Replace the current surface interaction set.
    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {

    // Validate structural consistency before creating the collider.
    validate();

    // If the user did not explicitly provide interaction models, install one
    // default interaction shared across all collider units.
    if (_surface_interactions.empty()) {

        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }

    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        _fluid);

    // Clear builder-owned state after successful construction.
    _units.clear();
    _fluid.reset();
    _surface_interactions.clear();

    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {

    // Build a value object first, then move it into host-managed shared storage.
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

    // Surface interactions must be either:
    // - omitted entirely, in which case a default one will be inserted later,
    // - a single shared interaction,
    // - or one interaction per unit.
    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != _units.size()) {
        atlas::logger::error()
            << "Collider::Builder: surface interaction count must be 1 or match unit count.";
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }
}

} // namespace atlas::system