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
    // Return a fresh builder so users can configure a collider through a fluent API.
    return Builder {};
}

template <typename T>
void
Collider<T>::update(const T dt) {
    // Skip invalid updates when there are no collider units or the time step is not positive.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    // Update the time-dependent state of each collider unit on the device.
    // This usually advances moving or rotating collider geometry before particle collision handling.
    auto* units = atlas::raw_pointer_cast(_units.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });

    // Apply particle-surface collisions after all collider units have been advanced.
    collide(dt);
}

template <typename T>
void
Collider<T>::collide(const T dt) const {
    // Collision handling is only meaningful for a positive time step and a fully configured collider.
    if (!(dt > T(0)) || empty()) {
        return;
    }

    // Build a compact device-side probe containing raw pointers to collider, fluid, and interaction data.
    // If any required fluid state is missing, no collision pass is performed.
    if (!make_probe()) {
        return;
    }
    const auto probe = _probe;

    // Process each particle independently on the device.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int i) {
            // Construct the particle trajectory over the current time step.
            const Vector3<T> p0        = probe.positions[i];
            const Vector3<T> velocity  = probe.velocities[i];
            const Vector3<T> direction = velocity * dt;
            const T segment_length     = direction.length();

            // Ignore nearly stationary particles because their swept segment is numerically degenerate.
            if (segment_length <= atlas::eps) {
                return;
            }

            // Track the closest valid surface intersection along the particle trajectory.
            bool any_hit = false;
            T best_t     = atlas::far;
            Vector3<T> best_pos {};
            Vector3<T> best_norm {};
            int best_index = -1;

            // Test the particle segment against every collider unit and keep only the nearest hit.
            for (int j = 0; j < probe.unit_count; ++j) {
                const auto& unit    = probe.units[j];
                const auto& sync_op = unit.sync_operator();
                const auto& geom_op = unit.geometry_operator();

                // Build the ray in world coordinates, then transform it into the local coordinate system
                // of the current collider unit before evaluating the geometry intersection.
                const atlas::spatial::Ray<T> world_ray(p0, direction);
                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);
                const HitSurface<T> local_hit          = geom_op(local_ray);

                // Reject missed intersections, hits beyond the current particle segment,
                // and hits farther than a previously found collision.
                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                // Store the closest intersection in world coordinates.
                any_hit    = true;
                best_t     = local_hit.distance;
                best_pos   = sync_op.sync_to_world(local_hit.point);
                best_norm  = sync_op.sync_dir_to_world(local_hit.normal);
                best_index = j;
            }

            // If no collider surface is reached during this time step, move the particle freely.
            if (!any_hit || best_index < 0) {
                probe.positions[i]  = p0 + direction;
                probe.velocities[i] = velocity;
                return;
            }

            // Select the surface-interaction model for the hit unit.
            // A single interaction entry is broadcast to all units; otherwise each unit uses its own entry.
            const int interaction_index = (probe.interaction_count == 1 || best_index >= probe.interaction_count) ? 0 : best_index;

            // Select the normal-flip flag for the hit unit.
            // A single flip entry is broadcast to all units; otherwise each unit uses its own entry.
            const int flip_index = (probe.flip_count == 1 || best_index >= probe.flip_count) ? 0 : best_index;

            // Optionally reverse the surface normal when the collider orientation requires it.
            const bool flip_normal      = probe.flip_count > 0 && probe.flips[flip_index] != std::uint8_t { 0 };
            const Vector3<T> hit_normal = flip_normal ? -best_norm : best_norm;

            // Compute the local surface velocity at the hit point.
            // This includes both translational motion and rotational motion of the collider unit.
            const auto& hit_unit = probe.units[best_index];
            Vector3<T> surface_velocity(T(0), T(0), T(0));

            // Add rigid-body translational velocity if the collider unit provides one.
            if (hit_unit.velocity().has_value()) {
                surface_velocity += *hit_unit.velocity();
            }

            // Add rotational surface velocity using omega x r when angular velocity is available.
            if (hit_unit.angular_velocity().has_value()) {
                const Vector3<T> radius = best_pos - hit_unit.sync_operator().translation;
                surface_velocity += atlas::math::cross(*hit_unit.angular_velocity(), radius);
            }

            // Convert the incoming particle velocity into the local moving-surface frame.
            const Vector3<T> relative_incident = velocity - surface_velocity;

            // Move the particle slightly outside the surface to avoid immediate self-intersection
            // in the next collision step.
            probe.positions[i] = best_pos + hit_normal * static_cast<T>(atlas::tol);

            // Apply the selected surface-interaction model in the surface frame, then transform
            // the reflected/emitted velocity back to the world frame by adding the surface velocity.
            probe.velocities[i] = probe.surface_interactions[interaction_index](relative_incident, hit_normal)
                + surface_velocity;
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {
    // A collider is considered unusable if it has no geometry, no surface interaction model,
    // or no associated fluid.
    return _units.empty() || _surface_interactions.empty() || !_fluid;
}

template <typename T>
bool
Collider<T>::make_probe() const noexcept {
    _probe = {};

    if (empty()) {
        return false;
    }

    auto& positions  = _fluid->template state<atlas::fluid::FluidPositionState<T>>()->data();
    auto& velocities = _fluid->template state<atlas::fluid::FluidVelocityState<T>>()->data();

    if (positions.empty() || velocities.empty() || _fluid->particle_count() <= 0) {
        return false;
    }

    _probe.units                = atlas::raw_pointer_cast(_units.data());
    _probe.surface_interactions = atlas::raw_pointer_cast(_surface_interactions.data());
    _probe.flips                = atlas::raw_pointer_cast(_flips.data());
    _probe.positions            = atlas::raw_pointer_cast(positions.data());
    _probe.velocities           = atlas::raw_pointer_cast(velocities.data());
    _probe.unit_count        = static_cast<int>(_units.size());
    _probe.interaction_count = static_cast<int>(_surface_interactions.size());
    _probe.flip_count        = static_cast<int>(_flips.size());
    _probe.particle_count    = static_cast<int>(_fluid->particle_count());

    return true;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    // Require at least one collider unit because an empty collider cannot perform intersections.
    if (units.empty()) {
        throw std::runtime_error("Collider::Builder: units must not be empty.");
    }

    // Store host-side units temporarily; they are copied to device memory during build().
    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {
    // Store the fluid shared pointer that provides particle position and velocity states.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    // Require at least one interaction model.
    // A single model can be broadcast to all collider units.
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_flip(const bool flip) noexcept {
    // Store one broadcast flip flag shared by all collider units.
    _flips.assign(1, flip ? std::uint8_t { 1 } : std::uint8_t { 0 });
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_flips(const HostBuffer<std::uint8_t>& flips) {
    // Require at least one flip flag.
    // One flag is broadcast to all units; otherwise the number of flags must match the unit count.
    if (flips.empty()) {
        throw std::runtime_error("Collider::Builder: flip flags must not be empty.");
    }

    _flips = flips;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    // Validate the required builder inputs and per-unit array sizes before constructing the collider.
    validate();

    // Use the default surface interaction when the caller did not provide one explicitly.
    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }

    // Use unflipped normals by default when the caller did not provide flip flags.
    if (_flips.empty()) {
        _flips.push_back(std::uint8_t { 0 });
    }

    // Transfer host-side builder data into device buffers used by the runtime collider.
    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        DeviceBuffer<std::uint8_t>(_flips.begin(), _flips.end()),
        _fluid);

    // Clear the builder after construction so it no longer retains stale configuration data.
    _units.clear();
    _fluid.reset();
    _surface_interactions.clear();
    _flips.clear();

    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {
    // Build a collider and wrap it in a host shared pointer for ownership-sharing workflows.
    return atlas::make_host_shared<Collider<T>>(build());
}

template <typename T>
void
Collider<T>::Builder::validate() const {
    // A collider must be attached to a valid fluid because particle states are read from it.
    if (!_fluid) {
        throw std::runtime_error("Collider::Builder: fluid must not be null.");
    }

    // At least one geometry unit is required for surface intersection tests.
    if (_units.empty()) {
        throw std::runtime_error("Collider::Builder: at least one unit must be provided.");
    }

    // Surface interactions are either broadcast from one entry or assigned one-to-one per unit.
    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != _units.size()) {
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }

    // Flip flags are either broadcast from one entry or assigned one-to-one per unit.
    if (!_flips.empty() && _flips.size() != 1 && _flips.size() != _units.size()) {
        throw std::runtime_error("Collider::Builder: flip count must be 1 or match unit count.");
    }
}

} // namespace atlas::system
