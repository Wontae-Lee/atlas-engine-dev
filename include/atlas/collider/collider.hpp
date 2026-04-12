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
                      DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept
    : _units(std::move(units))
    , _surface_interactions(std::move(surface_interactions)) {
    // Construct the collider directly from device-resident unit storage
    // and device-resident surface-interaction storage.
    //
    // This constructor assumes the caller has already prepared backend-native
    // buffers and therefore performs no additional structural validation.
    //
    // Ownership of both buffers is transferred into the collider instance.
}

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {
    // Return a fresh builder object for staged collider construction.
    //
    // The builder path is useful when collision units and surface-interaction
    // policies are assembled incrementally before the final collider object
    // is materialized.
    return Builder {};
}

template <typename T>
void
Collider<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    // Replace the collider's current unit buffer with prebuilt device storage.
    //
    // This overload accepts already-materialized backend-native data and
    // therefore skips host-side emptiness or consistency checks.
    //
    // Ownership of the incoming device buffer is transferred to `_units`.
    _units = std::move(units);
}

template <typename T>
void
Collider<T>::set_units(const HostBuffer<Unit<T>>& units) {
    // Reject empty host input because a collider without any collision units
    // cannot participate meaningfully in collision queries.
    if (units.empty()) {
        atlas::logger::error()
            << "Collider: units must not be empty.";
        throw std::runtime_error("Collider: units must not be empty.");
    }

    // Convert host-authored unit data into device-resident storage so the
    // collider can be used by device-side update and collision kernels.
    _units = DeviceBuffer<Unit<T>>(units.begin(), units.end());
}

template <typename T>
void
Collider<T>::set_surface_interactions(DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept {
    // Replace the collider's current interaction-policy buffer with prebuilt
    // device storage.
    //
    // This overload trusts the caller to provide a correctly prepared buffer
    // and therefore performs no host-side validation here.
    //
    // Ownership of the incoming device buffer is transferred to `_surface_interactions`.
    _surface_interactions = std::move(surface_interactions);
}

template <typename T>
void
Collider<T>::set_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    // Reject empty host input because collision response requires at least
    // one valid surface-interaction policy.
    if (surface_interactions.empty()) {
        atlas::logger::error()
            << "Collider: surface interactions must not be empty.";
        throw std::runtime_error("Collider: surface interactions must not be empty.");
    }

    // Convert host-authored interaction policies into device-resident storage
    // so device kernels can apply collision response directly.
    _surface_interactions = DeviceBuffer<ColliderSurfaceInteraction<T>>(
        surface_interactions.begin(),
        surface_interactions.end());
}

template <typename T>
DeviceBuffer<Unit<T>>&
Collider<T>::units() noexcept {
    // Return mutable access to the device-resident collision units.
    //
    // Callers can use this to modify or replace unit data in place.
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Collider<T>::units() const noexcept {
    // Return read-only access to the device-resident collision units.
    return _units;
}

template <typename T>
DeviceBuffer<ColliderSurfaceInteraction<T>>&
Collider<T>::surface_interactions() noexcept {
    // Return mutable access to the device-resident surface-interaction policies.
    //
    // Callers can use this to edit or replace response behavior associated
    // with collider units.
    return _surface_interactions;
}

template <typename T>
const DeviceBuffer<ColliderSurfaceInteraction<T>>&
Collider<T>::surface_interactions() const noexcept {
    // Return read-only access to the device-resident surface-interaction policies.
    return _surface_interactions;
}

template <typename T>
void
Collider<T>::update(const T dt) {
    // Early exit when:
    // - there are no units to advance, or
    // - the time step is not strictly positive.
    //
    // In either case, collider transforms remain unchanged.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    // Expose the contiguous device buffer as a raw pointer for use inside
    // the device parallel kernel.
    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance every collider unit independently on the device.
    //
    // Purpose:
    // - keep time-dependent unit state current,
    // - update any pose/synchronization state used later for collision tracing,
    // - ensure geometry queries operate against the latest unit transform.
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
    // Early exit when:
    // - there are no particles to process,
    // - the time step is not strictly positive, or
    // - the collider is effectively inactive because required buffers are missing.
    if (particle_probe.particle_count <= 0 || !(dt > T(0)) || empty()) {
        return;
    }

    // Extract raw device pointers for kernel-side traversal.
    const auto* units                = atlas::raw_pointer_cast(_units.data());
    const auto* surface_interactions = atlas::raw_pointer_cast(_surface_interactions.data());

    // Cache counts and numerical constants outside the kernel launch so the
    // lambda capture remains compact and device-friendly.
    const int unit_count        = static_cast<int>(_units.size());
    const int interaction_count = static_cast<int>(_surface_interactions.size());
    const T far                 = static_cast<T>(atlas::far);
    const T epsilon             = static_cast<T>(atlas::eps);
    const T tolerance           = epsilon;

    // Process particle sweeps independently in parallel.
    //
    // Each particle performs a segment-style collision query over the time step:
    // - start point = current particle position
    // - sweep vector = velocity * dt
    //
    // The kernel finds the nearest valid hit among all collider units and
    // applies the associated surface response.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_probe.particle_count,
        [particle_probe, dt, units, surface_interactions, unit_count, interaction_count, far, epsilon, tolerance] ATLAS_DEVICE(const int i) {
            // Read the particle's current world-space position.
            const Vector3<T> p0 = particle_probe.pos[i];

            // Read the current particle velocity.
            const Vector3<T> velocity = particle_probe.vel[i];

            // Convert the time-step motion into a sweep vector in world space.
            const Vector3<T> direction = velocity * dt;

            // Treat the particle trajectory over the step as a line segment
            // whose length is the distance traveled in this step.
            const T segment_length = direction.length();

            // Ignore particles with effectively zero movement.
            //
            // Such particles do not define a meaningful sweep segment and
            // therefore cannot contribute to this segment-based trace.
            if (segment_length <= tolerance) {
                return;
            }

            // Track whether any collider unit was hit during this step.
            bool any_hit = false;

            // Track the closest valid hit distance found so far along the sweep.
            T best_t = far;

            // Store the best hit position in world space.
            Vector3<T> best_pos {};

            // Store the best hit normal in world space.
            Vector3<T> best_norm {};

            // Store the index of the collider unit that produced the best hit.
            int best_index = -1;

            // Test the particle sweep against every registered collision unit.
            for (int j = 0; j < unit_count; ++j) {
                const auto& unit = units[j];

                // The sync operator converts data between world space and the
                // unit's local space.
                const auto& sync_op = unit.sync_operator();

                // The geometry operator performs local-space intersection queries.
                const auto& geom_op = unit.geometry_operator();

                // Build the world-space ray representing the particle sweep.
                //
                // The direction vector is the full step displacement rather than
                // a normalized direction; intersection distances are therefore
                // interpreted consistently with that same parameterization.
                const atlas::spatial::Ray<T> world_ray(p0, direction);

                // Transform the world-space particle sweep into the unit's
                // local frame before querying the local geometry.
                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);

                // Intersect the local-space sweep against the unit geometry.
                const HitSurface<T> local_hit = geom_op(local_ray);

                // Discard this candidate when:
                // - there is no actual intersection,
                // - the intersection lies beyond the swept segment length,
                // - the intersection is farther than the best hit found so far.
                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                // Record this as the current nearest hit.
                any_hit = true;
                best_t  = local_hit.distance;

                // Convert the hit point back into world coordinates so the
                // particle state can be updated consistently in world space.
                best_pos = sync_op.sync_to_world(local_hit.point);

                // Convert the local surface normal back into world direction space.
                best_norm = sync_op.sync_dir_to_world(local_hit.normal);

                // Remember which unit produced the nearest hit.
                best_index = j;
            }

            if (!any_hit || best_index < 0) {
                // No collision occurred during this step.
                //
                // Complete the free-flight motion unchanged:
                // - advance position by the full step displacement
                // - preserve the existing velocity
                particle_probe.pos[i] = p0 + direction;
                particle_probe.vel[i] = velocity;
                return;
            }

            // Resolve which surface-interaction policy should be used.
            //
            // Supported mapping rules:
            // - if there is exactly one interaction, broadcast it to all units
            // - otherwise, use the interaction that matches the hit unit index
            // - if the unit index exceeds the available interaction range,
            //   fall back to interaction 0 as a defensive default
            const int interaction_index = (interaction_count == 1 || best_index >= interaction_count) ? 0 : best_index;

            // Move the particle slightly away from the surface along the hit normal.
            //
            // This small offset helps avoid immediate re-collision caused by
            // numerical precision issues or exact surface re-entry on the next step.
            particle_probe.pos[i] = best_pos + best_norm * epsilon;

            // Apply the configured surface-interaction model to compute the
            // post-collision velocity from the incoming velocity and surface normal.
            particle_probe.vel[i] = surface_interactions[interaction_index](velocity, best_norm);
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {
    // The collider is considered effectively empty if either:
    // - there are no collision units to trace against, or
    // - there are no surface-interaction policies to define the response.
    //
    // Both ingredients are required for a usable collision system.
    return _units.empty() || _surface_interactions.empty();
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    // Reject empty input because a collider requires at least one unit.
    if (units.empty()) {
        atlas::logger::error()
            << "Collider::Builder: units must not be empty.";
        throw std::runtime_error("Collider::Builder: units must not be empty.");
    }

    // Store the host-side unit list in the builder until final construction.
    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {
    // Reject empty input because explicitly provided interaction lists
    // must contain at least one valid policy.
    if (surface_interactions.empty()) {
        atlas::logger::error()
            << "Collider::Builder: surface interactions must not be empty.";
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    // Store the host-side interaction policies in the builder until final construction.
    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    // Validate builder state before materializing the collider.
    validate();

    if (_surface_interactions.empty()) {
        // If the caller did not provide any explicit surface interaction,
        // install a default-constructed interaction policy.
        //
        // This gives the collider a usable response model instead of leaving
        // it incomplete.
        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }

    // Convert the staged host-side buffers into device-resident buffers and
    // construct the final collider from those backend-native arrays.
    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()));

    // Clear the builder's host-side staging buffers after transfer so the
    // builder no longer retains duplicated construction data.
    _units.clear();
    _surface_interactions.clear();

    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {
    // Build the collider by value and place it into host-shared managed storage.
    return atlas::make_host_shared<Collider<T>>(build());
}

template <typename T>
void
Collider<T>::Builder::validate() const {
    // A collider requires at least one collision unit.
    if (_units.empty()) {
        atlas::logger::error()
            << "Collider::Builder: at least one unit must be provided.";
        throw std::runtime_error("Collider::Builder: at least one unit must be provided.");
    }

    // Validate the number of surface-interaction policies when they are explicitly provided.
    //
    // Supported configurations are:
    // - exactly 1 interaction  : broadcast the same response to every unit
    // - exactly N interactions : one interaction per unit
    //
    // Any other count would make unit-to-interaction mapping ambiguous.
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