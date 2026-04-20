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
    // Store the fully prepared collider configuration.
    //
    // At this stage:
    // - _units contains the collider geometry and transform state on the device,
    // - _surface_interactions contains the device-side response models,
    // - _fluid points to the target fluid whose particles will be updated.
    //
    // No validation is performed here because the builder is expected to enforce
    // structural correctness before invoking this constructor.
}

template <typename T>
typename Collider<T>::Builder
Collider<T>::builder() noexcept {

    // Return a fresh builder instance for fluent construction.
    return Builder {};
}

template <typename T>
void
Collider<T>::update(const T dt) {

    // Reject degenerate updates early.
    //
    // There is nothing meaningful to do when:
    // - there are no collider units,
    // - or the time step is not strictly positive.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance every collider unit independently on the device.
    //
    // This step allows moving or rotating collider units to update their internal
    // sync operators before particle collisions are evaluated.
    //
    // Typical examples:
    // - animated rigid bodies,
    // - moving boundaries,
    // - rotating obstacles.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });

    // After unit transforms have been updated, resolve collisions against
    // the fluid particles for the same time step.
    collide(dt);
}

template <typename T>
void
Collider<T>::collide(const T dt) const {

    // Collision processing requires:
    // - a valid fluid object,
    // - a strictly positive time step,
    // - a non-empty collider configuration.
    if (!_fluid || !(dt > T(0)) || empty()) {
        return;
    }

    auto* position_state = _fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // A collision response needs both:
    // - particle positions, to define the motion segment,
    // - particle velocities, to define motion and compute post-collision velocity.
    //
    // If either state is missing, collision resolution cannot proceed safely.
    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }

    auto& positions  = position_state->data();
    auto& velocities = velocity_state->data();

    // Reject empty or inactive particle data.
    //
    // This avoids launching a device kernel when there is no particle work to do.
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
    // The particle's trajectory over the current step is approximated as a single
    // straight segment:
    //
    //   p(t) = p0 + velocity * dt
    //
    // This is effectively a swept segment test against all collider units.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [positions_ptr, velocities_ptr, dt, units, surface_interactions, unit_count, interaction_count] ATLAS_DEVICE(
            const int i) {
            // Current particle position at the beginning of the step.
            const Vector3<T> p0 = positions_ptr[i];

            // Current particle velocity.
            const Vector3<T> velocity = velocities_ptr[i];

            // Particle displacement over the current time step.
            const Vector3<T> direction = velocity * dt;

            // Length of the swept segment used for collision testing.
            const T segment_length = direction.length();

            // Skip nearly stationary particles.
            //
            // When the displacement is too small, the segment is too short to produce
            // a numerically meaningful sweep test.
            if (segment_length <= atlas::eps) {
                return;
            }

            // Track the closest valid collision, if any.
            bool any_hit = false;

            // best_t stores the distance from ray origin to the closest accepted hit.
            T best_t = atlas::far;

            // Best hit point in world space.
            Vector3<T> best_pos {};

            // Best hit normal in world space.
            Vector3<T> best_norm {};

            // Index of the collider unit that produced the closest hit.
            int best_index = -1;

            // Test this particle against every collider unit.
            //
            // The closest valid hit across all units is selected.
            for (int j = 0; j < unit_count; ++j) {
                const auto& unit = units[j];

                const auto& sync_op = unit.sync_operator();
                const auto& geom_op = unit.geometry_operator();

                // Construct a world-space ray that represents the particle motion
                // over the current step.
                //
                // The ray origin is the current particle position.
                // The ray direction is the finite displacement vector over dt.
                const atlas::spatial::Ray<T> world_ray(p0, direction);

                // Convert the motion ray into the collider unit's local space.
                //
                // This allows geometry intersection to be evaluated in the unit's
                // canonical local coordinate system, where the geometry operator
                // is typically defined.
                const atlas::spatial::Ray<T> local_ray = sync_op.sync_to_local(world_ray);

                // Query the local geometry for an intersection.
                const HitSurface<T> local_hit = geom_op(local_ray);

                // Reject invalid candidates:
                //
                // 1. no actual intersection,
                // 2. hit lies beyond the particle's swept segment length,
                // 3. hit is farther than the current best accepted hit.
                if (!local_hit.is_intersecting || local_hit.distance > segment_length
                    || local_hit.distance >= best_t) {
                    continue;
                }

                // Accept this hit as the new closest candidate.
                any_hit = true;
                best_t  = local_hit.distance;

                // Transform the local-space hit point back into world space.
                best_pos = sync_op.sync_to_world(local_hit.point);

                // Transform the local-space surface normal back into world space.
                //
                // Since normals are directional quantities, translation is not applied.
                best_norm = sync_op.sync_dir_to_world(local_hit.normal);

                best_index = j;
            }

            // If the swept segment does not hit any collider, simply advance the particle
            // along its original motion.
            if (!any_hit || best_index < 0) {
                positions_ptr[i]  = p0 + direction;
                velocities_ptr[i] = velocity;
                return;
            }

            // Select the surface interaction model to use for this collision.
            //
            // Policy:
            // - if only one interaction exists, use it for all units,
            // - if one interaction per unit exists, use the hit unit's interaction,
            // - if best_index exceeds interaction_count for any reason, fall back to 0.
            const int interaction_index = (interaction_count == 1 || best_index >= interaction_count) ? 0 : best_index;

            // Reposition the particle slightly outside the surface.
            //
            // This small offset helps reduce:
            // - immediate re-penetration,
            // - self-intersection on the next step,
            // - numerical sticking caused by finite precision.
            positions_ptr[i] = best_pos + best_norm * static_cast<T>(atlas::tol);

            // Compute the post-collision velocity using the chosen interaction model.
            //
            // The interaction model may encode behaviors such as:
            // - elastic reflection,
            // - restitution,
            // - tangential damping,
            // - friction-like projection.
            velocities_ptr[i] = surface_interactions[interaction_index](velocity, best_norm);
        });
}

template <typename T>
bool
Collider<T>::empty() const noexcept {

    // A collider is operational only when all three essential resources exist:
    // - at least one collider unit,
    // - at least one interaction model,
    // - a valid fluid object.
    return _units.empty() || _surface_interactions.empty() || !_fluid;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {

    // Reject invalid input immediately to keep the builder state meaningful.
    if (units.empty()) {
        throw std::runtime_error("Collider::Builder: units must not be empty.");
    }

    // Replace the current host-side collider unit set.
    _units = units;
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    // Store the target fluid that will later receive collision processing.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions) {

    // Reject explicitly empty interaction lists.
    //
    // If the caller wants default interaction insertion, the intended usage is
    // to omit this setter entirely rather than pass an empty buffer.
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    // Replace the current host-side interaction set.
    _surface_interactions = surface_interactions;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {

    // Ensure the builder configuration is structurally valid.
    validate();

    // If the caller did not specify any interaction model, install one default model.
    //
    // This guarantees that the final collider always has at least one valid
    // interaction function available during collision processing.
    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(ColliderSurfaceInteraction<T> {});
    }

    // Transfer host-side builder data into device-backed buffers used by Collider.
    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<ColliderSurfaceInteraction<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        _fluid);

    // Clear builder-owned state after successful construction.
    //
    // This prevents accidental reuse of stale configuration and makes the builder
    // behave more like a one-shot construction helper.
    _units.clear();
    _fluid.reset();
    _surface_interactions.clear();

    return collider;
}

template <typename T>
atlas::host_shared_ptr<Collider<T>>
Collider<T>::Builder::make_host_shared() {

    // Build a Collider value object first, then place it into host-managed shared storage.
    return atlas::make_host_shared<Collider<T>>(build());
}

template <typename T>
void
Collider<T>::Builder::validate() const {

    // A target fluid is mandatory because collision results are written directly
    // into the fluid particle states.
    if (!_fluid) {
        throw std::runtime_error("Collider::Builder: fluid must not be null.");
    }

    // At least one collider unit must exist, otherwise there is no collision geometry.
    if (_units.empty()) {
        throw std::runtime_error("Collider::Builder: at least one unit must be provided.");
    }

    // Surface interaction configuration rules:
    //
    // - empty: allowed here, because build() will inject a default interaction,
    // - size == 1: one shared interaction for all units,
    // - size == _units.size(): one interaction per unit.
    //
    // Any other count is ambiguous and therefore rejected.
    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != _units.size()) {
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }
}

} // namespace atlas::system