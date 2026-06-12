#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
Collider<T>::Collider(DeviceBuffer<Unit<T>> units,
                      DeviceBuffer<SurfaceInteractionKernel<T>> surface_interactions,
                      DeviceBuffer<std::uint8_t> flips,
                      PostColliderType post_collider_type,
                      atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept
    : _units(std::move(units))
    , _bound_cache(_units.size())
    , _fluid(std::move(fluid))
    , _surface_interactions(std::move(surface_interactions))
    , _flips(std::move(flips))
    , _post_collider_type(post_collider_type) {
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

    _bound_cache.refresh(_units);

    // Build a compact device-side probe containing raw pointers to collider, fluid, and interaction data.
    // If any required fluid state is missing, no collision pass is performed.
    if (!make_probe()) {
        return;
    }

    detail::ColliderCollisionKernel<T>::resolve_particles(_probe, _post_collider_type, dt);
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
    return detail::ColliderProbeBuilder<T>::make(_probe,
                                                 _units,
                                                 _bound_cache.unit_bounds(),
                                                 _surface_interactions,
                                                 _flips,
                                                 _bound_cache.scene_bound(),
                                                 _bound_cache.covers_units(),
                                                 _fluid);
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
    const HostBuffer<IsothermalSurfaceInteraction<T>>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions.clear();
    for (const auto& interaction : surface_interactions) {
        _surface_interactions.push_back(SurfaceInteractionKernel<T>(interaction));
    }
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interactions(
    const HostBuffer<MaxwellianSurfaceInteraction<T>>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions.clear();
    for (const auto& interaction : surface_interactions) {
        _surface_interactions.push_back(SurfaceInteractionKernel<T>(interaction));
    }
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction_kernel(
    const SurfaceInteractionKernel<T>& surface_interaction) {
    _surface_interactions.assign(1, surface_interaction);
    return *this;
}

template <typename T>
typename Collider<T>::Builder&
Collider<T>::Builder::with_surface_interaction_kernels(
    const HostBuffer<SurfaceInteractionKernel<T>>& surface_interactions) {
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
typename Collider<T>::Builder&
Collider<T>::Builder::with_post_collider_type(const PostColliderType type) noexcept {
    _post_collider_type = type;
    return *this;
}

template <typename T>
Collider<T>
Collider<T>::Builder::build() {
    // Validate the required builder inputs and per-unit array sizes before constructing the collider.
    validate();

    // Use the default surface interaction when the caller did not provide one explicitly.
    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(SurfaceInteractionKernel<T> {});
    }

    // Use unflipped normals by default when the caller did not provide flip flags.
    if (_flips.empty()) {
        _flips.push_back(std::uint8_t { 0 });
    }

    // Transfer host-side builder data into device buffers used by the runtime collider.
    Collider<T> collider(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<SurfaceInteractionKernel<T>>(_surface_interactions.begin(), _surface_interactions.end()),
        DeviceBuffer<std::uint8_t>(_flips.begin(), _flips.end()),
        _post_collider_type,
        _fluid);

    // Clear the builder after construction so it no longer retains stale configuration data.
    _units.clear();
    _fluid.reset();
    _surface_interactions.clear();
    _flips.clear();
    _post_collider_type = PostColliderType::fast;

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

} // namespace atlas
