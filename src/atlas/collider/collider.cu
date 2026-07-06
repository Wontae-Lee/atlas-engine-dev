#include <atlas/collider/collider.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Collider::Collider(UniverseHostPtr universe,
                   DeviceBuffer<SurfaceInteractionKernel> surface_interactions,
                   DeviceBuffer<std::uint8_t> flips,
                   const PostColliderType post_collider_type,
                   atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _surface_interactions(std::move(surface_interactions))
    , _flips(std::move(flips))
    , _post_collider_type(post_collider_type) {
}

Collider::Builder
Collider::builder() noexcept {
    return Builder {};
}

void
Collider::update(const float dt) {
    if (!_universe || _universe->collider_units().empty() || !(dt > 0.0f)) {
        return;
    }

    // Units must finish moving before collide() sweeps particles against
    // them, otherwise the sweep would test against stale (pre-motion) unit
    // transforms for this step.
    _universe->collider_units().advance(dt);

    collide(dt);
}

void
Collider::collide(const float dt) const {
    if (!(dt > 0.0f) || empty()) {
        return;
    }

    // Bounds must be refreshed before make_probe() copies them into the
    // probe view, and before the collision sweep's broad-phase reject test
    // runs against them this step.
    _universe->collider_units().refresh_bounds();

    if (!make_probe()) {
        return;
    }

    detail::ColliderCollisionKernel::resolve_particles(_probe, _post_collider_type, dt);
}

bool
Collider::empty() const noexcept {
    return !_universe || _universe->collider_units().empty() || _surface_interactions.empty() || !_fluid;
}

bool
Collider::make_probe() const noexcept {
    const UnitField& unit_field = _universe->collider_units();

    return detail::ColliderProbeBuilder::make(_probe,
                                              unit_field.units(),
                                              unit_field.unit_bounds(),
                                              _surface_interactions,
                                              _flips,
                                              unit_field.scene_bound(),
                                              unit_field.covers_units(),
                                              _fluid);
}

Collider::Builder&
Collider::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

Collider::Builder&
Collider::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

Collider::Builder&
Collider::Builder::with_surface_interactions(
    const HostBuffer<IsothermalSurfaceInteraction>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions.clear();
    for (const auto& interaction : surface_interactions) {
        _surface_interactions.push_back(SurfaceInteractionKernel(interaction));
    }
    return *this;
}

Collider::Builder&
Collider::Builder::with_surface_interactions(
    const HostBuffer<MaxwellianSurfaceInteraction>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions.clear();
    for (const auto& interaction : surface_interactions) {
        _surface_interactions.push_back(SurfaceInteractionKernel(interaction));
    }
    return *this;
}

Collider::Builder&
Collider::Builder::with_surface_interaction_kernel(
    const SurfaceInteractionKernel& surface_interaction) {
    _surface_interactions.assign(1, surface_interaction);
    return *this;
}

Collider::Builder&
Collider::Builder::with_surface_interaction_kernels(
    const HostBuffer<SurfaceInteractionKernel>& surface_interactions) {
    if (surface_interactions.empty()) {
        throw std::runtime_error("Collider::Builder: surface interactions must not be empty.");
    }

    _surface_interactions = surface_interactions;
    return *this;
}

Collider::Builder&
Collider::Builder::with_flip(const bool flip) noexcept {
    _flips.assign(1, flip ? std::uint8_t { 1 } : std::uint8_t { 0 });
    return *this;
}

Collider::Builder&
Collider::Builder::with_flips(const HostBuffer<std::uint8_t>& flips) {
    if (flips.empty()) {
        throw std::runtime_error("Collider::Builder: flip flags must not be empty.");
    }

    _flips = flips;
    return *this;
}

Collider::Builder&
Collider::Builder::with_post_collider_type(const PostColliderType type) noexcept {
    _post_collider_type = type;
    return *this;
}

Collider
Collider::Builder::build() {
    validate();

    // No interaction/flip supplied means "use the default for every unit";
    // filling in a single broadcast entry here keeps ColliderProbe's
    // broadcast convention (count == 1) satisfied without special-casing
    // an empty buffer downstream.
    if (_surface_interactions.empty()) {
        _surface_interactions.push_back(SurfaceInteractionKernel {});
    }

    if (_flips.empty()) {
        _flips.push_back(std::uint8_t { 0 });
    }

    Collider collider(
        _universe,
        DeviceBuffer<SurfaceInteractionKernel>(_surface_interactions.begin(), _surface_interactions.end()),
        DeviceBuffer<std::uint8_t>(_flips.begin(), _flips.end()),
        _post_collider_type,
        _fluid);

    // Builder is single-use: clear so a stale copy of this state can't leak
    // into a second build() call from the same Builder instance.
    _universe.reset();
    _fluid.reset();
    _surface_interactions.clear();
    _flips.clear();
    _post_collider_type = PostColliderType::fast;

    return collider;
}

atlas::host_shared_ptr<Collider>
Collider::Builder::make_host_shared() {
    return atlas::make_host_shared<Collider>(build());
}

void
Collider::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Collider::Builder: fluid must not be null.");
    }

    if (!_universe) {
        throw std::runtime_error("Collider::Builder: universe must not be null.");
    }

    const std::size_t unit_count = _universe->collider_units().size();

    if (unit_count == 0) {
        throw std::runtime_error("Collider::Builder: universe must hold at least one collider unit.");
    }

    if (!_surface_interactions.empty()
        && _surface_interactions.size() != 1
        && _surface_interactions.size() != unit_count) {
        throw std::runtime_error(
            "Collider::Builder: surface interaction count must be 1 or match unit count.");
    }

    if (!_flips.empty() && _flips.size() != 1 && _flips.size() != unit_count) {
        throw std::runtime_error("Collider::Builder: flip count must be 1 or match unit count.");
    }
}

}
