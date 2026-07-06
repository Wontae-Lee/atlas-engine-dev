#include <atlas/collider/detail/collider_probe_builder.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::detail {

bool
ColliderProbeBuilder::make(ColliderProbe& probe,
                           const DeviceBuffer<Unit>& units,
                           const DeviceBuffer<Bound>& unit_bounds,
                           const DeviceBuffer<SurfaceInteractionKernel>& surface_interactions,
                           const DeviceBuffer<std::uint8_t>& flips,
                           const Bound& scene_bound,
                           const bool scene_bound_covers_units,
                           const FluidHostPtr& fluid) noexcept {
    probe = {};

    if (units.empty() || surface_interactions.empty() || !fluid) {
        return false;
    }

    // Position/velocity/species states are required unconditionally (no
    // null check before ->data()): a Fluid intended for use with a Collider
    // is expected to always carry these three, unlike internal energy
    // below, which is genuinely optional.
    auto& positions             = fluid->state<atlas::FluidPositionState>()->data();
    auto& velocities            = fluid->state<atlas::FluidVelocityState>()->data();
    auto& species               = fluid->state<atlas::FluidSpeciesState>()->data();
    auto* internal_energy_state = fluid->state<atlas::FluidInternalEnergyState>();
    auto& materials             = fluid->particle_properties();

    if (positions.empty() || velocities.empty() || species.empty() || fluid->particle_count() <= 0) {
        return false;
    }

    probe.units                = atlas::raw_pointer_cast(units.data());
    probe.unit_bounds          = atlas::raw_pointer_cast(unit_bounds.data());
    probe.surface_interactions = atlas::raw_pointer_cast(surface_interactions.data());
    probe.flips                = atlas::raw_pointer_cast(flips.data());
    probe.scene_bound          = scene_bound;
    probe.positions            = atlas::raw_pointer_cast(positions.data());
    probe.velocities           = atlas::raw_pointer_cast(velocities.data());
    // internal_energies/species/materials are populated together only when
    // the fluid tracks internal energy AND has a non-empty material table;
    // otherwise they stay null/zero so DsmcEnergyExchangeSolver-style
    // internal-energy accommodation is simply skipped rather than the
    // whole probe build failing.
    if (internal_energy_state != nullptr
        && internal_energy_state->data().size() >= fluid->particle_count()
        && species.size() >= fluid->particle_count()
        && !materials.empty()) {
        probe.internal_energies = atlas::raw_pointer_cast(internal_energy_state->data().data());
        probe.species           = atlas::raw_pointer_cast(species.data());
        probe.materials         = atlas::raw_pointer_cast(materials.data());
        probe.material_count    = static_cast<int>(materials.size());
    }
    probe.unit_count               = static_cast<int>(units.size());
    probe.interaction_count        = static_cast<int>(surface_interactions.size());
    probe.flip_count               = static_cast<int>(flips.size());
    probe.particle_count           = static_cast<int>(fluid->particle_count());
    probe.scene_bound_covers_units = scene_bound_covers_units;

    return true;
}

}
