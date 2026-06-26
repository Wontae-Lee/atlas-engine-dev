#pragma once

#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::detail {

template <typename T>
bool
ColliderProbeBuilder<T>::make(ColliderProbe<T>& probe,
                              const DeviceBuffer<Unit<T>>& units,
                              const DeviceBuffer<Bound>& unit_bounds,
                              const DeviceBuffer<SurfaceInteractionKernel<T>>& surface_interactions,
                              const DeviceBuffer<std::uint8_t>& flips,
                              const Bound& scene_bound,
                              const bool scene_bound_covers_units,
                              const FluidHostPtr<T>& fluid) noexcept {
    probe = {};

    if (units.empty() || surface_interactions.empty() || !fluid) {
        return false;
    }

    auto& positions             = fluid->template state<atlas::FluidPositionState<T>>()->data();
    auto& velocities            = fluid->template state<atlas::FluidVelocityState<T>>()->data();
    auto& species               = fluid->template state<atlas::FluidSpeciesState<T>>()->data();
    auto* internal_energy_state = fluid->template state<atlas::FluidInternalEnergyState<T>>();
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