#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

template <typename T>
bool
SphProbeBuilder<T>::has_particle_states(const FluidHostPtr<T>& fluid) noexcept {
    return fluid != nullptr
        && fluid->template state<FluidPositionState<T>>() != nullptr
        && fluid->template state<FluidVelocityState<T>>() != nullptr
        && fluid->template state<FluidSpeciesState<T>>() != nullptr;
}

template <typename T>
bool
SphProbeBuilder<T>::ready(const UniverseHostPtr<T>& universe,
                          const FluidHostPtr<T>& fluid,
                          const SearcherHostPtr<T>& searcher) noexcept {
    return universe != nullptr
        && searcher != nullptr
        && has_particle_states(fluid)
        && universe->template state<UniverseNumberParticleState<T>>() != nullptr
        && universe->template state<UniverseFieldForceState<T>>() != nullptr;
}

template <typename T>
bool
SphProbeBuilder<T>::make(SphProbe<T>& probe,
                         const UniverseHostPtr<T>& universe,
                         const FluidHostPtr<T>& fluid,
                         const SearcherHostPtr<T>& searcher,
                         const SphKernel<T>& kernel) noexcept {
    probe = {};

    if (!ready(universe, fluid, searcher)) {
        return false;
    }

    probe.position_ptr         = atlas::raw_pointer_cast(fluid->template state<FluidPositionState<T>>()->data().data());
    probe.velocity_ptr         = atlas::raw_pointer_cast(fluid->template state<FluidVelocityState<T>>()->data().data());
    probe.species_ptr          = atlas::raw_pointer_cast(fluid->template state<FluidSpeciesState<T>>()->data().data());
    probe.properties_ptr       = atlas::raw_pointer_cast(fluid->particle_properties().data());
    probe.number_particle_ptr  = atlas::raw_pointer_cast(universe->template state<UniverseNumberParticleState<T>>()->data().data());
    probe.field_force_ptr      = atlas::raw_pointer_cast(universe->template state<UniverseFieldForceState<T>>()->data().data());
    probe.indices_ptr          = searcher->indices();
    probe.cell_start_ptr       = searcher->cell_start();
    probe.cell_end_ptr         = searcher->cell_end();
    probe.neighbor_offsets_ptr = searcher->neighbor_offsets();
    probe.neighbor_indices_ptr = searcher->neighbor_indices();
    probe.lower_corner         = searcher->lower_corner();
    probe.grid_size            = searcher->grid_size();
    probe.inverse_cell_size    = searcher->inverse_cell_size();
    probe.cell_size            = searcher->cell_size();
    probe.particle_count       = static_cast<int>(fluid->particle_count());
    probe.num_of_cells         = universe->number_of_cells();
    probe.num_of_properties    = static_cast<int>(fluid->particle_properties().size());
    probe.kernel               = kernel;
    return true;
}

}