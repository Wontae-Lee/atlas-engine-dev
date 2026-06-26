#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>

namespace atlas::detail {

template <typename T>
bool
DsmcProbeBuilder<T>::ready(const UniverseHostPtr<T>& universe,
                           const FluidHostPtr<T>& fluid,
                           const SearcherHostPtr<T>& searcher) noexcept {
    return universe != nullptr
        && fluid != nullptr
        && searcher != nullptr
        && fluid->template state<FluidVelocityState<T>>() != nullptr
        && fluid->template state<FluidSpeciesState<T>>() != nullptr
        && universe->template state<UniverseNumberParticleState<T>>() != nullptr
        && universe->template state<UniverseMaxRelativeSpeedState<T>>() != nullptr
        && universe->template state<UniverseMaxSigmaGState<T>>() != nullptr
        && universe->template state<UniverseCollisionRemainderState<T>>() != nullptr
        && universe->template state<UniverseCollisionCountState<int>>() != nullptr;
}

template <typename T>
bool
DsmcProbeBuilder<T>::make(DsmcProbe<T>& probe,
                          const UniverseHostPtr<T>& universe,
                          const FluidHostPtr<T>& fluid,
                          const SearcherHostPtr<T>& searcher,
                          const DsmcKernel<T>& kernel,
                          const std::uint64_t collision_seed) noexcept {
    probe = {};

    if (!ready(universe, fluid, searcher)) {
        return false;
    }

    probe.velocity_ptr = atlas::raw_pointer_cast(fluid->template state<FluidVelocityState<T>>()->data().data());
    if (auto* state = fluid->template state<FluidInternalEnergyState<T>>();
        state != nullptr && state->data().size() >= fluid->particle_count()) {
        probe.internal_energy_ptr = atlas::raw_pointer_cast(state->data().data());
    }
    probe.species_ptr             = atlas::raw_pointer_cast(fluid->template state<FluidSpeciesState<T>>()->data().data());
    probe.properties_ptr          = atlas::raw_pointer_cast(fluid->particle_properties().data());
    probe.number_particle_ptr     = atlas::raw_pointer_cast(universe->template state<UniverseNumberParticleState<T>>()->data().data());
    probe.max_relative_speed_ptr  = atlas::raw_pointer_cast(universe->template state<UniverseMaxRelativeSpeedState<T>>()->data().data());
    probe.max_sigma_g_ptr         = atlas::raw_pointer_cast(universe->template state<UniverseMaxSigmaGState<T>>()->data().data());
    probe.collision_remainder_ptr = atlas::raw_pointer_cast(universe->template state<UniverseCollisionRemainderState<T>>()->data().data());
    probe.collision_count_ptr     = atlas::raw_pointer_cast(universe->template state<UniverseCollisionCountState<int>>()->data().data());
    probe.indices_ptr             = searcher->indices();
    probe.cell_start_ptr          = searcher->cell_start();
    probe.cell_end_ptr            = searcher->cell_end();
    if (auto* state = universe->template state<UniverseVolumeState<T>>();
        state != nullptr && state->data().size() == static_cast<std::size_t>(universe->number_of_cells())) {
        probe.universe_volume_ptr = atlas::raw_pointer_cast(state->data().data());
    }
    probe.particle_count     = static_cast<int>(fluid->particle_count());
    probe.species_count      = static_cast<int>(fluid->particle_properties().size());
    probe.num_of_cells       = universe->number_of_cells();
    probe.cell_volume        = universe->cell_volume();
    probe.statistical_weight = fluid->statistical_weight();
    probe.kernel             = kernel;
    probe.collision_seed     = collision_seed;
    return true;
}

}