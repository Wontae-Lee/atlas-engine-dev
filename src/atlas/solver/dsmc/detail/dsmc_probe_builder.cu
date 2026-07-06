#include <atlas/solver/dsmc/detail/dsmc_probe_builder.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/solver/detail/solver_probe_common.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>

namespace atlas::detail {

bool
DsmcProbeBuilder::ready(const UniverseHostPtr& universe,
                        const FluidHostPtr& fluid,
                        const SearcherHostPtr& searcher) noexcept {
    return universe != nullptr
        && fluid != nullptr
        && searcher != nullptr
        && fluid->state<FluidVelocityState>() != nullptr
        && fluid->state<FluidSpeciesState>() != nullptr
        && universe->state<UniverseNumberParticleState>() != nullptr
        && universe->state<UniverseMaxRelativeSpeedState>() != nullptr
        && universe->state<UniverseMaxSigmaGState>() != nullptr
        && universe->state<UniverseCollisionRemainderState>() != nullptr
        && universe->state<UniverseCollisionCountState>() != nullptr;
}

bool
DsmcProbeBuilder::make(DsmcProbe& probe,
                       const UniverseHostPtr& universe,
                       const FluidHostPtr& fluid,
                       const SearcherHostPtr& searcher,
                       const DsmcKernel& kernel,
                       const std::uint64_t collision_seed) noexcept {
    probe = {};

    if (!ready(universe, fluid, searcher)) {
        return false;
    }

    fill_common_solver_probe(probe, universe, fluid, searcher);

    if (auto* state = fluid->state<FluidInternalEnergyState>();
        state != nullptr && state->data().size() >= fluid->particle_count()) {
        probe.internal_energy_ptr = atlas::raw_pointer_cast(state->data().data());
    }
    probe.max_relative_speed_ptr  = atlas::raw_pointer_cast(universe->state<UniverseMaxRelativeSpeedState>()->data().data());
    probe.max_sigma_g_ptr         = atlas::raw_pointer_cast(universe->state<UniverseMaxSigmaGState>()->data().data());
    probe.collision_remainder_ptr = atlas::raw_pointer_cast(universe->state<UniverseCollisionRemainderState>()->data().data());
    probe.collision_count_ptr     = atlas::raw_pointer_cast(universe->state<UniverseCollisionCountState>()->data().data());
    if (auto* state = universe->state<UniverseVolumeState>();
        state != nullptr && state->data().size() == static_cast<std::size_t>(universe->cell_count())) {
        probe.universe_volume_ptr = atlas::raw_pointer_cast(state->data().data());
    }
    probe.particle_count     = static_cast<int>(fluid->particle_count());
    probe.species_count      = static_cast<int>(fluid->particle_properties().size());
    probe.cell_count         = universe->cell_count();
    probe.cell_volume        = universe->cell_volume();
    probe.statistical_weight = fluid->statistical_weight();
    probe.kernel             = kernel;
    probe.collision_seed     = collision_seed;
    return true;
}

}
