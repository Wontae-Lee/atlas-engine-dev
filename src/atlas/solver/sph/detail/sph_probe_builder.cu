#include <atlas/solver/sph/detail/sph_probe_builder.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/solver/detail/solver_probe_common.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

bool
SphProbeBuilder::has_particle_states(const FluidHostPtr& fluid) noexcept {
    return fluid != nullptr
        && fluid->state<FluidPositionState>() != nullptr
        && fluid->state<FluidVelocityState>() != nullptr
        && fluid->state<FluidSpeciesState>() != nullptr;
}

bool
SphProbeBuilder::ready(const UniverseHostPtr& universe,
                       const FluidHostPtr& fluid,
                       const SearcherHostPtr& searcher) noexcept {
    return universe != nullptr
        && searcher != nullptr
        && has_particle_states(fluid)
        && universe->state<UniverseNumberParticleState>() != nullptr
        && universe->state<UniverseFieldForceState>() != nullptr;
}

bool
SphProbeBuilder::make(SphProbe& probe,
                      const UniverseHostPtr& universe,
                      const FluidHostPtr& fluid,
                      const SearcherHostPtr& searcher,
                      const SphKernel& kernel) noexcept {
    probe = {};

    if (!ready(universe, fluid, searcher)) {
        return false;
    }

    fill_common_solver_probe(probe, universe, fluid, searcher);

    probe.position_ptr         = atlas::raw_pointer_cast(fluid->state<FluidPositionState>()->data().data());
    probe.field_force_ptr      = atlas::raw_pointer_cast(universe->state<UniverseFieldForceState>()->data().data());
    probe.neighbor_offsets_ptr = searcher->neighbor_offsets();
    probe.neighbor_indices_ptr = searcher->neighbor_indices();
    probe.lower_corner         = searcher->lower_corner();
    probe.grid_size            = searcher->grid_size();
    probe.inverse_cell_size    = searcher->inverse_cell_size();
    probe.cell_size            = searcher->cell_size();
    probe.particle_count       = static_cast<int>(fluid->particle_count());
    probe.cell_count         = universe->cell_count();
    probe.property_count    = static_cast<int>(fluid->particle_properties().size());
    probe.kernel               = kernel;
    return true;
}

}
