#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

template <typename Probe>
ATLAS_HOST void
fill_common_solver_probe(Probe& probe,
                         const UniverseHostPtr& universe,
                         const FluidHostPtr& fluid,
                         const SearcherHostPtr& searcher) noexcept {
    probe.velocity_ptr        = atlas::raw_pointer_cast(fluid->state<FluidVelocityState>()->data().data());
    probe.species_ptr         = atlas::raw_pointer_cast(fluid->state<FluidSpeciesState>()->data().data());
    probe.properties_ptr      = atlas::raw_pointer_cast(fluid->particle_properties().data());
    probe.number_particle_ptr = atlas::raw_pointer_cast(universe->state<UniverseNumberParticleState>()->data().data());

    probe.indices_ptr    = searcher->indices();
    probe.cell_start_ptr = searcher->cell_start();
    probe.cell_end_ptr   = searcher->cell_end();
}

}
