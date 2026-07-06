#include <atlas/orchestrator/detail/orchestrator_probe_builder.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/probe/probe_common.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

bool
OrchestratorProbeBuilder::make(const UniverseHostPtr& universe,
                               const FluidHostPtr& fluid,
                               const SearcherHostPtr& searcher,
                               OrchestratorProbe& probe) noexcept {
    probe = {};

    if (!load_common_data(universe, fluid, searcher, probe)) {
        return false;
    }

    load_species_data(fluid, probe);
    load_field_force_data(universe, probe);
    load_gravity_data(universe, probe);

    return true;
}

bool
OrchestratorProbeBuilder::load_common_data(const UniverseHostPtr& universe,
                                           const FluidHostPtr& fluid,
                                           const SearcherHostPtr& searcher,
                                           OrchestratorProbe& probe) noexcept {
    if (!universe || !fluid || !searcher) {
        return false;
    }

    auto* velocity_state = fluid->state<atlas::FluidVelocityState>();

    if (velocity_state == nullptr) {
        return false;
    }

    auto& velocity = velocity_state->data();

    probe.particle_count = static_cast<int>(fluid->particle_count());
    probe.cell_count   = universe->cell_count();
    probe.velocity_ptr   = atlas::raw_pointer_cast(velocity.data());

    fill_cell_partition(probe, searcher);

    return !velocity.empty()
        && probe.particle_count > 0
        && probe.cell_count > 0
        && probe.indices_ptr != nullptr
        && probe.cell_start_ptr != nullptr
        && probe.cell_end_ptr != nullptr;
}

void
OrchestratorProbeBuilder::load_species_data(const FluidHostPtr& fluid,
                                            OrchestratorProbe& probe) noexcept {
    auto* species_state = fluid->state<atlas::FluidSpeciesState>();

    if (species_state == nullptr) {
        return;
    }

    auto& species             = species_state->data();
    auto& particle_properties = fluid->particle_properties();

    if (species.empty() || particle_properties.empty()) {
        return;
    }

    probe.species_ptr    = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr = atlas::raw_pointer_cast(particle_properties.data());
    probe.species_count = static_cast<int>(particle_properties.size());
}

void
OrchestratorProbeBuilder::load_field_force_data(const UniverseHostPtr& universe,
                                                OrchestratorProbe& probe) noexcept {
    auto* field_force_state = universe->state<atlas::UniverseFieldForceState>();

    if (field_force_state == nullptr) {
        return;
    }

    auto& field_force = field_force_state->data();

    if (field_force.empty()) {
        return;
    }

    probe.field_force_ptr        = atlas::raw_pointer_cast(field_force.data());
    probe.field_force_cell_count = static_cast<int>(field_force.size());
}

void
OrchestratorProbeBuilder::load_gravity_data(const UniverseHostPtr& universe,
                                            OrchestratorProbe& probe) noexcept {
    auto* gravity_state = universe->state<atlas::UniverseGravityState>();

    if (gravity_state == nullptr) {
        return;
    }

    auto& gravity = gravity_state->data();

    if (gravity.empty()) {
        return;
    }

    probe.gravity_ptr        = atlas::raw_pointer_cast(gravity.data());
    probe.gravity_cell_count = static_cast<int>(gravity.size());
}

}
