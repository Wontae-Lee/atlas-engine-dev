#pragma once

namespace atlas::detail {

template <typename T>
bool
OrchestratorProbeBuilder<T>::build(const UniverseHostPtr<T>& universe,
                                   const FluidHostPtr<T>& fluid,
                                   const SearcherHostPtr<T>& searcher,
                                   OrchestratorProbe<T>& probe) const noexcept {
    probe = {};

    if (!load_common_data(universe, fluid, searcher, probe)) {
        return false;
    }

    load_species_data(fluid, probe);
    load_field_force_data(universe, probe);
    load_gravity_data(universe, probe);

    return true;
}

template <typename T>
bool
OrchestratorProbeBuilder<T>::load_common_data(const UniverseHostPtr<T>& universe,
                                              const FluidHostPtr<T>& fluid,
                                              const SearcherHostPtr<T>& searcher,
                                              OrchestratorProbe<T>& probe) const noexcept {
    if (!universe || !fluid || !searcher) {
        return false;
    }

    auto* velocity_state = fluid->template state<atlas::FluidVelocityState<T>>();

    if (velocity_state == nullptr) {
        return false;
    }

    auto& velocity = velocity_state->data();

    probe.particle_count = static_cast<int>(fluid->particle_count());
    probe.num_of_cells   = universe->number_of_cells();
    probe.indices_ptr    = searcher->indices();
    probe.cell_start_ptr = searcher->cell_start();
    probe.cell_end_ptr   = searcher->cell_end();
    probe.velocity_ptr   = atlas::raw_pointer_cast(velocity.data());

    return !velocity.empty()
        && probe.particle_count > 0
        && probe.num_of_cells > 0
        && probe.indices_ptr != nullptr
        && probe.cell_start_ptr != nullptr
        && probe.cell_end_ptr != nullptr;
}

template <typename T>
void
OrchestratorProbeBuilder<T>::load_species_data(const FluidHostPtr<T>& fluid,
                                               OrchestratorProbe<T>& probe) const noexcept {
    auto* species_state = fluid->template state<atlas::FluidSpeciesState<T>>();

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
    probe.num_of_species = static_cast<int>(particle_properties.size());
}

template <typename T>
void
OrchestratorProbeBuilder<T>::load_field_force_data(const UniverseHostPtr<T>& universe,
                                                   OrchestratorProbe<T>& probe) const noexcept {
    auto* field_force_state = universe->template state<atlas::UniverseFieldForceState<T>>();

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

template <typename T>
void
OrchestratorProbeBuilder<T>::load_gravity_data(const UniverseHostPtr<T>& universe,
                                               OrchestratorProbe<T>& probe) const noexcept {
    auto* gravity_state = universe->template state<atlas::UniverseGravityState<T>>();

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