#pragma once

namespace atlas::detail {

template <typename T>
bool
SourceProbeBuilder<T>::build(const FluidHostPtr<T>& fluid,
                             const DeviceBuffer<Unit<T>>& units,
                             const DeviceBuffer<std::size_t>& shuffled_species,
                             const DeviceBuffer<Vector3<T>>& flat_local_positions,
                             const DeviceBuffer<int>& flat_unit_indices,
                             const T temperature,
                             const std::uint64_t emission_seed,
                             SourceProbe<T>& probe) const noexcept {
    probe = {};

    if (!fluid || units.empty() || shuffled_species.empty()) {
        return false;
    }

    auto* position_state = fluid->template state<FluidPositionState<T>>();
    auto* velocity_state = fluid->template state<FluidVelocityState<T>>();
    auto* species_state  = fluid->template state<FluidSpeciesState<T>>();
    auto* active_state   = fluid->template state<FluidActiveState<T>>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr || active_state == nullptr) {
        return false;
    }

    auto& positions_buf        = position_state->data();
    auto& velocities_buf       = velocity_state->data();
    auto& species_buf          = species_state->data();
    auto& active_buf           = active_state->data();
    const auto& generators_buf = fluid->generators();
    const auto& properties_buf = fluid->particle_properties();

    if (positions_buf.empty() || velocities_buf.empty() || species_buf.empty() || active_buf.empty()
        || generators_buf.empty() || properties_buf.empty()
        || flat_local_positions.empty() || flat_unit_indices.empty()) {
        return false;
    }

    probe.units                = atlas::raw_pointer_cast(units.data());
    probe.generators           = atlas::raw_pointer_cast(generators_buf.data());
    probe.properties           = atlas::raw_pointer_cast(properties_buf.data());
    probe.shuffled_species     = atlas::raw_pointer_cast(shuffled_species.data());
    probe.positions            = atlas::raw_pointer_cast(positions_buf.data());
    probe.velocities           = atlas::raw_pointer_cast(velocities_buf.data());
    probe.species              = atlas::raw_pointer_cast(species_buf.data());
    probe.active               = atlas::raw_pointer_cast(active_buf.data());
    probe.flat_local_positions = atlas::raw_pointer_cast(flat_local_positions.data());
    probe.flat_unit_indices    = atlas::raw_pointer_cast(flat_unit_indices.data());
    probe.temperature          = temperature;
    probe.property_count       = static_cast<int>(properties_buf.size());
    probe.emission_seed        = emission_seed;

    return true;
}

} // namespace atlas::detail
