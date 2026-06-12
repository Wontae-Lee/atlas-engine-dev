#pragma once

namespace atlas::detail {

template <typename T>
void
SourceEmitter<T>::emit(const SourceProbe<T>& probe,
                       const std::size_t dst_offset,
                       const std::size_t emit_count) const {
    const ShuffleOperator shuffle {};
    const auto device_probe = probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(emit_count),
        [=] ATLAS_DEVICE(const int i) {
            const int unit_index  = device_probe.flat_unit_indices[i];
            const int dst         = static_cast<int>(dst_offset) + i;
            const std::size_t sid = device_probe.shuffled_species[i];

            if (sid >= static_cast<std::size_t>(device_probe.property_count)) {
                return;
            }

            const auto sample_seed = static_cast<unsigned int>(shuffle(dst, device_probe.emission_seed));

            Vector3<T> world_pos;
            device_probe.units[unit_index].sync_operator().sync_to_world(
                device_probe.flat_local_positions[i],
                world_pos);

            device_probe.positions[dst]  = world_pos;
            device_probe.velocities[dst] = device_probe.generators[sid].generate(
                sample_seed,
                device_probe.temperature,
                device_probe.properties[sid].molecular_mass);
            device_probe.species[dst] = sid;
            device_probe.active[dst]  = 1;
        });
}

}