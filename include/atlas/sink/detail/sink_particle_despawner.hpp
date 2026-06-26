#pragma once

namespace atlas::detail {

template <typename T>
void
SinkParticleDespawner<T>::apply(const SinkProbe<T>& probe, int* removed_unit_indices) const {
    const auto device_probe = probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        device_probe.particle_count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (device_probe.active[i] == 0) {
                if (removed_unit_indices != nullptr) {
                    removed_unit_indices[i] = -1;
                }
                return;
            }

            const Vector3<T>& position = device_probe.positions[i];
            bool should_despawn        = false;
            int matched_unit_index     = -1;

            for (int unit_index = 0; unit_index < device_probe.unit_count; ++unit_index) {
                const int despawn_operator_index
                    = (device_probe.despawn_operator_count == 1 || unit_index >= device_probe.despawn_operator_count)
                    ? 0
                    : unit_index;
                const auto& despawn_operator = device_probe.despawn_operators[despawn_operator_index];

                const auto& unit_bound = device_probe.unit_bounds[unit_index];
                if (unit_bound.is_valid()) {
                    if (despawn_operator.type == DespawnType::Tracing) {
                        if (device_probe.velocities == nullptr) {
                            continue;
                        }

                        const Vector3<T>& velocity = device_probe.velocities[i];
                        const T speed              = velocity.length();
                        if (!(device_probe.time_step > T(0)) || !(speed > T(0))) {
                            continue;
                        }

                        const auto bound_hit = unit_bound.trace(atlas::Ray<T>(position, velocity));
                        if (!bound_hit.is_intersecting || bound_hit.enter > speed * device_probe.time_step) {
                            continue;
                        }
                    } else if (!unit_bound.contains(position)) {
                        continue;
                    }
                }

                const auto& unit                = device_probe.units[unit_index];
                const auto& sync_op             = unit.sync_operator();
                const auto& geometry_op         = unit.geometry_operator();
                const Vector3<T> local_position = sync_op.sync_to_local(position);
                Vector3<T> despawn_vector       = local_position;
                T despawn_value                 = device_probe.tolerance;

                if (despawn_operator.type == DespawnType::Tracing) {
                    if (device_probe.velocities == nullptr) {
                        continue;
                    }
                    despawn_vector = sync_op.sync_dir_to_local(device_probe.velocities[i]);
                    despawn_value  = device_probe.time_step;
                }

                if (despawn_operator.despawn(geometry_op, local_position, despawn_vector, despawn_value)) {
                    should_despawn     = true;
                    matched_unit_index = unit_index;
                    break;
                }
            }

            const bool keep_particle = device_probe.flip ? should_despawn : !should_despawn;
            int recorded_unit_index  = matched_unit_index;
            if (!keep_particle && recorded_unit_index < 0 && device_probe.flip && device_probe.unit_count == 1) {
                recorded_unit_index = 0;
            }

            device_probe.active[i] = keep_particle ? 1 : 0;

            if (removed_unit_indices != nullptr) {
                removed_unit_indices[i] = keep_particle ? -1 : recorded_unit_index;
            }
        });
}

}