#pragma once

namespace atlas::detail {

template <typename T>
void
OrchestratorForceApplier<T>::apply_all(const OrchestratorProbe<T>& probe, const T dt) const {
    const bool gravity     = has_gravity(probe);
    const bool field_force = has_field_force(probe);

    if (!gravity && !field_force) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        force_cell_count(probe, gravity, field_force),
        [=] ATLAS_DEVICE(const int cell) {
            const bool gravity_cell     = gravity && cell < probe.gravity_cell_count;
            const bool field_force_cell = field_force && cell < probe.field_force_cell_count;

            if (!gravity_cell && !field_force_cell) {
                return;
            }

            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Vector3<T> gravity_value = gravity_cell ? probe.gravity_ptr[cell] : Vector3<T>();
            const Vector3<T> force_value   = field_force_cell ? probe.field_force_ptr[cell] : Vector3<T>();

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                Vector3<T> delta_velocity;
                bool update_velocity = false;

                if (gravity_cell) {
                    delta_velocity += gravity_value * dt;
                    update_velocity = true;
                }

                if (field_force_cell) {
                    const std::size_t species_index = probe.species_ptr[particle_index];

                    if (species_index < static_cast<std::size_t>(probe.num_of_species)) {
                        const T mass = probe.properties_ptr[species_index].mass;

                        if (mass > T(0)) {
                            delta_velocity += force_value * (dt / mass);
                            update_velocity = true;
                        }
                    }
                }

                if (update_velocity) {
                    probe.velocity_ptr[particle_index] += delta_velocity;
                }
            }
        });
}

template <typename T>
void
OrchestratorForceApplier<T>::apply_gravity(const OrchestratorProbe<T>& probe, const T dt) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.gravity_cell_count,
        [=] ATLAS_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Vector3<T> cell_gravity = probe.gravity_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += cell_gravity * dt;
            }
        });
}

template <typename T>
void
OrchestratorForceApplier<T>::apply_field_force(const OrchestratorProbe<T>& probe, const T dt) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.field_force_cell_count,
        [=] ATLAS_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Vector3<T> force = probe.field_force_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                if (species_index >= static_cast<std::size_t>(probe.num_of_species)) {
                    continue;
                }

                const T mass = probe.properties_ptr[species_index].mass;

                if (!(mass > T(0))) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += force * (dt / mass);
            }
        });
}

template <typename T>
bool
OrchestratorForceApplier<T>::has_gravity(const OrchestratorProbe<T>& probe) const noexcept {
    return probe.gravity_ptr != nullptr && probe.gravity_cell_count > 0;
}

template <typename T>
bool
OrchestratorForceApplier<T>::has_field_force(const OrchestratorProbe<T>& probe) const noexcept {
    return probe.field_force_ptr != nullptr
        && probe.field_force_cell_count > 0
        && probe.species_ptr != nullptr
        && probe.properties_ptr != nullptr
        && probe.num_of_species > 0;
}

template <typename T>
int
OrchestratorForceApplier<T>::force_cell_count(const OrchestratorProbe<T>& probe,
                                              const bool gravity,
                                              const bool field_force) const noexcept {
    int cell_count = gravity ? probe.gravity_cell_count : 0;

    if (field_force && probe.field_force_cell_count > cell_count) {
        cell_count = probe.field_force_cell_count;
    }

    if (cell_count > probe.num_of_cells) {
        cell_count = probe.num_of_cells;
    }

    return cell_count;
}

}