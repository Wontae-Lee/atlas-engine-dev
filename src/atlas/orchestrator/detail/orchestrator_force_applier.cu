#include <atlas/orchestrator/detail/orchestrator_force_applier.h>

#include <atlas/parallel/parallel_for.h>

namespace atlas::detail {

void
OrchestratorForceApplier::apply_all(const OrchestratorProbe& probe, const float dt) const {
    const bool gravity     = has_gravity(probe);
    const bool field_force = has_field_force(probe);

    if (!gravity && !field_force) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        force_cell_count(probe, gravity, field_force),
        [=] ATLAS_ALL_DEVICE(const int cell) {
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

            const Float3 gravity_value = gravity_cell ? probe.gravity_ptr[cell] : Float3();
            const Float3 force_value   = field_force_cell ? probe.field_force_ptr[cell] : Float3();

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                Float3 delta_velocity;
                bool update_velocity = false;

                if (gravity_cell) {
                    delta_velocity += gravity_value * dt;
                    update_velocity = true;
                }

                if (field_force_cell) {
                    const std::size_t species_index = probe.species_ptr[particle_index];

                    if (species_index < static_cast<std::size_t>(probe.species_count)) {
                        const float mass = probe.properties_ptr[species_index].mass;

                        if (mass > 0.0f) {
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

void
OrchestratorForceApplier::apply_gravity(const OrchestratorProbe& probe, const float dt) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.gravity_cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Float3 cell_gravity = probe.gravity_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += cell_gravity * dt;
            }
        });
}

void
OrchestratorForceApplier::apply_field_force(const OrchestratorProbe& probe, const float dt) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.field_force_cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Float3 force = probe.field_force_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                if (species_index >= static_cast<std::size_t>(probe.species_count)) {
                    continue;
                }

                const float mass = probe.properties_ptr[species_index].mass;

                if (!(mass > 0.0f)) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += force * (dt / mass);
            }
        });
}

bool
OrchestratorForceApplier::has_gravity(const OrchestratorProbe& probe) const noexcept {
    return probe.gravity_ptr != nullptr && probe.gravity_cell_count > 0;
}

bool
OrchestratorForceApplier::has_field_force(const OrchestratorProbe& probe) const noexcept {
    return probe.field_force_ptr != nullptr
        && probe.field_force_cell_count > 0
        && probe.species_ptr != nullptr
        && probe.properties_ptr != nullptr
        && probe.species_count > 0;
}

int
OrchestratorForceApplier::force_cell_count(const OrchestratorProbe& probe,
                                           const bool gravity,
                                           const bool field_force) const noexcept {
    int cell_count = gravity ? probe.gravity_cell_count : 0;

    if (field_force && probe.field_force_cell_count > cell_count) {
        cell_count = probe.field_force_cell_count;
    }

    if (cell_count > probe.cell_count) {
        cell_count = probe.cell_count;
    }

    return cell_count;
}

}
