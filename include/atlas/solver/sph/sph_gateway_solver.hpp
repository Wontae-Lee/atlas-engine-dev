#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
SphGatewaySolver<T>::SphGatewaySolver(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher,
                                      const SphKernelType kernel_type,
                                      const int group_particle_count) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(kernel_type)
    , _group_particle_count(group_particle_count > 0 ? group_particle_count : 5) {
    // Ensure per-cell output states exist when a universe is already attached.
    ensure_states();
}

template <typename T>
typename SphGatewaySolver<T>::Builder
SphGatewaySolver<T>::builder() noexcept {
    // Return a fresh builder for fluent solver construction.
    return Builder {};
}

template <typename T>
SphKernelType
SphGatewaySolver<T>::kernel_type() const noexcept {
    // Expose the kernel type stored by the runtime kernel wrapper.
    return _kernel.type;
}

template <typename T>
int
SphGatewaySolver<T>::group_particle_count() const noexcept {
    // Return the configured target number of particles per representative group.
    return _group_particle_count;
}

template <typename T>
void
SphGatewaySolver<T>::solve(const T dt) {
    // Run the grouped SPH solver over all cells.
    solve(nullptr, 0, dt);
}

template <typename T>
void
SphGatewaySolver<T>::solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    // Validate dependencies, required fluid states, universe states, and searcher ordering.
    if (!initialize_context()) {
        return;
    }

    // Representative velocity integration requires a positive time step.
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphGatewaySolver: dt must be positive.");
    }

    // Allocate transient group buffers; representative slots are overwritten by later stages.
    if (!prepare_group_fields()) {
        reset_universe_fields();
        return;
    }

    // Refresh one shared SPH probe and reuse it across all grouped stages.
    if (!make_probe()) {
        return;
    }

    // Run the grouped-SPH pipeline.
    update_cell_particle_counts(allocated_solver, index);
    build_group_representatives(allocated_solver, index);
    estimate_group_density_and_pressure(allocated_solver, index);
    update_group_motion(allocated_solver, index, dt);
    scatter_group_states_to_particles(allocated_solver, index);
}

template <typename T>
void
SphGatewaySolver<T>::ensure_states() {
    // Nothing can be initialized without a universe.
    if (!this->_universe) {
        return;
    }

    // Ensure per-cell output buffers exist.
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(
            number_of_cells);
    }

    if (!this->_universe->template has_state<atlas::universe::UniverseFieldForceState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseFieldForceState<T>>(
            number_of_cells);
    }
}

template <typename T>
bool
SphGatewaySolver<T>::initialize_context() noexcept {
    // Required runtime dependencies must exist before launching grouped-SPH kernels.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_universe_fields();

        // Clear transient buffers to avoid carrying stale group data.
        _cell_group_count.resize(0);
        _group_position.resize(0);
        _group_velocity.resize(0);
        _group_updated_position.resize(0);
        _group_updated_velocity.resize(0);
        _group_mass.resize(0);
        _group_density.resize(0);
        _group_pressure.resize(0);
        _group_member_count.resize(0);
        _group_species.resize(0);

        return false;
    }

    // Grouped SPH requires particle position, velocity, and species states.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_universe_fields();

        // Clear transient buffers when the fluid state is incomplete.
        _cell_group_count.resize(0);
        _group_position.resize(0);
        _group_velocity.resize(0);
        _group_updated_position.resize(0);
        _group_updated_velocity.resize(0);
        _group_mass.resize(0);
        _group_density.resize(0);
        _group_pressure.resize(0);
        _group_member_count.resize(0);
        _group_species.resize(0);

        return false;
    }

    // Ensure output states exist and rebuild spatial hashing for the current particles.
    ensure_states();
    this->_searcher->build();

    return true;
}

template <typename T>
bool
SphGatewaySolver<T>::prepare_group_fields() {
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells   = this->_universe->number_of_cells();

    // No particle or no cell means there is no grouped update to perform.
    if (particle_count <= 0 || num_of_cells <= 0) {
        _cell_group_count.resize(0);
        _group_position.resize(0);
        _group_velocity.resize(0);
        _group_updated_position.resize(0);
        _group_updated_velocity.resize(0);
        _group_mass.resize(0);
        _group_density.resize(0);
        _group_pressure.resize(0);
        _group_member_count.resize(0);
        _group_species.resize(0);

        reset_universe_fields();
        return false;
    }

    // Allocate one group-count entry per cell and one representative slot per particle.
    _cell_group_count.resize(static_cast<std::size_t>(num_of_cells));
    _group_position.resize(static_cast<std::size_t>(particle_count));
    _group_velocity.resize(static_cast<std::size_t>(particle_count));
    _group_updated_position.resize(static_cast<std::size_t>(particle_count));
    _group_updated_velocity.resize(static_cast<std::size_t>(particle_count));
    _group_mass.resize(static_cast<std::size_t>(particle_count));
    _group_density.resize(static_cast<std::size_t>(particle_count));
    _group_pressure.resize(static_cast<std::size_t>(particle_count));
    _group_member_count.resize(static_cast<std::size_t>(particle_count));
    _group_species.resize(static_cast<std::size_t>(particle_count));

    // Clear universe outputs for this step.
    reset_universe_fields();

    return true;
}

template <typename T>
void
SphGatewaySolver<T>::reset_universe_fields() {
    // Nothing to reset when no universe is attached.
    if (!this->_universe) {
        return;
    }

    // Ensure reset targets exist.
    ensure_states();

    auto* number_particle_state = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* field_force_state     = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();

    // Reset per-cell particle counts.
    if (number_particle_state != nullptr) {
        number_particle_state->reset();
    }

    // Reset per-cell averaged force output.
    if (field_force_state != nullptr) {
        field_force_state->reset();
    }
}

template <typename T>
bool
SphGatewaySolver<T>::make_probe() noexcept {
    _probe = {};

    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return false;
    }

    _probe.position_ptr        = atlas::raw_pointer_cast(this->_fluid->template state<atlas::fluid::FluidPositionState<T>>()->data().data());
    _probe.velocity_ptr        = atlas::raw_pointer_cast(this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>()->data().data());
    _probe.species_ptr         = atlas::raw_pointer_cast(this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>()->data().data());
    _probe.properties_ptr      = atlas::raw_pointer_cast(this->_fluid->particle_properties().data());
    _probe.number_particle_ptr = atlas::raw_pointer_cast(this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>()->data().data());
    _probe.field_force_ptr     = atlas::raw_pointer_cast(this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>()->data().data());
    _probe.indices_ptr    = this->_searcher->indices();
    _probe.cell_start_ptr = this->_searcher->cell_start();
    _probe.cell_end_ptr   = this->_searcher->cell_end();
    _probe.lower_corner      = this->_searcher->lower_corner();
    _probe.grid_size         = this->_searcher->grid_size();
    _probe.inverse_cell_size = this->_searcher->inverse_cell_size();
    _probe.cell_size         = this->_searcher->cell_size();
    _probe.particle_count    = static_cast<int>(this->_fluid->particle_count());
    _probe.num_of_cells      = this->_universe->number_of_cells();
    _probe.num_of_properties = static_cast<int>(this->_fluid->particle_properties().size());
    _probe.kernel            = _kernel;

    return true;
}

template <typename T>
void
SphGatewaySolver<T>::update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver,
                                                 const int index) {
    const auto probe = _probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;
    auto* cell_group_count_ptr     = atlas::raw_pointer_cast(_cell_group_count.data());
    const int group_particle_count = _group_particle_count;

    // Count particles and required representative groups for each selected cell.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Cells assigned to another solver are excluded from this gateway stage.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell] = T(0);
                cell_group_count_ptr[cell]      = 0;
                return;
            }

            // Use the searcher's sorted range to count cell-local particles.
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            const int count = (begin >= 0 && end > begin) ? (end - begin) : 0;

            probe.number_particle_ptr[cell] = static_cast<T>(count);
            cell_group_count_ptr[cell]      = group_count_for_cell(count, group_particle_count);
        });
}

template <typename T>
void
SphGatewaySolver<T>::build_group_representatives(const DeviceBuffer<int>* allocated_solver,
                                                 const int index) {
    const auto probe = _probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    // Expose group buffers to the device kernel.
    auto* group_position_ptr         = atlas::raw_pointer_cast(_group_position.data());
    auto* group_velocity_ptr         = atlas::raw_pointer_cast(_group_velocity.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    auto* group_mass_ptr             = atlas::raw_pointer_cast(_group_mass.data());
    auto* group_member_count_ptr     = atlas::raw_pointer_cast(_group_member_count.data());
    auto* group_species_ptr          = atlas::raw_pointer_cast(_group_species.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    const int group_particle_count   = _group_particle_count;

    // Build deterministic representatives from sorted particle chunks.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            if (begin < 0 || group_count <= 0) {
                return;
            }

            for (int group_local = 0; group_local < group_count; ++group_local) {
                // Store each representative at the beginning of the cell range.
                const int representative_index = begin + group_local;

                // Convert the local group id into a sorted-particle subrange.
                const int sorted_begin         = begin + group_local * group_particle_count;
                const int candidate_sorted_end = sorted_begin + group_particle_count;
                const int sorted_end           = candidate_sorted_end < probe.cell_end_ptr[cell]
                              ? candidate_sorted_end
                              : probe.cell_end_ptr[cell];

                Vector3<T> position_sum(T(0), T(0), T(0));
                Vector3<T> velocity_sum(T(0), T(0), T(0));
                T mass_sum                         = T(0);
                int member_count                   = 0;
                std::size_t representative_species = 0;

                // Accumulate member particle data for this representative group.
                for (int sorted_index = sorted_begin; sorted_index < sorted_end; ++sorted_index) {
                    const int particle_index = probe.indices_ptr[sorted_index];

                    if (particle_index < 0 || particle_index >= probe.particle_count) {
                        continue;
                    }

                    // Use the first valid member's species as the representative species.
                    if (member_count == 0) {
                        representative_species = probe.species_ptr[particle_index];
                    }

                    position_sum += probe.position_ptr[particle_index];
                    velocity_sum += probe.velocity_ptr[particle_index];
                    ++member_count;

                    // Add mass only when the particle species has valid material data.
                    const std::size_t species_index = probe.species_ptr[particle_index];
                    if (species_index < static_cast<std::size_t>(probe.num_of_properties)) {
                        mass_sum += probe.properties_ptr[species_index].mass;
                    }
                }

                if (member_count <= 0) {
                    continue;
                }

                // Store representative mean state and group metadata.
                const Vector3<T> mean_position = position_sum / static_cast<T>(member_count);
                const Vector3<T> mean_velocity = velocity_sum / static_cast<T>(member_count);

                group_position_ptr[representative_index]     = mean_position;
                group_velocity_ptr[representative_index]     = mean_velocity;
                group_updated_pos_ptr[representative_index]  = mean_position;
                group_updated_vel_ptr[representative_index]  = mean_velocity;
                group_mass_ptr[representative_index]         = mass_sum;
                group_member_count_ptr[representative_index] = member_count;
                group_species_ptr[representative_index]      = representative_species;
            }
        });
}

template <typename T>
void
SphGatewaySolver<T>::estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver,
                                                         const int index) {
    const auto probe = _probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_density_ptr          = atlas::raw_pointer_cast(_group_density.data());
    auto* group_pressure_ptr         = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());

    // Estimate density and pressure for each group representative.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            if (begin < 0 || group_count <= 0) {
                return;
            }

            for (int lhs_group = 0; lhs_group < group_count; ++lhs_group) {
                const int lhs_index             = begin + lhs_group;
                const std::size_t species_index = group_species_ptr[lhs_index];

                // Invalid representative species disables this group.
                if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                    group_density_ptr[lhs_index]  = T(0);
                    group_pressure_ptr[lhs_index] = T(0);
                    continue;
                }

                // Resolve material parameters for the representative group.
                const auto& property                 = probe.properties_ptr[species_index];
                const T smoothing_length             = smoothing_length_for(property, probe.cell_size);
                const T smoothing_length_squared     = smoothing_length * smoothing_length;
                const T rest_density                 = rest_density_for(property);
                const T pressure_coeff               = pressure_coefficient_for(property);
                const Vector3<T> lhs_pos             = group_position_ptr[lhs_index];

                T density = T(0);

                // Accumulate density contributions from representatives in the same cell.
                for (int rhs_group = 0; rhs_group < group_count; ++rhs_group) {
                    const int rhs_index         = begin + rhs_group;
                    const Vector3<T> delta      = lhs_pos - group_position_ptr[rhs_index];
                    const T radius_squared      = delta.length_squared();
                    const T representative_mass = group_mass_ptr[rhs_index];

                    if (radius_squared > smoothing_length_squared) {
                        continue;
                    }

                    const T radius = static_cast<T>(sqrt(radius_squared));

                    density += representative_mass * probe.kernel.density_weight(radius, smoothing_length);
                }

                // Fall back to rest density when the estimate is invalid.
                if (!(density > T(0))) {
                    density = rest_density;
                }

                // Store group density and equation-of-state pressure.
                group_density_ptr[lhs_index]  = density;
                group_pressure_ptr[lhs_index] = pressure_coeff * (density - rest_density);
            }
        });
}

template <typename T>
void
SphGatewaySolver<T>::update_group_motion(const DeviceBuffer<int>* allocated_solver,
                                         const int index,
                                         const T dt) {
    const auto probe = _probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_velocity_ptr   = atlas::raw_pointer_cast(_group_velocity.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_density_ptr    = atlas::raw_pointer_cast(_group_density.data());
    const auto* group_pressure_ptr   = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());

    // Update group representative velocities and write per-cell force output.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            if (begin < 0 || group_count <= 0) {
                probe.field_force_ptr[cell] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            Vector3<T> cell_force(T(0), T(0), T(0));
            int active_group_count = 0;

            for (int lhs_group = 0; lhs_group < group_count; ++lhs_group) {
                const int lhs_index             = begin + lhs_group;
                const std::size_t species_index = group_species_ptr[lhs_index];

                if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                    continue;
                }

                // Resolve representative material and state.
                const auto& property                 = probe.properties_ptr[species_index];
                const T smoothing_length             = smoothing_length_for(property, probe.cell_size);
                const T smoothing_length_squared     = smoothing_length * smoothing_length;
                const T viscosity                    = property.dynamic_viscosity.value_or(T(0));
                const T group_mass                   = group_mass_ptr[lhs_index];
                const Vector3<T> lhs_pos             = group_position_ptr[lhs_index];
                const Vector3<T> lhs_vel             = group_velocity_ptr[lhs_index];

                Vector3<T> acceleration(T(0), T(0), T(0));

                // Accumulate pressure and viscosity terms from other groups in the same cell.
                for (int rhs_group = 0; rhs_group < group_count; ++rhs_group) {
                    if (lhs_group == rhs_group) {
                        continue;
                    }

                    const int rhs_index = begin + rhs_group;
                    const T rhs_density = group_density_ptr[rhs_index];

                    if (!(rhs_density > T(0))) {
                        continue;
                    }

                    const Vector3<T> delta = lhs_pos - group_position_ptr[rhs_index];
                    const T radius_squared = delta.length_squared();

                    if (!(radius_squared > T(0)) || radius_squared > smoothing_length_squared) {
                        continue;
                    }

                    const T radius = static_cast<T>(sqrt(radius_squared));

                    // Add pressure-gradient acceleration.
                    const Vector3<T> grad = probe.kernel.pressure_gradient(delta, radius, smoothing_length);

                    const T pressure_term = (group_pressure_ptr[lhs_index] + group_pressure_ptr[rhs_index])
                        / (static_cast<T>(2) * rhs_density);

                    acceleration -= grad * (group_mass_ptr[rhs_index] * pressure_term);

                    // Add viscosity acceleration when the material viscosity is enabled.
                    if (viscosity > T(0)) {
                        const T laplacian = probe.kernel.viscosity_laplacian(radius, smoothing_length);

                        acceleration += (group_velocity_ptr[rhs_index] - lhs_vel)
                            * (viscosity * group_mass_ptr[rhs_index] * laplacian / rhs_density);
                    }
                }

                if (!(group_mass > T(0))) {
                    continue;
                }

                // Update representative velocity only; position remains unchanged here.
                const Vector3<T> updated_velocity = lhs_vel + acceleration * dt;

                group_updated_vel_ptr[lhs_index] = updated_velocity;
                group_updated_pos_ptr[lhs_index] = lhs_pos;

                // Accumulate force-like output for the cell.
                cell_force += acceleration * group_mass;
                ++active_group_count;
            }

            // Store averaged group force for this cell.
            probe.field_force_ptr[cell] = active_group_count > 0
                ? cell_force / static_cast<T>(active_group_count)
                : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
void
SphGatewaySolver<T>::scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver,
                                                       const int index) {
    const auto probe = _probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    const auto* group_updated_vel_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());
    const auto* cell_group_count_ptr  = atlas::raw_pointer_cast(this->_cell_group_count.data());
    const int group_particle_count    = _group_particle_count;

    // Copy each representative velocity back to its member particles.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            if (begin < 0 || group_count <= 0) {
                return;
            }

            for (int group_local = 0; group_local < group_count; ++group_local) {
                const int representative_index = begin + group_local;

                // Reconstruct the same sorted-particle subrange used during grouping.
                const int sorted_begin         = begin + group_local * group_particle_count;
                const int candidate_sorted_end = sorted_begin + group_particle_count;
                const int sorted_end           = candidate_sorted_end < probe.cell_end_ptr[cell]
                              ? candidate_sorted_end
                              : probe.cell_end_ptr[cell];

                for (int sorted_index = sorted_begin; sorted_index < sorted_end; ++sorted_index) {
                    const int particle_index = probe.indices_ptr[sorted_index];

                    if (particle_index < 0 || particle_index >= probe.particle_count) {
                        continue;
                    }

                    // Apply the group representative velocity to the particle.
                    probe.velocity_ptr[particle_index] = group_updated_vel_ptr[representative_index];
                }
            }
        });
}

template <typename T>
T
SphGatewaySolver<T>::smoothing_length_for(const MaterialProperties<T>& property,
                                          const T cell_size) noexcept {
    // Prefer material smoothing length; fall back to searcher cell size.
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }

    return cell_size;
}

template <typename T>
T
SphGatewaySolver<T>::rest_density_for(const MaterialProperties<T>& property) noexcept {
    // Prefer material rest density; fall back to unit density.
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }

    return T(1);
}

template <typename T>
T
SphGatewaySolver<T>::pressure_coefficient_for(const MaterialProperties<T>& property) noexcept {
    // Missing pressure coefficient disables pressure response.
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphGatewaySolver<T>::search_radius_for(const T smoothing_length,
                                       const T cell_size) noexcept {
    // Convert smoothing length to an integer grid-cell search radius.
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphGatewaySolver<T>::particle_cell(const Vector3<T>& position,
                                   const Vector3<T>& lower_corner,
                                   const T inverse_cell_size,
                                   const Vector3<int>& grid_size) noexcept {
    // Map world position to grid coordinates and clamp to the valid search domain.
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();

    return atlas::math::clamp(
        cell,
        Vector3<int>(0, 0, 0),
        grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
SphGatewaySolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell,
                                            const Vector3<int>& grid_size) noexcept {
    // Check half-open grid bounds: [0, grid_size).
    return atlas::math::all(cell >= Vector3<int>(0, 0, 0))
        && atlas::math::all(cell < grid_size);
}

template <typename T>
int
SphGatewaySolver<T>::group_count_for_cell(const int particle_count,
                                          const int group_particle_count) noexcept {
    // Use ceiling division to compute the number of required groups.
    if (particle_count <= 0 || group_particle_count <= 0) {
        return 0;
    }

    return (particle_count + group_particle_count - 1) / group_particle_count;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store universe dependency.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store fluid dependency.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_searcher(
    SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store searcher dependency.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    // Store selected SPH kernel type.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_group_particle_count(
    const int group_particle_count) noexcept {
    // Store target group size; validation checks positivity later.
    _group_particle_count = group_particle_count;
    return *this;
}

template <typename T>
void
SphGatewaySolver<T>::Builder::validate() const {
    // Builder requires all runtime dependencies.
    if (!_universe) {
        throw std::runtime_error("SphGatewaySolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("SphGatewaySolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("SphGatewaySolver::Builder: searcher must not be null.");
    }

    if (_group_particle_count <= 0) {
        throw std::runtime_error("SphGatewaySolver::Builder: group_particle_count must be positive.");
    }
}

template <typename T>
SphGatewaySolver<T>
SphGatewaySolver<T>::Builder::build() const {
    // Validate configuration and construct a solver value.
    validate();

    return SphGatewaySolver<T>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type,
        _group_particle_count);
}

template <typename T>
atlas::host_shared_ptr<SphGatewaySolver<T>>
SphGatewaySolver<T>::Builder::make_host_shared() const {
    // Validate configuration and construct a host-shared solver.
    validate();

    return atlas::make_host_shared<SphGatewaySolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type,
        _group_particle_count);
}

} // namespace atlas::system
