#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
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
    // Initialize the shared solver infrastructure with the simulation universe,
    // fluid state container, and spatial searcher.
    //
    // The base Solver class owns these common dependencies so derived solvers
    // can focus on their domain-specific update logic.
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))

    // Construct the selected SPH kernel wrapper from the requested kernel type.
    //
    // This kernel object provides the density, gradient, and Laplacian
    // operators required by the group-based SPH approximation.
    , _kernel(kernel_type)

    // Store the target number of particles represented by one gateway group.
    //
    // If the caller provides a non-positive value, fall back to a conservative
    // default of 5 particles per group to keep the grouping scheme valid.
    , _group_particle_count(group_particle_count > 0 ? group_particle_count : 5) {
    // Ensure that all universe-side state buffers required by this solver exist.
    //
    // The gateway SPH solver uses:
    //   - a per-cell particle count state,
    //   - a per-cell field force state.
    ensure_universe_states();
}

template <typename T>
typename SphGatewaySolver<T>::Builder
SphGatewaySolver<T>::builder() noexcept {
    // Return a fresh builder with default-initialized configuration fields.
    //
    // This gives the caller a fluent construction entry point for assembling
    // the solver dependencies step by step.
    return Builder {};
}

template <typename T>
SphKernelType
SphGatewaySolver<T>::kernel_type() const noexcept {
    // Return the currently selected runtime SPH kernel type.
    return _kernel.type;
}

template <typename T>
int
SphGatewaySolver<T>::group_particle_count() const noexcept {
    // Return the configured number of particles represented by each gateway group.
    return _group_particle_count;
}

template <typename T>
void
SphGatewaySolver<T>::solve(const T dt) {
    // Dispatch to the partition-aware overload without explicit cell partitioning.
    //
    // In this mode:
    //   - all cells are eligible for processing,
    //   - the solver index is only a dummy placeholder.
    solve(nullptr, 0, dt);
}

template <typename T>
void
SphGatewaySolver<T>::solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    // Initialize and validate all solver dependencies and required runtime state.
    //
    // This stage ensures that:
    //   - the universe, fluid, and searcher exist,
    //   - required particle states are present,
    //   - required universe states exist,
    //   - the spatial search structure is up to date.
    if (!initialize_context()) {
        return;
    }

    // The time step must be strictly positive for a meaningful motion update.
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphGatewaySolver: dt must be positive.");
    }

    // Allocate and clear all internal group-level working buffers.
    //
    // If that preparation fails, also reset the universe-side field outputs so
    // no stale values remain visible to downstream systems.
    if (!prepare_group_fields()) {
        reset_universe_fields();
        return;
    }

    // Measure how many particles belong to each cell and derive how many
    // representative gateway groups each cell should contain.
    update_cell_particle_counts(allocated_solver, index);

    // Build one representative state per local group by aggregating the member
    // particles assigned to that group inside each cell.
    build_group_representatives(allocated_solver, index);

    // Estimate group-level density and pressure using the selected SPH kernel.
    estimate_group_density_and_pressure(allocated_solver, index);

    // Update the velocity of each representative group using pressure and
    // viscosity interactions inside the cell.
    update_group_motion(allocated_solver, index, dt);

    // Scatter the updated representative velocities back to the actual particles.
    //
    // All particles belonging to a group inherit the same updated velocity.
    scatter_group_states_to_particles(allocated_solver, index);
}

template <typename T>
void
SphGatewaySolver<T>::ensure_universe_states() {
    // Nothing can be initialized if the universe dependency is absent.
    if (!this->_universe) {
        return;
    }

    // Cache the total number of spatial cells to size all per-cell state buffers.
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    // Ensure the universe stores a per-cell particle count state.
    //
    // This state is updated every solve step and records how many particles are
    // currently associated with each cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
    }

    // Ensure the universe stores a per-cell field force state.
    //
    // This state is used to expose the average force-like quantity accumulated
    // from the group dynamics inside each cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseFieldForceState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseFieldForceState<T>>(number_of_cells);
    }
}

template <typename T>
bool
SphGatewaySolver<T>::initialize_context() noexcept {
    // The gateway SPH solver requires all three core dependencies.
    //
    // Without:
    //   - a universe, there is no cell domain,
    //   - a fluid, there are no particle states,
    //   - a searcher, particles cannot be organized by cell.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        // Reset published universe-side fields and clear all internal buffers so
        // the solver leaves behind a consistent empty state.
        reset_universe_fields();
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

    // Retrieve the particle position state.
    //
    // Group representatives are built from particle positions and later scatter
    // updated positions back to the particles.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();

    // Retrieve the particle velocity state.
    //
    // Group representatives also store averaged velocities and write updated
    // group velocities back to the member particles.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Retrieve the particle species state.
    //
    // Species identifiers are required to access material properties such as
    // smoothing length, rest density, pressure coefficient, and viscosity.
    auto* species_state = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Abort when any required particle state is missing.
    //
    // The solver clears all derived buffers and universe outputs to ensure the
    // failure leaves no stale intermediate data.
    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_universe_fields();
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

    // Ensure required universe-side output states exist.
    ensure_universe_states();

    // Rebuild the spatial search structure so the current particle positions are
    // reflected in the cell ranges used by all later stages.
    this->_searcher->build();
    return true;
}

template <typename T>
bool
SphGatewaySolver<T>::prepare_group_fields() {
    // Cache the global particle count and cell count.
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells   = this->_universe->number_of_cells();

    // If there are no particles or no valid cells, clear all internal group
    // buffers and published universe fields, then report that no work can be done.
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

    // Allocate one per-cell buffer that stores how many representative groups
    // are created inside each spatial cell.
    _cell_group_count.resize(static_cast<std::size_t>(num_of_cells));

    // Allocate group-level buffers large enough to store one representative
    // entry per particle index.
    //
    // This implementation uses representative_index = begin + group_local,
    // so indexing remains within the particle-index space.
    _group_position.resize(static_cast<std::size_t>(particle_count));
    _group_velocity.resize(static_cast<std::size_t>(particle_count));
    _group_updated_position.resize(static_cast<std::size_t>(particle_count));
    _group_updated_velocity.resize(static_cast<std::size_t>(particle_count));
    _group_mass.resize(static_cast<std::size_t>(particle_count));
    _group_density.resize(static_cast<std::size_t>(particle_count));
    _group_pressure.resize(static_cast<std::size_t>(particle_count));
    _group_member_count.resize(static_cast<std::size_t>(particle_count));
    _group_species.resize(static_cast<std::size_t>(particle_count));

    // Clear all group-related working arrays before filling them with fresh data.
    atlas::parallel_fill<ExecutionPolicy::device>(_cell_group_count.begin(), _cell_group_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_group_position.begin(), _group_position.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_velocity.begin(), _group_velocity.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_updated_position.begin(), _group_updated_position.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_updated_velocity.begin(), _group_updated_velocity.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_mass.begin(), _group_mass.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_density.begin(), _group_density.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_pressure.begin(), _group_pressure.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_member_count.begin(), _group_member_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_group_species.begin(), _group_species.end(), std::size_t(0));

    // Also clear the universe-side output fields so they are rebuilt entirely
    // from the current solve step.
    reset_universe_fields();
    return true;
}

template <typename T>
void
SphGatewaySolver<T>::reset_universe_fields() {
    // Nothing can be reset if the universe does not exist.
    if (!this->_universe) {
        return;
    }

    // Ensure the required universe-side states exist before clearing them.
    ensure_universe_states();

    // Retrieve the per-cell particle count state.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the per-cell field force state.
    auto* field_force_state
        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();

    // Clear the per-cell particle counts.
    if (number_particle_state != nullptr) {
        auto& number_particle = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(number_particle.begin(), number_particle.end(), T(0));
    }

    // Clear the per-cell field force vectors.
    if (field_force_state != nullptr) {
        auto& field_force = field_force_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(field_force.begin(), field_force.end(), Vector3<T>(T(0), T(0), T(0)));
    }
}

template <typename T>
void
SphGatewaySolver<T>::update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, const int index) {
    // Retrieve the universe-side particle-count state that will be updated.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    auto& number_particle = number_particle_state->data();

    // Convert the required buffers to raw pointers for device-side access.
    auto* number_particle_ptr      = atlas::raw_pointer_cast(number_particle.data());
    auto* cell_group_count_ptr     = atlas::raw_pointer_cast(_cell_group_count.data());
    const auto* cell_start_ptr     = this->_searcher->cell_start();
    const auto* cell_end_ptr       = this->_searcher->cell_end();
    const int num_of_cells         = this->_universe->number_of_cells();
    const int group_particle_count = _group_particle_count;

    // Optional mapping that assigns cells to solver partitions.
    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    // Process one spatial cell per device work item.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver partition.
            //
            // In that case explicitly zero their local outputs for this solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                number_particle_ptr[cell]  = T(0);
                cell_group_count_ptr[cell] = 0;
                return;
            }

            // Read the particle range belonging to the current cell.
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // Derive the particle count from the valid range length.
            const int count = (begin >= 0 && end > begin) ? (end - begin) : 0;

            // Publish the per-cell particle count.
            number_particle_ptr[cell] = static_cast<T>(count);

            // Convert the particle count into the number of representative groups
            // needed for this cell based on the configured grouping size.
            cell_group_count_ptr[cell] = group_count_for_cell(count, group_particle_count);
        });
}

template <typename T>
void
SphGatewaySolver<T>::build_group_representatives(const DeviceBuffer<int>* allocated_solver, const int index) {
    // Retrieve the particle states used to construct group representatives.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Cache references to particle buffers and material properties.
    auto& positions           = position_state->data();
    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& particle_properties = this->_fluid->particle_properties();

    // Convert all needed containers to raw pointers for device execution.
    const auto* position_ptr         = atlas::raw_pointer_cast(positions.data());
    const auto* velocity_ptr         = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr          = atlas::raw_pointer_cast(particle_species.data());
    const auto* properties_ptr       = atlas::raw_pointer_cast(particle_properties.data());
    auto* group_position_ptr         = atlas::raw_pointer_cast(_group_position.data());
    auto* group_velocity_ptr         = atlas::raw_pointer_cast(_group_velocity.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    auto* group_mass_ptr             = atlas::raw_pointer_cast(_group_mass.data());
    auto* group_member_count_ptr     = atlas::raw_pointer_cast(_group_member_count.data());
    auto* group_species_ptr          = atlas::raw_pointer_cast(_group_species.data());
    const auto* indices_ptr          = this->_searcher->indices();
    const auto* cell_start_ptr       = this->_searcher->cell_start();
    const auto* cell_end_ptr         = this->_searcher->cell_end();
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    const int num_of_cells           = this->_universe->number_of_cells();
    const int particle_count         = static_cast<int>(this->_fluid->particle_count());
    const int num_of_properties      = static_cast<int>(particle_properties.size());
    const int group_particle_count   = _group_particle_count;
    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    // Build representative groups independently for each cell.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells outside the current solver partition.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            // Ignore cells with no valid particle range or no representative groups.
            if (begin < 0 || group_count <= 0) {
                return;
            }

            // Build each representative by aggregating a contiguous chunk of
            // group_particle_count particles inside the cell.
            for (int group_local = 0; group_local < group_count; ++group_local) {
                // Use the beginning of the group block in the cell-local sorted
                // range as the representative storage index.
                const int representative_index = begin + group_local;

                // Compute the sorted range of particles belonging to this group.
                const int sorted_begin         = begin + group_local * group_particle_count;
                const int candidate_sorted_end = sorted_begin + group_particle_count;
                const int sorted_end           = candidate_sorted_end < cell_end_ptr[cell]
                              ? candidate_sorted_end
                              : cell_end_ptr[cell];

                // Accumulate position, velocity, and mass over all member particles.
                Vector3<T> position_sum(T(0), T(0), T(0));
                Vector3<T> velocity_sum(T(0), T(0), T(0));
                T mass_sum       = T(0);
                int member_count = 0;

                // Store the species of the first valid member as the group species.
                //
                // This is a simplification that assumes a representative species
                // can stand in for the whole group during later SPH evaluation.
                std::size_t representative_species = 0;

                for (int sorted_index = sorted_begin; sorted_index < sorted_end; ++sorted_index) {
                    const int particle_index = indices_ptr[sorted_index];

                    // Skip invalid particle references defensively.
                    if (particle_index < 0 || particle_index >= particle_count) {
                        continue;
                    }

                    // Use the first valid member to define the representative species.
                    if (member_count == 0) {
                        representative_species = species_ptr[particle_index];
                    }

                    // Accumulate position and velocity for mean representative values.
                    position_sum += position_ptr[particle_index];
                    velocity_sum += velocity_ptr[particle_index];
                    ++member_count;

                    // Accumulate material mass when the species index is valid.
                    const std::size_t species_index = species_ptr[particle_index];
                    if (species_index < static_cast<std::size_t>(num_of_properties)) {
                        mass_sum += properties_ptr[species_index].mass;
                    }
                }

                // Ignore empty groups, which can happen if all candidate indices were invalid.
                if (member_count <= 0) {
                    continue;
                }

                // Compute mean representative position and velocity.
                const Vector3<T> mean_position = position_sum / static_cast<T>(member_count);
                const Vector3<T> mean_velocity = velocity_sum / static_cast<T>(member_count);

                // Store the initial representative state.
                group_position_ptr[representative_index] = mean_position;
                group_velocity_ptr[representative_index] = mean_velocity;

                // Initialize the updated state buffers with the same values before
                // the motion update stage modifies them.
                group_updated_pos_ptr[representative_index] = mean_position;
                group_updated_vel_ptr[representative_index] = mean_velocity;

                // Store group-level aggregate metadata.
                group_mass_ptr[representative_index]         = mass_sum;
                group_member_count_ptr[representative_index] = member_count;
                group_species_ptr[representative_index]      = representative_species;
            }
        });
}

template <typename T>
void
SphGatewaySolver<T>::estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, const int index) {
    // Cache searcher spatial metadata.
    //
    // These values are often useful for spatially aware logic. In this current
    // implementation, cell_size is used directly while the others remain part of
    // the available local context.
    const auto lower_corner = this->_searcher->lower_corner();
    const auto grid_size    = this->_searcher->grid_size();
    const T inv_cell_size   = this->_searcher->inverse_cell_size();
    const T cell_size       = this->_searcher->cell_size();

    auto& particle_properties = this->_fluid->particle_properties();

    // Convert buffers to raw pointers for device execution.
    const auto* properties_ptr       = atlas::raw_pointer_cast(particle_properties.data());
    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_density_ptr          = atlas::raw_pointer_cast(_group_density.data());
    auto* group_pressure_ptr         = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* cell_start_ptr       = this->_searcher->cell_start();
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    const int num_of_cells           = this->_universe->number_of_cells();
    const int num_of_properties      = static_cast<int>(particle_properties.size());

    // Capture the selected SPH kernel implementation by value.
    const auto kernel = _kernel;

    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells outside the active solver partition.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            // Nothing to do when the cell contains no valid group representatives.
            if (begin < 0 || group_count <= 0) {
                return;
            }

            // Estimate density and pressure for each representative group inside the cell.
            for (int lhs_group = 0; lhs_group < group_count; ++lhs_group) {
                const int lhs_index             = begin + lhs_group;
                const std::size_t species_index = group_species_ptr[lhs_index];

                // Reject invalid representative species entries.
                if (species_index >= static_cast<std::size_t>(num_of_properties)) {
                    group_density_ptr[lhs_index]  = T(0);
                    group_pressure_ptr[lhs_index] = T(0);
                    continue;
                }

                // Read material properties that control the SPH evaluation.
                const auto& property = properties_ptr[species_index];

                // Derive the effective smoothing length, rest density, and pressure coefficient.
                const T smoothing_length = smoothing_length_for(property, cell_size);
                const T rest_density     = rest_density_for(property);
                const T pressure_coeff   = pressure_coefficient_for(property);
                const Vector3<T> lhs_pos = group_position_ptr[lhs_index];

                T density = T(0);

                // Accumulate density from all representative groups in the same cell.
                //
                // This is a simplified local SPH density estimate that uses only
                // intra-cell representative interactions.
                for (int rhs_group = 0; rhs_group < group_count; ++rhs_group) {
                    const int rhs_index         = begin + rhs_group;
                    const Vector3<T> delta      = lhs_pos - group_position_ptr[rhs_index];
                    const T radius              = delta.length();
                    const T representative_mass = group_mass_ptr[rhs_index];

                    density += representative_mass * kernel.density_weight(radius, smoothing_length);
                }

                // Fall back to rest density when the estimated density is not positive.
                //
                // This keeps the later pressure evaluation numerically stable and
                // avoids propagating degenerate density values.
                if (!(density > T(0))) {
                    density = rest_density;
                }

                // Store the estimated density and the corresponding equation-of-state pressure.
                group_density_ptr[lhs_index]  = density;
                group_pressure_ptr[lhs_index] = pressure_coeff * (density - rest_density);
            }
        });
}

template <typename T>
void
SphGatewaySolver<T>::update_group_motion(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    // Retrieve the universe-side per-cell field force state that will be updated.
    auto* field_force_state
        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();

    auto& particle_properties = this->_fluid->particle_properties();
    auto& field_force         = field_force_state->data();

    // Convert group and property buffers to raw pointers for device execution.
    const T cell_size                = this->_searcher->cell_size();
    const auto* properties_ptr       = atlas::raw_pointer_cast(particle_properties.data());
    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_velocity_ptr   = atlas::raw_pointer_cast(_group_velocity.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_density_ptr    = atlas::raw_pointer_cast(_group_density.data());
    const auto* group_pressure_ptr   = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    auto* field_force_ptr            = atlas::raw_pointer_cast(field_force.data());
    const auto* cell_start_ptr       = this->_searcher->cell_start();
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    const int num_of_cells           = this->_universe->number_of_cells();
    const int num_of_properties      = static_cast<int>(particle_properties.size());
    const auto kernel                = _kernel;
    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells outside the active solver partition.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            // When the cell has no valid groups, publish a zero field force and stop.
            if (begin < 0 || group_count <= 0) {
                field_force_ptr[cell] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            // Accumulate a cell-level mean force-like quantity from all updated groups.
            Vector3<T> cell_force(T(0), T(0), T(0));
            int active_group_count = 0;

            for (int lhs_group = 0; lhs_group < group_count; ++lhs_group) {
                const int lhs_index             = begin + lhs_group;
                const std::size_t species_index = group_species_ptr[lhs_index];

                // Skip representatives with invalid material-property lookup.
                if (species_index >= static_cast<std::size_t>(num_of_properties)) {
                    continue;
                }

                const auto& property = properties_ptr[species_index];

                // Derive the kernel support size and viscosity from the representative material.
                const T smoothing_length = smoothing_length_for(property, cell_size);
                const T viscosity        = property.dynamic_viscosity.value_or(T(0));
                const T group_mass       = group_mass_ptr[lhs_index];
                const Vector3<T> lhs_pos = group_position_ptr[lhs_index];
                const Vector3<T> lhs_vel = group_velocity_ptr[lhs_index];

                // Accumulate pressure and viscosity contributions as acceleration.
                Vector3<T> acceleration(T(0), T(0), T(0));

                for (int rhs_group = 0; rhs_group < group_count; ++rhs_group) {
                    // Exclude self-interaction.
                    if (lhs_group == rhs_group) {
                        continue;
                    }

                    const int rhs_index = begin + rhs_group;
                    const T rhs_density = group_density_ptr[rhs_index];

                    // A valid SPH interaction requires strictly positive neighbor density.
                    if (!(rhs_density > T(0))) {
                        continue;
                    }

                    const Vector3<T> delta = lhs_pos - group_position_ptr[rhs_index];
                    const T radius         = delta.length();

                    // Only consider neighbors inside the smoothing support.
                    if (!(radius > T(0)) || radius > smoothing_length) {
                        continue;
                    }

                    // Compute the pressure-gradient kernel term for this pair.
                    const Vector3<T> grad = kernel.pressure_gradient(delta, radius, smoothing_length);

                    // Build a symmetric pressure term based on the pressures of both representatives.
                    const T pressure_term = (group_pressure_ptr[lhs_index] + group_pressure_ptr[rhs_index])
                        / (static_cast<T>(2) * rhs_density);

                    // Add the pressure-force contribution.
                    acceleration -= grad * (group_mass_ptr[rhs_index] * pressure_term);

                    // Add a viscosity contribution when a positive viscosity coefficient exists.
                    if (viscosity > T(0)) {
                        const T laplacian = kernel.viscosity_laplacian(radius, smoothing_length);
                        acceleration += (group_velocity_ptr[rhs_index] - lhs_vel)
                            * (viscosity * group_mass_ptr[rhs_index] * laplacian / rhs_density);
                    }
                }

                // Ignore degenerate representatives with non-positive total mass.
                if (!(group_mass > T(0))) {
                    continue;
                }

                // Integrate one explicit Euler step for velocity only.
                const Vector3<T> updated_velocity = lhs_vel + acceleration * dt;

                // Keep the representative position unchanged and store the updated velocity.
                group_updated_vel_ptr[lhs_index] = updated_velocity;
                group_updated_pos_ptr[lhs_index] = lhs_pos;

                // Accumulate this representative's contribution to the cell-level force summary.
                cell_force += acceleration * group_mass;
                ++active_group_count;
            }

            // Publish the average force-like vector over active representatives in the cell.
            field_force_ptr[cell] = active_group_count > 0
                ? cell_force / static_cast<T>(active_group_count)
                : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
void
SphGatewaySolver<T>::scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, const int index) {
    // Retrieve the particle velocity state that will receive the updated
    // representative velocities.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    auto& velocities = velocity_state->data();

    // Convert all required buffers to raw pointers for device execution.
    auto* velocity_ptr                = atlas::raw_pointer_cast(velocities.data());
    const auto* group_updated_vel_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());
    const auto* indices_ptr           = this->_searcher->indices();
    const auto* cell_start_ptr        = this->_searcher->cell_start();
    const auto* cell_end_ptr          = this->_searcher->cell_end();
    const auto* cell_group_count_ptr  = atlas::raw_pointer_cast(this->_cell_group_count.data());
    const int num_of_cells            = this->_universe->number_of_cells();
    const int particle_count          = static_cast<int>(this->_fluid->particle_count());
    const int group_particle_count    = _group_particle_count;
    const auto* allocated_solver_ptr  = allocated_solver != nullptr
         ? atlas::raw_pointer_cast(allocated_solver->data())
         : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to a different solver partition.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin       = cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];

            // Ignore cells without valid representative groups.
            if (begin < 0 || group_count <= 0) {
                return;
            }

            // For each group, write the updated representative velocity back to
            // all member particles in that group's chunk.
            for (int group_local = 0; group_local < group_count; ++group_local) {
                const int representative_index = begin + group_local;
                const int sorted_begin         = begin + group_local * group_particle_count;
                const int candidate_sorted_end = sorted_begin + group_particle_count;
                const int sorted_end           = candidate_sorted_end < cell_end_ptr[cell]
                              ? candidate_sorted_end
                              : cell_end_ptr[cell];

                for (int sorted_index = sorted_begin; sorted_index < sorted_end; ++sorted_index) {
                    const int particle_index = indices_ptr[sorted_index];

                    // Skip invalid particle references defensively.
                    if (particle_index < 0 || particle_index >= particle_count) {
                        continue;
                    }

                    // All particles in the group inherit the same updated representative velocity.
                    velocity_ptr[particle_index] = group_updated_vel_ptr[representative_index];
                }
            }
        });
}

template <typename T>
T
SphGatewaySolver<T>::smoothing_length_for(const MatrialProperties<T>& property, const T cell_size) noexcept {
    // Prefer an explicitly configured positive smoothing length from the material.
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }

    // Fall back to the search-grid cell size when the material does not define one.
    //
    // This keeps the kernel support scale aligned with the spatial partitioning.
    return cell_size;
}

template <typename T>
T
SphGatewaySolver<T>::rest_density_for(const MatrialProperties<T>& property) noexcept {
    // Prefer an explicitly configured positive rest density from the material.
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }

    // Fall back to a neutral default rest density.
    return T(1);
}

template <typename T>
T
SphGatewaySolver<T>::pressure_coefficient_for(const MatrialProperties<T>& property) noexcept {
    // Return the equation-of-state pressure coefficient, defaulting to zero when absent.
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphGatewaySolver<T>::search_radius_for(const T smoothing_length, const T cell_size) noexcept {
    // Convert the smoothing support radius into a search-grid radius measured in cells.
    //
    // The ceiling ensures the support interval is fully covered by the cell search.
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphGatewaySolver<T>::particle_cell(const Vector3<T>& position,
                                   const Vector3<T>& lower_corner,
                                   const T inverse_cell_size,
                                   const Vector3<int>& grid_size) noexcept {
    // Convert the particle position into an integer grid coordinate by:
    //   1. shifting into the grid frame,
    //   2. scaling by the inverse cell size,
    //   3. taking the floor of each component.
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();

    // Clamp the result into the valid grid bounds to avoid out-of-range cell coordinates.
    return atlas::math::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
SphGatewaySolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept {
    // Return whether the candidate cell coordinate lies inside the valid 3D grid bounds.
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
int
SphGatewaySolver<T>::group_count_for_cell(const int particle_count, const int group_particle_count) noexcept {
    // No groups can be formed when either quantity is non-positive.
    if (particle_count <= 0 || group_particle_count <= 0) {
        return 0;
    }

    // Compute the ceiling of particle_count / group_particle_count using integer arithmetic.
    //
    // This yields the number of representative groups needed so every particle
    // in the cell belongs to some group.
    return (particle_count + group_particle_count - 1) / group_particle_count;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store the universe dependency in the builder.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid dependency in the builder.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial searcher dependency in the builder.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    // Store the selected SPH kernel type in the builder.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_group_particle_count(const int group_particle_count) noexcept {
    // Store the desired representative group size in the builder.
    _group_particle_count = group_particle_count;
    return *this;
}

template <typename T>
void
SphGatewaySolver<T>::Builder::validate() const {
    // The solver requires a valid universe dependency.
    if (!_universe) {
        throw std::runtime_error("SphGatewaySolver::Builder: universe must not be null.");
    }

    // The solver requires a valid fluid dependency.
    if (!_fluid) {
        throw std::runtime_error("SphGatewaySolver::Builder: fluid must not be null.");
    }

    // The solver requires a valid spatial searcher dependency.
    if (!_searcher) {
        throw std::runtime_error("SphGatewaySolver::Builder: searcher must not be null.");
    }

    // The group size must be strictly positive so the grouping scheme is valid.
    if (_group_particle_count <= 0) {
        throw std::runtime_error("SphGatewaySolver::Builder: group_particle_count must be positive.");
    }
}

template <typename T>
SphGatewaySolver<T>
SphGatewaySolver<T>::Builder::build() const {
    // Validate the builder configuration before constructing the solver value.
    validate();
    return SphGatewaySolver<T>(_universe, _fluid, _searcher, _kernel_type, _group_particle_count);
}

template <typename T>
atlas::host_shared_ptr<SphGatewaySolver<T>>
SphGatewaySolver<T>::Builder::make_host_shared() const {
    // Validate the builder configuration before constructing a shared host object.
    validate();
    return atlas::make_host_shared<SphGatewaySolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type,
        _group_particle_count);
}

} // namespace atlas::system
