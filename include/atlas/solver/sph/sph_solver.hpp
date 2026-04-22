#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
SphSolver<T>::SphSolver(UniverseHostPtr<T> universe,
                        FluidHostPtr<T> fluid,
                        SpatialHashingSearcherHostPtr<T> searcher,
                        const SphKernelType kernel_type) noexcept
    // Initialize the common solver infrastructure with the universe, fluid,
    // and spatial searcher dependencies.
    //
    // The base Solver class owns these shared references so this SPH solver
    // can focus on particle-based hydrodynamic updates.
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher)) {
    // Construct and store the selected SPH kernel implementation.
    //
    // The kernel wrapper provides:
    //   - density weighting,
    //   - pressure gradient evaluation,
    //   - viscosity Laplacian evaluation,
    // based on the runtime-selected kernel type.
    _kernel = SphKernel<T>(kernel_type);

    // Ensure the universe already contains the per-cell states required by
    // this solver before any solve step is executed.
    ensure_universe_states();
}

template <typename T>
typename SphSolver<T>::Builder
SphSolver<T>::builder() noexcept {
    // Return a new builder with default-initialized configuration.
    //
    // This supports fluent construction of the solver and mirrors the pattern
    // used by other solver components in the codebase.
    return Builder {};
}

template <typename T>
SphKernelType
SphSolver<T>::kernel_type() const noexcept {
    // Return the currently active runtime kernel type.
    return _kernel.type;
}

template <typename T>
void
SphSolver<T>::solve(const T dt) {
    // Initialize and validate the SPH runtime context.
    //
    // This step ensures that:
    //   - the universe, fluid, and searcher exist,
    //   - required particle states are present,
    //   - required universe states exist,
    //   - the spatial search structure is rebuilt for the current particle layout.
    if (!initialize_sph_context()) {
        return;
    }

    // The time step must be strictly positive because the explicit velocity
    // update later in the solve step scales acceleration by dt.
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphSolver: dt must be positive.");
    }

    // Allocate and clear all particle-level temporary fields used by the SPH step.
    //
    // If allocation or preparation fails, also clear universe-side outputs to
    // avoid leaving stale data from a previous solve step.
    if (!prepare_particle_fields()) {
        reset_universe_fields();
        return;
    }

    // Compute the density and pressure fields and update auxiliary universe data.
    update();

    // Compute particle accelerations from pressure and viscosity interactions,
    // then apply the explicit velocity update and publish the cell field force.
    accumulate_acceleration(dt);
}

template <typename T>
void
SphSolver<T>::solve(const DeviceBuffer<int>*, const int, const T) {
    // Reserved for future codec-aware SPH execution.
    //
    // This overload exists for interface compatibility with solvers that support
    // cell partitioning or other distributed execution strategies.
}

template <typename T>
void
SphSolver<T>::ensure_universe_states() {
    // Nothing can be created when the universe dependency is absent.
    if (!this->_universe) {
        return;
    }

    // Cache the number of cells so all per-cell state buffers can be sized consistently.
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    // Ensure the universe stores a per-cell particle count state.
    //
    // This state is updated each solve step and records how many valid particles
    // belong to each search cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
    }

    // Ensure the universe stores a per-cell field force state.
    //
    // This state is later populated with an averaged force-like quantity derived
    // from particle accelerations inside each cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseFieldForceState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseFieldForceState<T>>(number_of_cells);
    }
}

template <typename T>
bool
SphSolver<T>::initialize_sph_context() noexcept {
    // The SPH solver requires all three core dependencies.
    //
    // Without:
    //   - a universe, there is no spatial cell domain,
    //   - a fluid, there are no particle states or material properties,
    //   - a searcher, neighbor lookup by cell is impossible.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        // Clear published universe-side data and internal temporary fields so
        // the solver exits in a consistent empty state.
        reset_universe_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    // Retrieve the particle position state required for neighbor search and
    // distance-based SPH kernel evaluation.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();

    // Retrieve the particle velocity state required for viscosity interaction
    // and for applying the final explicit velocity update.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Retrieve the particle species state required for material-property lookup.
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Abort if any mandatory particle state is unavailable.
    //
    // The solver clears all temporary fields and universe outputs to avoid
    // propagating partial or stale results.
    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_universe_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    // Ensure required universe-side states exist before the solve step continues.
    ensure_universe_states();

    // Rebuild the spatial search structure so cell ranges match the current
    // particle positions before neighbor traversal begins.
    this->_searcher->build();
    return true;
}

template <typename T>
bool
SphSolver<T>::prepare_particle_fields() {
    // Cache the number of particles in the fluid.
    const int particle_count = static_cast<int>(this->_fluid->particle_count());

    // If there are no particles, there is no meaningful SPH work to perform.
    if (particle_count <= 0) {
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        reset_universe_fields();
        return false;
    }

    // Allocate one entry per particle for:
    //   - density,
    //   - pressure,
    //   - accumulated acceleration.
    _density.resize(static_cast<std::size_t>(particle_count));
    _pressure.resize(static_cast<std::size_t>(particle_count));
    _acceleration.resize(static_cast<std::size_t>(particle_count));

    // Clear all particle-level working buffers before they are rebuilt from
    // scratch during the current solve step.
    atlas::parallel_fill<ExecutionPolicy::device>(_density.begin(), _density.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_pressure.begin(), _pressure.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_acceleration.begin(), _acceleration.end(), Vector3<T>(T(0), T(0), T(0)));

    // Also clear universe-side outputs so they reflect only the current step.
    reset_universe_fields();
    return true;
}

template <typename T>
void
SphSolver<T>::reset_universe_fields() {
    // Nothing can be reset when the universe is absent.
    if (!this->_universe) {
        return;
    }

    // Ensure the required universe-side states exist before clearing them.
    ensure_universe_states();

    // Retrieve the per-cell particle-count state.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the per-cell field-force state.
    auto* field_force_state
        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();

    // Clear the per-cell particle counts.
    if (number_particle_state != nullptr) {
        auto& number_particle = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(number_particle.begin(), number_particle.end(), T(0));
    }

    // Clear the per-cell force vectors.
    if (field_force_state != nullptr) {
        auto& field_force = field_force_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(field_force.begin(), field_force.end(), Vector3<T>(T(0), T(0), T(0)));
    }
}

template <typename T>
void
SphSolver<T>::update() {
    // First estimate particle density and pressure using neighbor SPH sampling.
    estimate_particle_density_and_pressure();

    // Then update the universe-side particle counts per cell.
    update_cell_number_particles();
}

template <typename T>
void
SphSolver<T>::estimate_particle_density_and_pressure() {
    // Retrieve the particle states required by density and pressure estimation.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    auto& positions           = position_state->data();
    auto& particle_species    = species_state->data();
    auto& particle_properties = this->_fluid->particle_properties();

    // Cache grid-space metadata from the spatial searcher.
    //
    // These values are used to:
    //   - determine the base cell of a particle,
    //   - determine the cell neighborhood that covers the smoothing support,
    //   - convert 3D neighbor-cell coordinates into linear cell indices.
    const auto lower_corner = this->_searcher->lower_corner();
    const auto grid_size    = this->_searcher->grid_size();
    const T inv_cell_size   = this->_searcher->inverse_cell_size();
    const T cell_size       = this->_searcher->cell_size();

    // Convert all buffers to raw pointers for device-side traversal.
    const auto* position_ptr        = atlas::raw_pointer_cast(positions.data());
    const auto* species_ptr         = atlas::raw_pointer_cast(particle_species.data());
    const auto* properties_ptr      = atlas::raw_pointer_cast(particle_properties.data());
    auto* density_ptr               = atlas::raw_pointer_cast(_density.data());
    auto* pressure_ptr              = atlas::raw_pointer_cast(_pressure.data());
    const auto* indices_ptr         = this->_searcher->indices();
    const auto* cell_start_ptr      = this->_searcher->cell_start();
    const auto* cell_end_ptr        = this->_searcher->cell_end();
    const int particle_count        = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells          = this->_universe->number_of_cells();
    const int num_of_properties     = static_cast<int>(particle_properties.size());
    const auto kernel               = _kernel;

    // Launch one device work item per particle.
    //
    // Each particle independently estimates its density by sampling neighboring
    // particles inside the smoothing support.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            // Read the species index of the target particle.
            const std::size_t species_index = species_ptr[particle_index];

            // If the species index is invalid, publish zero density and pressure
            // and skip all further work for this particle.
            if (species_index >= static_cast<std::size_t>(num_of_properties)) {
                density_ptr[particle_index]  = T(0);
                pressure_ptr[particle_index] = T(0);
                return;
            }

            // Read the target particle's material properties.
            const auto& property = properties_ptr[species_index];

            // Determine the effective smoothing length used by this particle.
            const T h            = smoothing_length_for(property, cell_size);

            // Determine the reference density used by the pressure model.
            const T rest_density = rest_density_for(property);

            // Determine the pressure coefficient used by the simple equation of state.
            const T k            = pressure_coefficient_for(property);

            const Vector3<T> position = position_ptr[particle_index];

            // Determine which cell the target particle belongs to.
            const Vector3<int> base_cell = particle_cell(position, lower_corner, inv_cell_size, grid_size);

            // Convert the smoothing radius into a search radius in grid cells.
            const int neighbor_search_radius = search_radius_for(h, cell_size);

            // Accumulate the SPH density estimate here.
            T density = T(0);

            // Visit all neighboring search cells required to fully cover the
            // particle's smoothing support.
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);

                        // Ignore cells outside the grid bounds.
                        if (!is_valid_neighbor_cell(neighbor_cell, grid_size)) {
                            continue;
                        }

                        // Convert the 3D cell coordinate into the searcher's linear cell key.
                        const int cell = static_cast<int>(SpatialHashingSearcher<T>::linear_key(
                            neighbor_cell.x,
                            neighbor_cell.y,
                            neighbor_cell.z,
                            grid_size));

                        // Ignore invalid linear cell keys defensively.
                        if (cell < 0 || cell >= num_of_cells) {
                            continue;
                        }

                        // Read the contiguous sorted particle range stored for this cell.
                        const int begin = cell_start_ptr[cell];
                        const int end   = cell_end_ptr[cell];

                        // Skip empty or invalid cell spans.
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        // Traverse all particles in the neighbor cell.
                        for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                            const int neighbor_index = indices_ptr[sorted_index];

                            // Skip invalid particle references.
                            if (neighbor_index < 0 || neighbor_index >= particle_count) {
                                continue;
                            }

                            const Vector3<T> delta = position - position_ptr[neighbor_index];
                            const T radius         = delta.length();

                            // Only include neighbors inside the smoothing support.
                            if (radius > h) {
                                continue;
                            }

                            // Read the neighbor species index so its particle mass can be retrieved.
                            const std::size_t neighbor_species_index = species_ptr[neighbor_index];
                            if (neighbor_species_index >= static_cast<std::size_t>(num_of_properties)) {
                                continue;
                            }

                            const T neighbor_mass = properties_ptr[neighbor_species_index].mass;

                            // Standard SPH density accumulation:
                            //
                            //     rho_i += m_j * W(r_ij, h)
                            density += neighbor_mass * kernel.density_weight(radius, h);
                        }
                    }
                }
            }

            // Fall back to the rest density when the estimated density is not positive.
            //
            // This avoids degenerate pressures and keeps the subsequent force
            // computation numerically well-defined.
            if (!(density > T(0))) {
                density = rest_density;
            }

            // Store the final density estimate and the pressure computed from a
            // simple linear equation of state:
            //
            //     p = k * (rho - rho0)
            density_ptr[particle_index]  = density;
            pressure_ptr[particle_index] = k * (density - rest_density);
        });
}

template <typename T>
void
SphSolver<T>::update_cell_number_particles() {
    // Retrieve the universe-side per-cell particle-count state.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    auto& number_particle = number_particle_state->data();

    // Convert all required buffers to raw pointers for device execution.
    auto* number_particle_ptr       = atlas::raw_pointer_cast(number_particle.data());
    const auto* indices_ptr         = this->_searcher->indices();
    const auto* cell_start_ptr      = this->_searcher->cell_start();
    const auto* cell_end_ptr        = this->_searcher->cell_end();
    const int particle_count        = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells          = this->_universe->number_of_cells();

    // Process one cell per device work item.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // Publish zero when the cell contains no valid span.
            if (begin < 0 || end <= begin) {
                number_particle_ptr[cell] = T(0);
                return;
            }

            int count = 0;

            // Count the number of valid particle references in the cell range.
            for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                const int particle_index = indices_ptr[sorted_index];
                if (particle_index >= 0 && particle_index < particle_count) {
                    ++count;
                }
            }

            // Store the final per-cell particle count.
            number_particle_ptr[cell] = static_cast<T>(count);
        });
}

template <typename T>
void
SphSolver<T>::accumulate_acceleration(const T dt) {
    // Retrieve all states required for force accumulation and velocity update.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* field_force_state
        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();

    auto& positions           = position_state->data();
    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& particle_properties = this->_fluid->particle_properties();
    auto& field_force         = field_force_state->data();

    // Cache grid-space metadata needed for neighborhood traversal.
    const auto lower_corner = this->_searcher->lower_corner();
    const auto grid_size    = this->_searcher->grid_size();
    const T inv_cell_size   = this->_searcher->inverse_cell_size();
    const T cell_size       = this->_searcher->cell_size();

    // Convert all required buffers to raw pointers for device-side execution.
    const auto* position_ptr        = atlas::raw_pointer_cast(positions.data());
    auto* velocity_ptr              = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr         = atlas::raw_pointer_cast(particle_species.data());
    const auto* properties_ptr      = atlas::raw_pointer_cast(particle_properties.data());
    const auto* density_ptr         = atlas::raw_pointer_cast(_density.data());
    const auto* pressure_ptr        = atlas::raw_pointer_cast(_pressure.data());
    auto* acceleration_ptr          = atlas::raw_pointer_cast(_acceleration.data());
    auto* field_force_ptr           = atlas::raw_pointer_cast(field_force.data());
    const auto* indices_ptr         = this->_searcher->indices();
    const auto* cell_start_ptr      = this->_searcher->cell_start();
    const auto* cell_end_ptr        = this->_searcher->cell_end();
    const int particle_count        = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells          = this->_universe->number_of_cells();
    const int num_of_properties     = static_cast<int>(particle_properties.size());
    const auto kernel               = _kernel;

    // First pass: compute per-particle acceleration and immediately apply the
    // explicit Euler velocity update.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            const std::size_t species_index = species_ptr[particle_index];

            // Invalid species indices produce zero acceleration.
            if (species_index >= static_cast<std::size_t>(num_of_properties)) {
                acceleration_ptr[particle_index] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            const auto& property = properties_ptr[species_index];

            // Read the SPH support length, dynamic viscosity, and particle mass.
            const T h            = smoothing_length_for(property, cell_size);
            const T mu           = property.dynamic_viscosity.value_or(T(0));
            const T mass         = property.mass;

            const Vector3<T> position = position_ptr[particle_index];
            const Vector3<T> velocity = velocity_ptr[particle_index];

            // Determine the base cell and the cell-radius needed to cover the
            // particle's smoothing support.
            const Vector3<int> base_cell = particle_cell(position, lower_corner, inv_cell_size, grid_size);
            const int neighbor_search_radius = search_radius_for(h, cell_size);

            Vector3<T> acceleration(T(0), T(0), T(0));

            // Visit all cells that may contain neighbors inside the smoothing radius.
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);

                        // Ignore cells outside the grid bounds.
                        if (!is_valid_neighbor_cell(neighbor_cell, grid_size)) {
                            continue;
                        }

                        // Convert the neighbor cell coordinate into the linear searcher key.
                        const int cell = static_cast<int>(SpatialHashingSearcher<T>::linear_key(
                            neighbor_cell.x,
                            neighbor_cell.y,
                            neighbor_cell.z,
                            grid_size));

                        // Ignore invalid keys defensively.
                        if (cell < 0 || cell >= num_of_cells) {
                            continue;
                        }

                        const int begin = cell_start_ptr[cell];
                        const int end   = cell_end_ptr[cell];

                        // Skip empty or invalid cell spans.
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        // Traverse all particles stored in the neighbor cell.
                        for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                            const int neighbor_index = indices_ptr[sorted_index];

                            // Skip invalid references and self-interaction.
                            if (neighbor_index < 0 || neighbor_index >= particle_count
                                || neighbor_index == particle_index) {
                                continue;
                            }

                            const std::size_t neighbor_species_index = species_ptr[neighbor_index];

                            // Skip neighbors whose material-property lookup would be invalid.
                            if (neighbor_species_index >= static_cast<std::size_t>(num_of_properties)) {
                                continue;
                            }

                            const auto& neighbor_property = properties_ptr[neighbor_species_index];
                            const T neighbor_mass         = neighbor_property.mass;
                            const T neighbor_density      = density_ptr[neighbor_index];

                            // SPH interaction requires a strictly positive neighbor density.
                            if (!(neighbor_density > T(0))) {
                                continue;
                            }

                            const Vector3<T> delta = position - position_ptr[neighbor_index];
                            const T radius         = delta.length();

                            // Only include neighbors strictly inside the smoothing support.
                            //
                            // radius must also be positive because the pressure-gradient
                            // implementation divides by radius when converting the radial
                            // derivative into a vector gradient.
                            if (!(radius > T(0)) || radius > h) {
                                continue;
                            }

                            // Evaluate the SPH pressure-gradient term.
                            const Vector3<T> grad = kernel.pressure_gradient(delta, radius, h);

                            // Build a symmetric pressure contribution using the pressures of
                            // the target and neighbor particles and normalize by neighbor density.
                            const T pressure_term = (pressure_ptr[particle_index] + pressure_ptr[neighbor_index])
                                / (static_cast<T>(2) * neighbor_density);

                            // Add the pressure-force contribution:
                            //
                            //     a_i -= m_j * pressure_term * grad W_ij
                            acceleration -= grad * (neighbor_mass * pressure_term);

                            // Add a viscosity contribution when the target particle defines
                            // a positive dynamic viscosity coefficient.
                            if (mu > T(0)) {
                                const T laplacian = kernel.viscosity_laplacian(radius, h);
                                acceleration += (velocity_ptr[neighbor_index] - velocity)
                                    * (mu * neighbor_mass * laplacian / neighbor_density);
                            }
                        }
                    }
                }
            }

            // Discard the result if the target particle mass is not physically valid.
            if (!(mass > T(0))) {
                acceleration = Vector3<T>(T(0), T(0), T(0));
            }

            // Store the final particle acceleration.
            acceleration_ptr[particle_index] = acceleration;

            // Apply one explicit Euler velocity update:
            //
            //     v_new = v_old + a * dt
            velocity_ptr[particle_index]     = velocity + acceleration * dt;
        });

    // Second pass: accumulate a per-cell average force-like quantity from the
    // updated particle accelerations.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // Empty or invalid cells publish zero field force.
            if (begin < 0 || end <= begin) {
                field_force_ptr[cell] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            Vector3<T> accumulated_force(T(0), T(0), T(0));
            int count = 0;

            // Accumulate mass-weighted particle accelerations across the cell.
            for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                const int particle_index = indices_ptr[sorted_index];
                if (particle_index < 0 || particle_index >= particle_count) {
                    continue;
                }

                const std::size_t species_index = species_ptr[particle_index];
                if (species_index >= static_cast<std::size_t>(num_of_properties)) {
                    continue;
                }

                accumulated_force += acceleration_ptr[particle_index] * properties_ptr[species_index].mass;
                ++count;
            }

            // Publish the average mass-weighted force-like vector for the cell.
            field_force_ptr[cell] = count > 0
                ? accumulated_force / static_cast<T>(count)
                : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
T
SphSolver<T>::smoothing_length_for(const MatrialProperties<T>& property, const T cell_size) noexcept {
    // Prefer an explicitly configured positive smoothing length from the material.
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }

    // Fall back to the search-grid cell size when no smoothing length is specified.
    return cell_size;
}

template <typename T>
T
SphSolver<T>::rest_density_for(const MatrialProperties<T>& property) noexcept {
    // Prefer an explicitly configured positive rest density from the material.
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }

    // Fall back to a neutral default rest density.
    return T(1);
}

template <typename T>
T
SphSolver<T>::pressure_coefficient_for(const MatrialProperties<T>& property) noexcept {
    // Return the equation-of-state pressure coefficient, defaulting to zero when absent.
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphSolver<T>::search_radius_for(const T smoothing_length, const T cell_size) noexcept {
    // Convert the smoothing support radius into a search radius measured in cells.
    //
    // The ceiling ensures the full kernel support is covered during neighbor traversal.
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphSolver<T>::particle_cell(const Vector3<T>& position,
                            const Vector3<T>& lower_corner,
                            const T inverse_cell_size,
                            const Vector3<int>& grid_size) noexcept {
    // Convert the particle position into grid coordinates by:
    //   1. shifting into the grid frame,
    //   2. scaling by the inverse cell size,
    //   3. taking the floor of each component.
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();

    // Clamp the result into the valid grid bounds.
    return atlas::math::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
SphSolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept {
    // Return whether the candidate neighbor cell lies inside the valid 3D grid domain.
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store the universe dependency in the builder.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid dependency in the builder.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial searcher dependency in the builder.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    // Store the selected runtime SPH kernel type in the builder.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
SphSolver<T>::Builder::validate() const {
    // The solver requires a valid universe dependency.
    if (!_universe) {
        throw std::runtime_error("SphSolver::Builder: universe must not be null.");
    }

    // The solver requires a valid fluid dependency.
    if (!_fluid) {
        throw std::runtime_error("SphSolver::Builder: fluid must not be null.");
    }

    // The solver requires a valid spatial searcher dependency.
    if (!_searcher) {
        throw std::runtime_error("SphSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
SphSolver<T>
SphSolver<T>::Builder::build() const {
    // Validate the builder state before constructing a value instance.
    validate();
    return SphSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<SphSolver<T>>
SphSolver<T>::Builder::make_host_shared() const {
    // Validate the builder state before constructing a shared host instance.
    validate();
    return atlas::make_host_shared<SphSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

} // namespace atlas::system