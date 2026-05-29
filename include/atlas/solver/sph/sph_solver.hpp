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
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher)) {
    // Initialize the runtime SPH kernel and ensure required universe output states exist.
    _kernel = SphKernel<T>(kernel_type);
    ensure_states();
}

template <typename T>
typename SphSolver<T>::Builder
SphSolver<T>::builder() noexcept {
    // Return a fresh builder for fluent solver construction.
    return Builder {};
}

template <typename T>
SphKernelType
SphSolver<T>::kernel_type() const noexcept {
    // Expose the kernel type stored by the runtime kernel wrapper.
    return _kernel.type;
}

template <typename T>
void
SphSolver<T>::solve(const T dt) {
    // Validate dependencies, required fluid states, universe states, and searcher ordering.
    if (!initialize_context()) {
        return;
    }

    // SPH velocity integration requires a positive time step.
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphSolver: dt must be positive.");
    }

    // Allocate and clear per-particle working buffers.
    if (!prepare_fields()) {
        reset_fields();
        return;
    }

    // Refresh the cached probe containing raw device pointers for kernel execution.
    if (!make_probe()) {
        return;
    }

    // Compute density/pressure and particle counts.
    update();

    // Compute acceleration, update particle velocity, and write cell force output.
    accelerate(dt);
}

template <typename T>
void
SphSolver<T>::solve(const DeviceBuffer<int>*, const int, const T) {
    // Plain SPH solver does not support solver-index filtering.
    // Filtered execution is implemented by specialized solvers.
}

template <typename T>
void
SphSolver<T>::ensure_states() {
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
SphSolver<T>::initialize_context() noexcept {
    // Required runtime dependencies must exist before launching SPH kernels.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    // SPH requires particle position, velocity, and species states.
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }

    // Ensure output states exist and rebuild spatial hashing for the current particles.
    ensure_states();
    this->_searcher->build();

    return true;
}

template <typename T>
bool
SphSolver<T>::make_probe() noexcept {
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
bool
SphSolver<T>::prepare_fields() {
    // No particle means no SPH update.
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    if (particle_count <= 0) {
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        reset_fields();
        return false;
    }

    // Allocate one working entry per particle.
    _density.resize(static_cast<std::size_t>(particle_count));
    _pressure.resize(static_cast<std::size_t>(particle_count));
    _acceleration.resize(static_cast<std::size_t>(particle_count));

    // Clear transient particle fields before the current solve step.
    atlas::parallel_fill<ExecutionPolicy::device>(_density.begin(), _density.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_pressure.begin(), _pressure.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(
        _acceleration.begin(),
        _acceleration.end(),
        Vector3<T>(T(0), T(0), T(0)));

    // Clear universe outputs for this step.
    reset_fields();

    return true;
}

template <typename T>
void
SphSolver<T>::reset_fields() {
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
void
SphSolver<T>::update() {
    // First compute particle density and pressure, then update cell particle counts.
    estimate_density();
    count_particles();
}

template <typename T>
void
SphSolver<T>::estimate_density() {
    const auto probe = _probe;
    auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());

    // Compute density and pressure independently for each particle.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            // Skip particles with invalid species.
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                density_ptr[particle_index]  = T(0);
                pressure_ptr[particle_index] = T(0);
                return;
            }

            // Resolve material parameters for this particle.
            const auto& property      = probe.properties_ptr[species_index];
            const T h                 = smoothing_length(property, probe.cell_size);
            const T rest_density      = SphSolver<T>::rest_density(property);
            const T k                 = pressure_coefficient(property);
            const Vector3<T> position = probe.position_ptr[particle_index];

            // Locate the particle's base search cell.
            const Vector3<int> base_cell = particle_cell(position, probe.lower_corner, probe.inverse_cell_size, probe.grid_size);

            const int neighbor_search_radius = search_radius(h, probe.cell_size);
            T density                        = T(0);

            // Visit all neighboring grid cells inside the smoothing-length search radius.
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);

                        if (!valid_cell(neighbor_cell, probe.grid_size)) {
                            continue;
                        }

                        const int cell = static_cast<int>(SpatialHashingSearcher<T>::linear_key(
                            neighbor_cell.x,
                            neighbor_cell.y,
                            neighbor_cell.z,
                            probe.grid_size));

                        if (cell < 0 || cell >= probe.num_of_cells) {
                            continue;
                        }

                        const int begin = probe.cell_start_ptr[cell];
                        const int end   = probe.cell_end_ptr[cell];

                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        // Accumulate density contribution from valid neighbor particles.
                        for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                            const int neighbor_index = probe.indices_ptr[sorted_index];

                            if (neighbor_index < 0 || neighbor_index >= probe.particle_count) {
                                continue;
                            }

                            const Vector3<T> delta = position - probe.position_ptr[neighbor_index];
                            const T radius         = delta.length();

                            if (radius > h) {
                                continue;
                            }

                            const std::size_t neighbor_species_index = probe.species_ptr[neighbor_index];

                            if (neighbor_species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                                continue;
                            }

                            const T neighbor_mass = probe.properties_ptr[neighbor_species_index].mass;

                            density += neighbor_mass * probe.kernel.density_weight(radius, h);
                        }
                    }
                }
            }

            // Use rest density as a fallback for isolated or numerically invalid particles.
            if (!(density > T(0))) {
                density = rest_density;
            }

            // Store density and equation-of-state pressure.
            density_ptr[particle_index]  = density;
            pressure_ptr[particle_index] = k * (density - rest_density);
        });
}

template <typename T>
void
SphSolver<T>::count_particles() {
    const auto probe = _probe;

    // Count valid particles in each search cell.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell] = T(0);
                return;
            }

            int count = 0;

            for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index >= 0 && particle_index < probe.particle_count) {
                    ++count;
                }
            }

            probe.number_particle_ptr[cell] = static_cast<T>(count);
        });
}

template <typename T>
void
SphSolver<T>::accelerate(const T dt) {
    const auto probe = _probe;
    const auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    const auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());
    auto* acceleration_ptr   = atlas::raw_pointer_cast(_acceleration.data());

    // Compute particle acceleration and update particle velocity.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            // Skip particles with invalid species.
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                acceleration_ptr[particle_index] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            // Resolve material properties and particle state.
            const auto& property      = probe.properties_ptr[species_index];
            const T h                 = smoothing_length(property, probe.cell_size);
            const T mu                = property.dynamic_viscosity.value_or(T(0));
            const T mass              = property.mass;
            const Vector3<T> position = probe.position_ptr[particle_index];
            const Vector3<T> velocity = probe.velocity_ptr[particle_index];

            const Vector3<int> base_cell = particle_cell(position, probe.lower_corner, probe.inverse_cell_size, probe.grid_size);

            const int neighbor_search_radius = search_radius(h, probe.cell_size);
            Vector3<T> acceleration(T(0), T(0), T(0));

            // Accumulate pressure and viscosity interaction from neighbor particles.
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);

                        if (!valid_cell(neighbor_cell, probe.grid_size)) {
                            continue;
                        }

                        const int cell = static_cast<int>(SpatialHashingSearcher<T>::linear_key(
                            neighbor_cell.x,
                            neighbor_cell.y,
                            neighbor_cell.z,
                            probe.grid_size));

                        if (cell < 0 || cell >= probe.num_of_cells) {
                            continue;
                        }

                        const int begin = probe.cell_start_ptr[cell];
                        const int end   = probe.cell_end_ptr[cell];

                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                            const int neighbor_index = probe.indices_ptr[sorted_index];

                            if (neighbor_index < 0 || neighbor_index >= probe.particle_count
                                || neighbor_index == particle_index) {
                                continue;
                            }

                            const std::size_t neighbor_species_index = probe.species_ptr[neighbor_index];

                            if (neighbor_species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                                continue;
                            }

                            const auto& neighbor_property = probe.properties_ptr[neighbor_species_index];

                            const T neighbor_mass    = neighbor_property.mass;
                            const T neighbor_density = density_ptr[neighbor_index];

                            if (!(neighbor_density > T(0))) {
                                continue;
                            }

                            const Vector3<T> delta = position - probe.position_ptr[neighbor_index];
                            const T radius         = delta.length();

                            if (!(radius > T(0)) || radius > h) {
                                continue;
                            }

                            // Add pressure-gradient acceleration.
                            const Vector3<T> grad = probe.kernel.pressure_gradient(delta, radius, h);

                            const T pressure_term = (pressure_ptr[particle_index] + pressure_ptr[neighbor_index])
                                / (static_cast<T>(2) * neighbor_density);

                            acceleration -= grad * (neighbor_mass * pressure_term);

                            // Add viscosity acceleration when the material viscosity is enabled.
                            if (mu > T(0)) {
                                const T laplacian = probe.kernel.viscosity_laplacian(radius, h);

                                acceleration += (probe.velocity_ptr[neighbor_index] - velocity)
                                    * (mu * neighbor_mass * laplacian / neighbor_density);
                            }
                        }
                    }
                }
            }

            // Invalid mass disables acceleration for this particle.
            if (!(mass > T(0))) {
                acceleration = Vector3<T>(T(0), T(0), T(0));
            }

            // Store acceleration and explicitly update velocity.
            acceleration_ptr[particle_index]   = acceleration;
            probe.velocity_ptr[particle_index] = velocity + acceleration * dt;
        });

    // Convert particle accelerations into per-cell averaged force output.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.field_force_ptr[cell] = Vector3<T>(T(0), T(0), T(0));
                return;
            }

            Vector3<T> accumulated_force(T(0), T(0), T(0));
            int count = 0;

            for (int sorted_index = begin; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                    continue;
                }

                accumulated_force += acceleration_ptr[particle_index] * probe.properties_ptr[species_index].mass;

                ++count;
            }

            probe.field_force_ptr[cell] = count > 0 ? accumulated_force / static_cast<T>(count)
                                                    : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
T
SphSolver<T>::smoothing_length(const MaterialProperties<T>& property,
                               const T cell_size) noexcept {
    // Prefer material smoothing length; fall back to searcher cell size.
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }

    return cell_size;
}

template <typename T>
T
SphSolver<T>::rest_density(const MaterialProperties<T>& property) noexcept {
    // Prefer material rest density; fall back to unit density.
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }

    return T(1);
}

template <typename T>
T
SphSolver<T>::pressure_coefficient(const MaterialProperties<T>& property) noexcept {
    // Missing pressure coefficient disables pressure response.
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphSolver<T>::search_radius(const T smoothing_length, const T cell_size) noexcept {
    // Convert smoothing length to an integer grid-cell search radius.
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphSolver<T>::particle_cell(const Vector3<T>& position,
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
SphSolver<T>::valid_cell(const Vector3<int>& cell,
                         const Vector3<int>& grid_size) noexcept {
    // Check half-open grid bounds: [0, grid_size).
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store universe dependency.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store fluid dependency.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store searcher dependency.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    // Store selected SPH kernel type.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
SphSolver<T>::Builder::validate() const {
    // Builder requires all runtime dependencies.
    if (!_universe) {
        throw std::runtime_error("SphSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("SphSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("SphSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
SphSolver<T>
SphSolver<T>::Builder::build() const {
    // Validate configuration and construct a solver value.
    validate();
    return SphSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<SphSolver<T>>
SphSolver<T>::Builder::make_host_shared() const {
    // Validate configuration and construct a host-shared solver.
    validate();
    return atlas::make_host_shared<SphSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

} // namespace atlas::system
