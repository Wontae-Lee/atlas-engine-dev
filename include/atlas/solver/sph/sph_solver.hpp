#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
SphSolver<T>::SphSolver(UniverseHostPtr<T> universe,
                        FluidHostPtr<T> fluid,
                        SearcherHostPtr<T> searcher,
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

    // Allocate per-particle working buffers; active entries are overwritten by later stages.
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
    _probe.neighbor_offsets_ptr = this->_searcher->neighbor_offsets();
    _probe.neighbor_indices_ptr = this->_searcher->neighbor_indices();
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

            // SPH uses the searcher cell size as the kernel support.
            const auto& property      = probe.properties_ptr[species_index];
            const T h                 = probe.cell_size;
            const T h_squared         = h * h;
            const T rest_density      = SphSolver<T>::rest_density(property);
            const T k                 = pressure_coefficient(property);
            const Vector3<T> position = probe.position_ptr[particle_index];
            T density                 = property.mass * probe.kernel.density_weight(T(0), h);

            const int begin = probe.neighbor_offsets_ptr[particle_index];
            const int end   = probe.neighbor_offsets_ptr[particle_index + 1];

            for (int neighbor_offset = begin; neighbor_offset < end; ++neighbor_offset) {
                const int neighbor_index = probe.neighbor_indices_ptr[neighbor_offset];

                if (neighbor_index < 0 || neighbor_index >= probe.particle_count) {
                    continue;
                }

                const Vector3<T> delta = position - probe.position_ptr[neighbor_index];
                const T radius_squared = delta.length_squared();

                if (radius_squared > h_squared) {
                    continue;
                }

                const T radius = atlas::math::sqrt_nonnegative(radius_squared);

                const std::size_t neighbor_species_index = probe.species_ptr[neighbor_index];

                if (neighbor_species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                    continue;
                }

                const T neighbor_mass = probe.properties_ptr[neighbor_species_index].mass;

                density += neighbor_mass * probe.kernel.density_weight(radius, h);
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

    // Searcher ranges contain active particles, so the range length is the cell count.
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

            probe.number_particle_ptr[cell] = static_cast<T>(end - begin);
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
            const T h                 = probe.cell_size;
            const T h_squared         = h * h;
            const T mu                = property.dynamic_viscosity.value_or(T(0));
            const T mass              = property.mass;
            const Vector3<T> position = probe.position_ptr[particle_index];
            const Vector3<T> velocity = probe.velocity_ptr[particle_index];
            Vector3<T> acceleration(T(0), T(0), T(0));

            const int begin = probe.neighbor_offsets_ptr[particle_index];
            const int end   = probe.neighbor_offsets_ptr[particle_index + 1];

            for (int neighbor_offset = begin; neighbor_offset < end; ++neighbor_offset) {
                const int neighbor_index = probe.neighbor_indices_ptr[neighbor_offset];

                if (neighbor_index < 0 || neighbor_index >= probe.particle_count) {
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
                const T radius_squared = delta.length_squared();

                if (!(radius_squared > T(0)) || radius_squared > h_squared) {
                    continue;
                }

                const T radius = atlas::math::sqrt_nonnegative(radius_squared);

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
SphSolver<T>::Builder::with_searcher(SearcherHostPtr<T> searcher) noexcept {
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
