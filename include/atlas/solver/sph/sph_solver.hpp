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
    _kernel = SphKernel<T>(kernel_type);
    ensure_universe_states();
}

template <typename T>
typename SphSolver<T>::Builder
SphSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
SphKernelType
SphSolver<T>::kernel_type() const noexcept {
    return _kernel.type;
}

template <typename T>
void
SphSolver<T>::solve(const T dt) {
    if (!initialize_sph_context()) {
        return;
    }
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphSolver: dt must be positive.");
    }
    if (!prepare_particle_fields()) {
        reset_universe_fields();
        return;
    }
    SphSolverProbe probe;
    if (!make_probe(nullptr, probe)) {
        return;
    }
    update(probe);
    accumulate_acceleration(probe, dt);
}

template <typename T>
void
SphSolver<T>::solve(const DeviceBuffer<int>*, const int, const T) {
}

template <typename T>
void
SphSolver<T>::ensure_universe_states() {
    if (!this->_universe) {
        return;
    }
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
    }
    if (!this->_universe->template has_state<atlas::universe::UniverseFieldForceState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseFieldForceState<T>>(number_of_cells);
    }
}

template <typename T>
bool
SphSolver<T>::initialize_sph_context() noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_universe_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_universe_fields();
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        return false;
    }
    ensure_universe_states();
    this->_searcher->build();
    return true;
}

template <typename T>
bool
SphSolver<T>::make_probe(const DeviceBuffer<int>* allocated_solver,
                         SphSolverProbe& probe) noexcept {
    return make_probe(
        this->_universe,
        this->_fluid,
        this->_searcher,
        _kernel,
        allocated_solver,
        probe);
}

template <typename T>
bool
SphSolver<T>::make_probe(const UniverseHostPtr<T>& universe,
                         const FluidHostPtr<T>& fluid,
                         const SpatialHashingSearcherHostPtr<T>& searcher,
                         const SphKernel<T>& kernel,
                         const DeviceBuffer<int>* allocated_solver,
                         SphSolverProbe& probe) noexcept {
    if (!universe || !fluid || !searcher) {
        return false;
    }
    auto* position_state = fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* field_force_state
        = universe->template state<atlas::universe::UniverseFieldForceState<T>>();
    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr
        || number_particle_state == nullptr || field_force_state == nullptr) {
        return false;
    }
    auto& positions            = position_state->data();
    auto& velocities           = velocity_state->data();
    auto& species              = species_state->data();
    auto& properties           = fluid->particle_properties();
    auto& number_particle      = number_particle_state->data();
    auto& field_force          = field_force_state->data();
    probe.position_ptr         = atlas::raw_pointer_cast(positions.data());
    probe.velocity_ptr         = atlas::raw_pointer_cast(velocities.data());
    probe.species_ptr          = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr       = atlas::raw_pointer_cast(properties.data());
    probe.number_particle_ptr  = atlas::raw_pointer_cast(number_particle.data());
    probe.field_force_ptr      = atlas::raw_pointer_cast(field_force.data());
    probe.indices_ptr          = searcher->indices();
    probe.cell_start_ptr       = searcher->cell_start();
    probe.cell_end_ptr         = searcher->cell_end();
    probe.allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;
    probe.lower_corner         = searcher->lower_corner();
    probe.grid_size            = searcher->grid_size();
    probe.inverse_cell_size    = searcher->inverse_cell_size();
    probe.cell_size            = searcher->cell_size();
    probe.particle_count       = static_cast<int>(fluid->particle_count());
    probe.num_of_cells         = universe->number_of_cells();
    probe.num_of_properties    = static_cast<int>(properties.size());
    probe.kernel               = kernel;
    return true;
}

template <typename T>
bool
SphSolver<T>::prepare_particle_fields() {
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    if (particle_count <= 0) {
        _density.resize(0);
        _pressure.resize(0);
        _acceleration.resize(0);
        reset_universe_fields();
        return false;
    }
    _density.resize(static_cast<std::size_t>(particle_count));
    _pressure.resize(static_cast<std::size_t>(particle_count));
    _acceleration.resize(static_cast<std::size_t>(particle_count));
    atlas::parallel_fill<ExecutionPolicy::device>(_density.begin(), _density.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_pressure.begin(), _pressure.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_acceleration.begin(), _acceleration.end(), Vector3<T>(T(0), T(0), T(0)));
    reset_universe_fields();
    return true;
}

template <typename T>
void
SphSolver<T>::reset_universe_fields() {
    if (!this->_universe) {
        return;
    }
    ensure_universe_states();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* field_force_state
        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();
    if (number_particle_state != nullptr) {
        auto& number_particle = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(number_particle.begin(), number_particle.end(), T(0));
    }
    if (field_force_state != nullptr) {
        auto& field_force = field_force_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(field_force.begin(), field_force.end(), Vector3<T>(T(0), T(0), T(0)));
    }
}

template <typename T>
void
SphSolver<T>::update() {
    SphSolverProbe probe;
    if (!make_probe(nullptr, probe)) {
        return;
    }
    update(probe);
}

template <typename T>
void
SphSolver<T>::update(const SphSolverProbe& probe) {
    estimate_particle_density_and_pressure(probe);
    update_cell_number_particles(probe);
}

template <typename T>
void
SphSolver<T>::estimate_particle_density_and_pressure() {
    SphSolverProbe probe;
    if (!make_probe(nullptr, probe)) {
        return;
    }
    estimate_particle_density_and_pressure(probe);
}

template <typename T>
void
SphSolver<T>::estimate_particle_density_and_pressure(const SphSolverProbe& probe) {
    auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                density_ptr[particle_index]  = T(0);
                pressure_ptr[particle_index] = T(0);
                return;
            }
            const auto& property         = probe.properties_ptr[species_index];
            const T h                    = smoothing_length_for(property, probe.cell_size);
            const T rest_density         = rest_density_for(property);
            const T k                    = pressure_coefficient_for(property);
            const Vector3<T> position    = probe.position_ptr[particle_index];
            const Vector3<int> base_cell = particle_cell(
                position,
                probe.lower_corner,
                probe.inverse_cell_size,
                probe.grid_size);
            const int neighbor_search_radius = search_radius_for(h, probe.cell_size);
            T density                        = T(0);
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);
                        if (!is_valid_neighbor_cell(neighbor_cell, probe.grid_size)) {
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
            if (!(density > T(0))) {
                density = rest_density;
            }
            density_ptr[particle_index]  = density;
            pressure_ptr[particle_index] = k * (density - rest_density);
        });
}

template <typename T>
void
SphSolver<T>::update_cell_number_particles() {
    SphSolverProbe probe;
    if (!make_probe(nullptr, probe)) {
        return;
    }
    update_cell_number_particles(probe);
}

template <typename T>
void
SphSolver<T>::update_cell_number_particles(const SphSolverProbe& probe) {
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
SphSolver<T>::accumulate_acceleration(const T dt) {
    SphSolverProbe probe;
    if (!make_probe(nullptr, probe)) {
        return;
    }
    accumulate_acceleration(probe, dt);
}

template <typename T>
void
SphSolver<T>::accumulate_acceleration(const SphSolverProbe& probe, const T dt) {
    const auto* density_ptr  = atlas::raw_pointer_cast(_density.data());
    const auto* pressure_ptr = atlas::raw_pointer_cast(_pressure.data());
    auto* acceleration_ptr   = atlas::raw_pointer_cast(_acceleration.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.particle_count,
        [=] ATLAS_DEVICE(const int particle_index) {
            const std::size_t species_index = probe.species_ptr[particle_index];
            if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                acceleration_ptr[particle_index] = Vector3<T>(T(0), T(0), T(0));
                return;
            }
            const auto& property         = probe.properties_ptr[species_index];
            const T h                    = smoothing_length_for(property, probe.cell_size);
            const T mu                   = property.dynamic_viscosity.value_or(T(0));
            const T mass                 = property.mass;
            const Vector3<T> position    = probe.position_ptr[particle_index];
            const Vector3<T> velocity    = probe.velocity_ptr[particle_index];
            const Vector3<int> base_cell = particle_cell(
                position,
                probe.lower_corner,
                probe.inverse_cell_size,
                probe.grid_size);
            const int neighbor_search_radius = search_radius_for(h, probe.cell_size);
            Vector3<T> acceleration(T(0), T(0), T(0));
            for (int dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                for (int dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                    for (int dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        const Vector3<int> neighbor_cell = base_cell + Vector3<int>(dx, dy, dz);
                        if (!is_valid_neighbor_cell(neighbor_cell, probe.grid_size)) {
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
                            const T neighbor_mass         = neighbor_property.mass;
                            const T neighbor_density      = density_ptr[neighbor_index];
                            if (!(neighbor_density > T(0))) {
                                continue;
                            }
                            const Vector3<T> delta = position - probe.position_ptr[neighbor_index];
                            const T radius         = delta.length();
                            if (!(radius > T(0)) || radius > h) {
                                continue;
                            }
                            const Vector3<T> grad = probe.kernel.pressure_gradient(delta, radius, h);
                            const T pressure_term = (pressure_ptr[particle_index] + pressure_ptr[neighbor_index])
                                / (static_cast<T>(2) * neighbor_density);
                            acceleration -= grad * (neighbor_mass * pressure_term);
                            if (mu > T(0)) {
                                const T laplacian = probe.kernel.viscosity_laplacian(radius, h);
                                acceleration += (probe.velocity_ptr[neighbor_index] - velocity)
                                    * (mu * neighbor_mass * laplacian / neighbor_density);
                            }
                        }
                    }
                }
            }
            if (!(mass > T(0))) {
                acceleration = Vector3<T>(T(0), T(0), T(0));
            }
            acceleration_ptr[particle_index]   = acceleration;
            probe.velocity_ptr[particle_index] = velocity + acceleration * dt;
        });
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
            probe.field_force_ptr[cell] = count > 0
                ? accumulated_force / static_cast<T>(count)
                : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
T
SphSolver<T>::smoothing_length_for(const MaterialProperties<T>& property, const T cell_size) noexcept {
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }
    return cell_size;
}

template <typename T>
T
SphSolver<T>::rest_density_for(const MaterialProperties<T>& property) noexcept {
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }
    return T(1);
}

template <typename T>
T
SphSolver<T>::pressure_coefficient_for(const MaterialProperties<T>& property) noexcept {
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphSolver<T>::search_radius_for(const T smoothing_length, const T cell_size) noexcept {
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphSolver<T>::particle_cell(const Vector3<T>& position,
                            const Vector3<T>& lower_corner,
                            const T inverse_cell_size,
                            const Vector3<int>& grid_size) noexcept {
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();
    return atlas::math::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
SphSolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept {
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphSolver<T>::Builder&
SphSolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
SphSolver<T>::Builder::validate() const {
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
    validate();
    return SphSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<SphSolver<T>>
SphSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<SphSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

}