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
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(kernel_type)
    , _group_particle_count(group_particle_count > 0 ? group_particle_count : 5) {
    ensure_universe_states();
}

template <typename T>
typename SphGatewaySolver<T>::Builder
SphGatewaySolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
SphKernelType
SphGatewaySolver<T>::kernel_type() const noexcept {
    return _kernel.type;
}

template <typename T>
int
SphGatewaySolver<T>::group_particle_count() const noexcept {
    return _group_particle_count;
}

template <typename T>
void
SphGatewaySolver<T>::solve(const T dt) {
    solve(nullptr, 0, dt);
}

template <typename T>
void
SphGatewaySolver<T>::solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    if (!initialize_context()) {
        return;
    }
    if (!(dt > T(0))) {
        throw std::invalid_argument("SphGatewaySolver: dt must be positive.");
    }
    if (!prepare_group_fields()) {
        reset_universe_fields();
        return;
    }
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    update_cell_particle_counts(probe, index);
    build_group_representatives(probe, index);
    estimate_group_density_and_pressure(probe, index);
    update_group_motion(probe, index, dt);
    scatter_group_states_to_particles(probe, index);
}

template <typename T>
void
SphGatewaySolver<T>::ensure_universe_states() {
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
SphGatewaySolver<T>::initialize_context() noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
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
    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
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
    ensure_universe_states();
    this->_searcher->build();
    return true;
}

template <typename T>
bool
SphGatewaySolver<T>::prepare_group_fields() {
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells   = this->_universe->number_of_cells();
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
    reset_universe_fields();
    return true;
}

template <typename T>
void
SphGatewaySolver<T>::reset_universe_fields() {
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
SphGatewaySolver<T>::update_cell_particle_counts(const DeviceBuffer<int>* allocated_solver, const int index) {
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    update_cell_particle_counts(probe, index);
}

template <typename T>
void
SphGatewaySolver<T>::update_cell_particle_counts(const typename SphSolver<T>::SphSolverProbe& probe, const int index) {
    auto* cell_group_count_ptr     = atlas::raw_pointer_cast(_cell_group_count.data());
    const int group_particle_count = _group_particle_count;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell] = T(0);
                cell_group_count_ptr[cell]      = 0;
                return;
            }
            const int begin                 = probe.cell_start_ptr[cell];
            const int end                   = probe.cell_end_ptr[cell];
            const int count                 = (begin >= 0 && end > begin) ? (end - begin) : 0;
            probe.number_particle_ptr[cell] = static_cast<T>(count);
            cell_group_count_ptr[cell]      = group_count_for_cell(count, group_particle_count);
        });
}

template <typename T>
void
SphGatewaySolver<T>::build_group_representatives(const DeviceBuffer<int>* allocated_solver, const int index) {
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    build_group_representatives(probe, index);
}

template <typename T>
void
SphGatewaySolver<T>::build_group_representatives(const typename SphSolver<T>::SphSolverProbe& probe, const int index) {
    auto* group_position_ptr         = atlas::raw_pointer_cast(_group_position.data());
    auto* group_velocity_ptr         = atlas::raw_pointer_cast(_group_velocity.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    auto* group_mass_ptr             = atlas::raw_pointer_cast(_group_mass.data());
    auto* group_member_count_ptr     = atlas::raw_pointer_cast(_group_member_count.data());
    auto* group_species_ptr          = atlas::raw_pointer_cast(_group_species.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    const int group_particle_count   = _group_particle_count;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                return;
            }
            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];
            if (begin < 0 || group_count <= 0) {
                return;
            }
            for (int group_local = 0; group_local < group_count; ++group_local) {
                const int representative_index = begin + group_local;
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
                for (int sorted_index = sorted_begin; sorted_index < sorted_end; ++sorted_index) {
                    const int particle_index = probe.indices_ptr[sorted_index];
                    if (particle_index < 0 || particle_index >= probe.particle_count) {
                        continue;
                    }
                    if (member_count == 0) {
                        representative_species = probe.species_ptr[particle_index];
                    }
                    position_sum += probe.position_ptr[particle_index];
                    velocity_sum += probe.velocity_ptr[particle_index];
                    ++member_count;
                    const std::size_t species_index = probe.species_ptr[particle_index];
                    if (species_index < static_cast<std::size_t>(probe.num_of_properties)) {
                        mass_sum += probe.properties_ptr[species_index].mass;
                    }
                }
                if (member_count <= 0) {
                    continue;
                }
                const Vector3<T> mean_position               = position_sum / static_cast<T>(member_count);
                const Vector3<T> mean_velocity               = velocity_sum / static_cast<T>(member_count);
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
SphGatewaySolver<T>::estimate_group_density_and_pressure(const DeviceBuffer<int>* allocated_solver, const int index) {
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    estimate_group_density_and_pressure(probe, index);
}

template <typename T>
void
SphGatewaySolver<T>::estimate_group_density_and_pressure(const typename SphSolver<T>::SphSolverProbe& probe, const int index) {
    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_density_ptr          = atlas::raw_pointer_cast(_group_density.data());
    auto* group_pressure_ptr         = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
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
                if (species_index >= static_cast<std::size_t>(probe.num_of_properties)) {
                    group_density_ptr[lhs_index]  = T(0);
                    group_pressure_ptr[lhs_index] = T(0);
                    continue;
                }
                const auto& property     = probe.properties_ptr[species_index];
                const T smoothing_length = smoothing_length_for(property, probe.cell_size);
                const T rest_density     = rest_density_for(property);
                const T pressure_coeff   = pressure_coefficient_for(property);
                const Vector3<T> lhs_pos = group_position_ptr[lhs_index];
                T density                = T(0);
                for (int rhs_group = 0; rhs_group < group_count; ++rhs_group) {
                    const int rhs_index         = begin + rhs_group;
                    const Vector3<T> delta      = lhs_pos - group_position_ptr[rhs_index];
                    const T radius              = delta.length();
                    const T representative_mass = group_mass_ptr[rhs_index];
                    density += representative_mass * probe.kernel.density_weight(radius, smoothing_length);
                }
                if (!(density > T(0))) {
                    density = rest_density;
                }
                group_density_ptr[lhs_index]  = density;
                group_pressure_ptr[lhs_index] = pressure_coeff * (density - rest_density);
            }
        });
}

template <typename T>
void
SphGatewaySolver<T>::update_group_motion(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    update_group_motion(probe, index, dt);
}

template <typename T>
void
SphGatewaySolver<T>::update_group_motion(const typename SphSolver<T>::SphSolverProbe& probe, const int index, const T dt) {
    const auto* group_position_ptr   = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_velocity_ptr   = atlas::raw_pointer_cast(_group_velocity.data());
    const auto* group_mass_ptr       = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_density_ptr    = atlas::raw_pointer_cast(_group_density.data());
    const auto* group_pressure_ptr   = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* group_species_ptr    = atlas::raw_pointer_cast(_group_species.data());
    auto* group_updated_pos_ptr      = atlas::raw_pointer_cast(_group_updated_position.data());
    auto* group_updated_vel_ptr      = atlas::raw_pointer_cast(_group_updated_velocity.data());
    const auto* cell_group_count_ptr = atlas::raw_pointer_cast(_cell_group_count.data());
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
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
                const auto& property     = probe.properties_ptr[species_index];
                const T smoothing_length = smoothing_length_for(property, probe.cell_size);
                const T viscosity        = property.dynamic_viscosity.value_or(T(0));
                const T group_mass       = group_mass_ptr[lhs_index];
                const Vector3<T> lhs_pos = group_position_ptr[lhs_index];
                const Vector3<T> lhs_vel = group_velocity_ptr[lhs_index];
                Vector3<T> acceleration(T(0), T(0), T(0));
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
                    const T radius         = delta.length();
                    if (!(radius > T(0)) || radius > smoothing_length) {
                        continue;
                    }
                    const Vector3<T> grad = probe.kernel.pressure_gradient(delta, radius, smoothing_length);
                    const T pressure_term = (group_pressure_ptr[lhs_index] + group_pressure_ptr[rhs_index])
                        / (static_cast<T>(2) * rhs_density);
                    acceleration -= grad * (group_mass_ptr[rhs_index] * pressure_term);
                    if (viscosity > T(0)) {
                        const T laplacian = probe.kernel.viscosity_laplacian(radius, smoothing_length);
                        acceleration += (group_velocity_ptr[rhs_index] - lhs_vel)
                            * (viscosity * group_mass_ptr[rhs_index] * laplacian / rhs_density);
                    }
                }
                if (!(group_mass > T(0))) {
                    continue;
                }
                const Vector3<T> updated_velocity = lhs_vel + acceleration * dt;
                group_updated_vel_ptr[lhs_index]  = updated_velocity;
                group_updated_pos_ptr[lhs_index]  = lhs_pos;
                cell_force += acceleration * group_mass;
                ++active_group_count;
            }
            probe.field_force_ptr[cell] = active_group_count > 0
                ? cell_force / static_cast<T>(active_group_count)
                : Vector3<T>(T(0), T(0), T(0));
        });
}

template <typename T>
void
SphGatewaySolver<T>::scatter_group_states_to_particles(const DeviceBuffer<int>* allocated_solver, const int index) {
    typename SphSolver<T>::SphSolverProbe probe;
    if (!SphSolver<T>::make_probe(this->_universe, this->_fluid, this->_searcher, _kernel, allocated_solver, probe)) {
        return;
    }
    scatter_group_states_to_particles(probe, index);
}

template <typename T>
void
SphGatewaySolver<T>::scatter_group_states_to_particles(const typename SphSolver<T>::SphSolverProbe& probe, const int index) {
    const auto* group_updated_vel_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());
    const auto* cell_group_count_ptr  = atlas::raw_pointer_cast(this->_cell_group_count.data());
    const int group_particle_count    = _group_particle_count;
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                return;
            }
            const int begin       = probe.cell_start_ptr[cell];
            const int group_count = cell_group_count_ptr[cell];
            if (begin < 0 || group_count <= 0) {
                return;
            }
            for (int group_local = 0; group_local < group_count; ++group_local) {
                const int representative_index = begin + group_local;
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
                    probe.velocity_ptr[particle_index] = group_updated_vel_ptr[representative_index];
                }
            }
        });
}

template <typename T>
T
SphGatewaySolver<T>::smoothing_length_for(const MaterialProperties<T>& property, const T cell_size) noexcept {
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }
    return cell_size;
}

template <typename T>
T
SphGatewaySolver<T>::rest_density_for(const MaterialProperties<T>& property) noexcept {
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }
    return T(1);
}

template <typename T>
T
SphGatewaySolver<T>::pressure_coefficient_for(const MaterialProperties<T>& property) noexcept {
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
SphGatewaySolver<T>::search_radius_for(const T smoothing_length, const T cell_size) noexcept {
    return static_cast<int>(std::ceil(smoothing_length / cell_size));
}

template <typename T>
Vector3<int>
SphGatewaySolver<T>::particle_cell(const Vector3<T>& position,
                                   const Vector3<T>& lower_corner,
                                   const T inverse_cell_size,
                                   const Vector3<int>& grid_size) noexcept {
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();
    return atlas::math::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
SphGatewaySolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept {
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
int
SphGatewaySolver<T>::group_count_for_cell(const int particle_count, const int group_particle_count) noexcept {
    if (particle_count <= 0 || group_particle_count <= 0) {
        return 0;
    }
    return (particle_count + group_particle_count - 1) / group_particle_count;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_kernel_type(const SphKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename SphGatewaySolver<T>::Builder&
SphGatewaySolver<T>::Builder::with_group_particle_count(const int group_particle_count) noexcept {
    _group_particle_count = group_particle_count;
    return *this;
}

template <typename T>
void
SphGatewaySolver<T>::Builder::validate() const {
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
    validate();
    return SphGatewaySolver<T>(_universe, _fluid, _searcher, _kernel_type, _group_particle_count);
}

template <typename T>
atlas::host_shared_ptr<SphGatewaySolver<T>>
SphGatewaySolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<SphGatewaySolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type,
        _group_particle_count);
}

}