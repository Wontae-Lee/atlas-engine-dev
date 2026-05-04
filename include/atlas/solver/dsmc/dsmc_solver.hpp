#pragma once
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <cmath>
#include <stdexcept>
namespace atlas::system {
template <typename T>
DsmcSolver<T>::DsmcSolver(UniverseHostPtr<T> universe,
                          FluidHostPtr<T> fluid,
                          SpatialHashingSearcherHostPtr<T> searcher,
                          const DsmcKernelType kernel_type) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type))
    , _kernel_type(kernel_type) {
    ensure_universe_states();
}

template <typename T>
void
DsmcSolver<T>::solve(const T dt) {
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSolver<T>::solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    if (!build_collision_workload(allocated_solver, index, dt)) {
        return;
    }
    apply_collisions(allocated_solver, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_universe_states() {
    if (!this->_universe) {
        return;
    }
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
    }
    if (!this->_universe->template has_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(number_of_cells);
    }
    if (!this->_universe->template has_state<atlas::universe::UniverseCollisionCountState<int>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseCollisionCountState<int>>(number_of_cells);
    }
    _collision_offsets.resize(number_of_cells);
}

template <typename T>
void
DsmcSolver<T>::reset_collision_data() {
    if (!this->_universe) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }
    const auto num_of_cells = this->_universe->number_of_cells();
    if (num_of_cells <= 0) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }
    ensure_universe_states();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();
    if (number_particle_state != nullptr) {
        auto& buffer = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }
    if (max_relative_speed_state != nullptr) {
        auto& buffer = max_relative_speed_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }
    if (collision_count_state != nullptr) {
        auto& buffer = collision_count_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), 0);
    }
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }
    atlas::parallel_fill<ExecutionPolicy::device>(_collision_offsets.begin(), _collision_offsets.end(), 0);
    _flattened_collision_cells.resize(0);
}

template <typename T>
bool
DsmcSolver<T>::build_collision_workload(const DeviceBuffer<int>* allocated_solver,
                                        const int index,
                                        const T dt) {
    if (!initialize_collision_context()) {
        return false;
    }
    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }
    if (!measure_cell_collision_statistics(allocated_solver, index, dt)) {
        return false;
    }
    return true;
}

template <typename T>
bool
DsmcSolver<T>::initialize_collision_context() noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_collision_data();
        return false;
    }
    ensure_universe_states();
    this->_searcher->build();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();
    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }
    return true;
}

template <typename T>
bool
DsmcSolver<T>::make_probe(const DeviceBuffer<int>* allocated_solver,
                          DsmcSolverProbe& probe) noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return false;
    }
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();
    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || collision_count_state == nullptr) {
        return false;
    }
    auto& velocities             = velocity_state->data();
    auto& species                = species_state->data();
    auto& number_particle        = number_particle_state->data();
    auto& max_relative_speed     = max_relative_speed_state->data();
    auto& collision_count        = collision_count_state->data();
    auto& properties             = this->_fluid->particle_properties();
    probe.velocity_ptr           = atlas::raw_pointer_cast(velocities.data());
    probe.species_ptr            = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr         = atlas::raw_pointer_cast(properties.data());
    probe.number_particle_ptr    = atlas::raw_pointer_cast(number_particle.data());
    probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
    probe.collision_count_ptr    = atlas::raw_pointer_cast(collision_count.data());
    probe.indices_ptr            = this->_searcher->indices();
    probe.cell_start_ptr         = this->_searcher->cell_start();
    probe.cell_end_ptr           = this->_searcher->cell_end();
    probe.allocated_solver_ptr   = allocated_solver != nullptr
          ? atlas::raw_pointer_cast(allocated_solver->data())
          : nullptr;
    probe.particle_count         = static_cast<int>(this->_fluid->particle_count());
    probe.num_of_cells           = this->_universe->number_of_cells();
    probe.num_of_properties      = static_cast<int>(properties.size());
    probe.cell_volume            = this->_universe->cell_volume();
    probe.statistical_weight     = this->_fluid->statistical_weight();
    probe.kernel_type            = _kernel_type;
    probe.kernel                 = _kernel;
    return true;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                                 const int index,
                                                 const T dt) {
    DsmcSolverProbe probe;
    if (!make_probe(allocated_solver, probe)) {
        reset_collision_data();
        return false;
    }
    if (probe.particle_count < 2 || probe.num_of_cells <= 0 || probe.num_of_properties <= 0
        || !(probe.cell_volume > T(0))) {
        reset_collision_data();
        return false;
    }
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }
            const int count = end - begin;
            T max_relative  = T(0);
            T max_sigma_g   = T(0);
            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];
                for (int b = a + 1; b < end; ++b) {
                    const int particle_j        = probe.indices_ptr[b];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];
                    if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                        || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                        continue;
                    }
                    const Vector3<T> relative_velocity
                        = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed = relative_velocity.length();
                    if (relative_speed > max_relative) {
                        max_relative = relative_speed;
                    }
                    const T sigma = DsmcKernel<T>::cross_section(
                        probe.kernel_type,
                        probe.properties_ptr[species_i],
                        probe.properties_ptr[species_j],
                        relative_speed);
                    const T sigma_g = sigma * relative_speed;
                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }
            probe.number_particle_ptr[cell]    = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative;
            if (count < 2 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }
            const T pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T ntc_count
                = pair_count * max_sigma_g * probe.statistical_weight * dt / probe.cell_volume;
            int collisions      = static_cast<int>(std::floor(ntc_count));
            const int max_pairs = count * (count - 1) / 2;
            if (collisions < 0) {
                collisions = 0;
            } else if (collisions > max_pairs) {
                collisions = max_pairs;
            }
            probe.collision_count_ptr[cell] = collisions;
        });
    return true;
}

template <typename T>
bool
DsmcSolver<T>::build_flattened_collision_workload() noexcept {
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();
    if (collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }
    auto& collision_count     = collision_count_state->data();
    auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
    const int num_of_cells    = this->_universe->number_of_cells();
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }
    atlas::exclusive_scan<ExecutionPolicy::device>(
        collision_count.begin(),
        collision_count.end(),
        _collision_offsets.begin(),
        0);
    const auto last_cell       = static_cast<std::size_t>(num_of_cells - 1);
    const int total_collisions = _collision_offsets[last_cell] + collision_count[last_cell];
    if (total_collisions <= 0) {
        _flattened_collision_cells.resize(0);
        return false;
    }
    _flattened_collision_cells.resize(static_cast<std::size_t>(total_collisions));
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [                              =,
         collision_offsets_ptr         = atlas::raw_pointer_cast(_collision_offsets.data()),
         flattened_collision_cells_ptr = atlas::raw_pointer_cast(_flattened_collision_cells.data())] ATLAS_DEVICE(const int cell) {
            const int collisions = collision_count_ptr[cell];
            if (collisions <= 0) {
                return;
            }
            const int offset = collision_offsets_ptr[cell];
            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                flattened_collision_cells_ptr[offset + local_collision] = cell;
            }
        });
    return true;
}

template <typename T>
int
DsmcSolver<T>::nth_valid_particle(const int nth,
                                  const int begin,
                                  const int end,
                                  const int particle_count,
                                  const int* indices_ptr) noexcept {
    const int sorted_index = begin + nth;
    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }
    const int particle_index = indices_ptr[sorted_index];
    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

template <typename T>
int
DsmcSolver<T>::pair_ordinal_to_rhs(const int count, int ordinal, int& lhs_local) noexcept {
    lhs_local = 0;
    while (lhs_local < count - 1) {
        const int row_width = count - lhs_local - 1;
        if (ordinal < row_width) {
            return lhs_local + 1 + ordinal;
        }
        ordinal -= row_width;
        ++lhs_local;
    }
    return -1;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {
    return _kernel_type;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::collision_offsets() const noexcept {
    return _collision_offsets;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::flattened_collision_cells() const noexcept {
    return _flattened_collision_cells;
}

}