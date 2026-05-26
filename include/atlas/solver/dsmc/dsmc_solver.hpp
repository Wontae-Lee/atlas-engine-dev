#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scheduler/dsmc_cell_sequential_scheduler.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcSolver<T>::DsmcSolver(UniverseHostPtr<T> universe,
                          FluidHostPtr<T> fluid,
                          SpatialHashingSearcherHostPtr<T> searcher,
                          const DsmcKernelType kernel_type,
                          atlas::host_shared_ptr<atlas::scheduler::DsmcCollisionScheduler<T>> scheduler) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type))
    , _scheduler(std::move(scheduler)) {

    if (!_scheduler) {
        _scheduler = atlas::make_host_shared<atlas::scheduler::DsmcCellSequentialScheduler<T>>();
    }
    ensure_states();
}

template <typename T>
void
DsmcSolver<T>::solve(const T dt) {

    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSolver<T>::solve(const DeviceBuffer<int>* allocated_solver,
                     const int index,
                     const T dt) {

    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_states();
        return;
    }

    ensure_states();
    this->_searcher->build();

    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    if (!make_probe()) {
        reset_states();
        return;
    }

    if (!measure_cell_collision_statistics(allocated_solver, index, dt)) {
        return;
    }

    apply_collision(allocated_solver, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_states() {

    if (!this->_universe) {
        return;
    }

    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    if (auto* state = this->_universe->template state<UniverseNumberParticleState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseNumberParticleState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseMaxRelativeSpeedState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseMaxSigmaGState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseMaxSigmaGState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseCollisionCountState<int>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseCollisionCountState<int>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }
}

template <typename T>
void
DsmcSolver<T>::reset_states() {

    if (!this->_universe) {
        return;
    }

    const auto num_of_cells = this->_universe->number_of_cells();

    if (num_of_cells <= 0) {
        return;
    }

    ensure_states();

    auto* number_particle_state    = this->_universe->template state<UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<UniverseCollisionCountState<int>>();

    if (number_particle_state != nullptr) {
        number_particle_state->reset();
    }

    if (max_relative_speed_state != nullptr) {
        max_relative_speed_state->reset();
    }

    if (max_sigma_g_state != nullptr) {
        max_sigma_g_state->reset();
    }

    if (collision_count_state != nullptr) {
        collision_count_state->reset();
    }
}

template <typename T>
bool
DsmcSolver<T>::make_probe() noexcept {

    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return false;
    }

    auto* velocity_state = this->_fluid->template state<FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<FluidSpeciesState<T>>();

    auto* number_particle_state    = this->_universe->template state<UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || max_sigma_g_state == nullptr || collision_count_state == nullptr) {
        return false;
    }

    auto& velocities         = velocity_state->data();
    auto& species            = species_state->data();
    auto& number_particle    = number_particle_state->data();
    auto& max_relative_speed = max_relative_speed_state->data();
    auto& max_sigma_g        = max_sigma_g_state->data();
    auto& collision_count    = collision_count_state->data();
    auto& properties         = this->_fluid->particle_properties();

    _probe.velocity_ptr   = atlas::raw_pointer_cast(velocities.data());
    _probe.species_ptr    = atlas::raw_pointer_cast(species.data());
    _probe.properties_ptr = atlas::raw_pointer_cast(properties.data());

    _probe.number_particle_ptr    = atlas::raw_pointer_cast(number_particle.data());
    _probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
    _probe.max_sigma_g_ptr        = atlas::raw_pointer_cast(max_sigma_g.data());
    _probe.collision_count_ptr    = atlas::raw_pointer_cast(collision_count.data());

    _probe.indices_ptr    = this->_searcher->indices();
    _probe.cell_start_ptr = this->_searcher->cell_start();
    _probe.cell_end_ptr   = this->_searcher->cell_end();

    _probe.particle_count     = static_cast<int>(this->_fluid->particle_count());
    _probe.num_of_cells       = this->_universe->number_of_cells();
    _probe.cell_volume        = this->_universe->cell_volume();
    _probe.statistical_weight = this->_fluid->statistical_weight();
    _probe.kernel             = _kernel;

    _probe.collision_seed = _collision_seed++;

    return true;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                                 const int index,
                                                 const T dt) {

    const auto probe = _probe;

    if (probe.particle_count < 2 || probe.num_of_cells <= 0
        || probe.properties_ptr == nullptr || !(probe.cell_volume > T(0))) {
        reset_states();
        return false;
    }

    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;
    const bool collision_count_limited = _scheduler && _scheduler->limits_collision_count();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {

            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.max_sigma_g_ptr[cell]        = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.max_sigma_g_ptr[cell]        = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }

            const int count        = end - begin;
            T max_relative_squared = T(0);
            T max_sigma_g          = T(0);

            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];

                for (int b = a + 1; b < end; ++b) {
                    const int particle_j        = probe.indices_ptr[b];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed_squared     = relative_velocity.length_squared();

                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = probe.kernel.sigma_g(
                        probe.properties_ptr,
                        species_i,
                        species_j,
                        relative_speed_squared);

                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            probe.number_particle_ptr[cell]    = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);
            probe.max_sigma_g_ptr[cell]        = max_sigma_g;

            if (count < 2 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const T ntc_pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T ntc_count      = ntc_pair_count * max_sigma_g * probe.statistical_weight * dt / probe.cell_volume;

            if (!(ntc_count > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            constexpr int max_collision_count  = std::numeric_limits<int>::max();
            const T max_collision_count_scalar = static_cast<T>(max_collision_count);
            if (ntc_count >= max_collision_count_scalar) {
                probe.collision_count_ptr[cell] = collision_count_limited
                    ? count / 2
                    : max_collision_count;
                return;
            }

            const T base_count = std::floor(ntc_count);
            int collisions     = static_cast<int>(base_count);
            const T remainder  = ntc_count - base_count;

            if (remainder > T(0) && atlas::sampling::sample_hashed_unit_interval<T>(cell, probe.collision_seed) < remainder) {
                ++collisions;
            }

            if (collision_count_limited) {
                const int max_unique_pairs = count / 2;
                if (collisions > max_unique_pairs) {
                    collisions = max_unique_pairs;
                }
            }

            probe.collision_count_ptr[cell] = collisions > 0 ? collisions : 0;
        });

    return true;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {

    return _kernel.type;
}

template <typename T>
const atlas::host_shared_ptr<atlas::scheduler::DsmcCollisionScheduler<T>>&
DsmcSolver<T>::scheduler() const noexcept {
    return _scheduler;
}

template <typename T>
void
DsmcSolver<T>::apply_collision(const DeviceBuffer<int>* allocated_solver,
                               const int index,
                               const T) {
    if (_scheduler) {
        _scheduler->schedule(_probe, allocated_solver, index);
    }
}

}
