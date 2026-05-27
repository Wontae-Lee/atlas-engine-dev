#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcSpartaSolver<T>::DsmcSpartaSolver(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher)) {
    ensure_states();
}

template <typename T>
void
DsmcSpartaSolver<T>::solve(const T dt) {
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSpartaSolver<T>::solve(const DeviceBuffer<int>* allocated_solver,
                           const int index,
                           const T dt) {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_states();
        return;
    }

    ensure_states();
    this->_searcher->build();

    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSpartaSolver: dt must be positive.");
    }

    if (!make_probe()) {
        reset_states();
        return;
    }

    apply_collisions(allocated_solver, index, dt);
}

template <typename T>
void
DsmcSpartaSolver<T>::ensure_states() {
    if (!this->_universe) {
        return;
    }

    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    if (auto* state = this->_universe->template state<UniverseNumberParticleState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseNumberParticleState<T>>(number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseMaxRelativeSpeedState<T>>(number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseMaxSigmaGState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseMaxSigmaGState<T>>(number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<UniverseCollisionCountState<int>>();
        state == nullptr) {
        this->_universe->template emplace_state<UniverseCollisionCountState<int>>(number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }
}

template <typename T>
void
DsmcSpartaSolver<T>::reset_states() {
    if (!this->_universe) {
        return;
    }

    ensure_states();

    if (auto* state = this->_universe->template state<UniverseNumberParticleState<T>>();
        state != nullptr) {
        state->reset();
    }

    if (auto* state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
        state != nullptr) {
        state->reset();
    }

    if (auto* state = this->_universe->template state<UniverseMaxSigmaGState<T>>();
        state != nullptr) {
        state->reset();
    }

    if (auto* state = this->_universe->template state<UniverseCollisionCountState<int>>();
        state != nullptr) {
        state->reset();
    }
}

template <typename T>
bool
DsmcSpartaSolver<T>::make_probe() noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return false;
    }

    auto* velocity_state = this->_fluid->template state<FluidVelocityState<T>>();
    auto* species_state = this->_fluid->template state<FluidSpeciesState<T>>();

    auto* number_particle_state = this->_universe->template state<UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state = this->_universe->template state<UniverseMaxSigmaGState<T>>();
    auto* collision_count_state = this->_universe->template state<UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || max_sigma_g_state == nullptr || collision_count_state == nullptr) {
        return false;
    }

    auto& velocities = velocity_state->data();
    auto& species = species_state->data();
    auto& properties = this->_fluid->particle_properties();

    _probe.velocity_ptr = atlas::raw_pointer_cast(velocities.data());
    _probe.species_ptr = atlas::raw_pointer_cast(species.data());
    _probe.properties_ptr = atlas::raw_pointer_cast(properties.data());

    _probe.number_particle_ptr = atlas::raw_pointer_cast(number_particle_state->data().data());
    _probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed_state->data().data());
    _probe.max_sigma_g_ptr = atlas::raw_pointer_cast(max_sigma_g_state->data().data());
    _probe.collision_count_ptr = atlas::raw_pointer_cast(collision_count_state->data().data());

    _probe.indices_ptr = this->_searcher->indices();
    _probe.cell_start_ptr = this->_searcher->cell_start();
    _probe.cell_end_ptr = this->_searcher->cell_end();

    _probe.particle_count = static_cast<int>(this->_fluid->particle_count());
    _probe.num_of_cells = this->_universe->number_of_cells();
    _probe.cell_volume = this->_universe->cell_volume();
    _probe.statistical_weight = this->_fluid->statistical_weight();
    _probe.collision_seed = _collision_seed++;

    return true;
}

template <typename T>
void
DsmcSpartaSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver,
                                      const int index,
                                      const T dt) {
    const auto probe = _probe;
    const auto kernel = _kernel;

    if (probe.particle_count < 2 || probe.num_of_cells <= 0
        || probe.properties_ptr == nullptr || !(probe.cell_volume > T(0))) {
        reset_states();
        return;
    }

    const int* allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell] = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.max_sigma_g_ptr[cell] = T(0);
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end = probe.cell_end_ptr[cell];
            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell] = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.max_sigma_g_ptr[cell] = T(0);
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const int count = end - begin;
            T max_relative_squared = T(0);
            T max_collision_frequency = T(0);

            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];
                for (int b = a + 1; b < end; ++b) {
                    const int particle_j = probe.indices_ptr[b];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    const T relative_speed_squared = (probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j]).length_squared();
                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T collision_frequency = kernel.collision_frequency(
                        probe.properties_ptr,
                        species_i,
                        species_j,
                        relative_speed_squared);
                    if (collision_frequency > max_collision_frequency) {
                        max_collision_frequency = collision_frequency;
                    }
                }
            }

            probe.number_particle_ptr[cell] = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);
            probe.max_sigma_g_ptr[cell] = max_collision_frequency;

            if (count < 2 || !(max_collision_frequency > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const T pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T expected_attempts = pair_count * max_collision_frequency * probe.statistical_weight * dt / probe.cell_volume;
            if (!(expected_attempts > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            constexpr int max_attempt_count = std::numeric_limits<int>::max();
            const T max_attempt_count_scalar = static_cast<T>(max_attempt_count);
            if (expected_attempts >= max_attempt_count_scalar) {
                probe.collision_count_ptr[cell] = max_attempt_count;
                return;
            }

            const T base_count = std::floor(expected_attempts);
            int attempts = static_cast<int>(base_count);
            const T remainder = expected_attempts - base_count;
            if (remainder > T(0) && atlas::sampling::sample_hashed_unit_interval<T>(cell, probe.collision_seed) < remainder) {
                ++attempts;
            }

            probe.collision_count_ptr[cell] = attempts > 0 ? attempts : 0;

            for (int attempt = 0; attempt < attempts; ++attempt) {
                const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(attempt);
                const int lhs_local = atlas::sampling::sample_hashed_index(
                    cell,
                    count,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);
                int rhs_local = atlas::sampling::sample_hashed_index(
                    cell,
                    count - 1,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);
                if (rhs_local >= lhs_local) {
                    ++rhs_local;
                }

                const int particle_i = probe.indices_ptr[begin + lhs_local];
                const int particle_j = probe.indices_ptr[begin + rhs_local];
                const std::size_t species_i = probe.species_ptr[particle_i];
                const std::size_t species_j = probe.species_ptr[particle_j];

                Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];
                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T collision_frequency = kernel.collision_frequency(
                    probe.properties_ptr,
                    species_i,
                    species_j,
                    relative_speed_squared);
                T probability = collision_frequency / max_collision_frequency;
                if (probability > T(1)) {
                    probability = T(1);
                }

                const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                    cell,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

                if (accept_sample >= probability) {
                    continue;
                }

                kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.properties_ptr[species_i],
                    probe.properties_ptr[species_j]);

                probe.velocity_ptr[particle_i] = lhs_velocity;
                probe.velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

} // namespace atlas::system
