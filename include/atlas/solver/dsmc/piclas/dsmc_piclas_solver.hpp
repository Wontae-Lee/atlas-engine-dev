#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/piclas/piclas_pairing.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcPiclasSolver<T>::DsmcPiclasSolver(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher)) {
    ensure_states();
}

template <typename T>
void
DsmcPiclasSolver<T>::solve(const T dt) {
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcPiclasSolver<T>::solve(const DeviceBuffer<int>* allocated_solver,
                           const int index,
                           const T dt) {
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_states();
        return;
    }

    ensure_states();
    this->_searcher->build();

    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcPiclasSolver: dt must be positive.");
    }

    if (!make_probe()) {
        reset_states();
        return;
    }

    apply_collisions(allocated_solver, index, dt);
}

template <typename T>
void
DsmcPiclasSolver<T>::ensure_states() {
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
DsmcPiclasSolver<T>::reset_states() {
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
DsmcPiclasSolver<T>::make_probe() noexcept {
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
DsmcPiclasSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver,
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
            const int pair_count = count / 2;
            T max_relative_squared = T(0);
            T max_sigma_g = T(0);

            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];
                for (int b = a + 1; b < end; ++b) {
                    const int particle_j = probe.indices_ptr[b];
                    const T relative_speed_squared = (probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j]).length_squared();
                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = kernel.sigma_g(
                        probe.properties_ptr,
                        probe.species_ptr[particle_i],
                        probe.species_ptr[particle_j],
                        relative_speed_squared);
                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            probe.number_particle_ptr[cell] = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);
            probe.max_sigma_g_ptr[cell] = max_sigma_g;

            if (pair_count <= 0 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            probe.collision_count_ptr[cell] = pair_count;

            for (int local_pair = 0; local_pair < pair_count; ++local_pair) {
                int lhs_local = -1;
                int rhs_local = -1;
                PiclasPairing::select_pair_offsets(
                    lhs_local,
                    rhs_local,
                    local_pair,
                    count,
                    cell,
                    probe.collision_seed + static_cast<std::uint64_t>(cell));

                if (lhs_local < 0 || rhs_local < 0) {
                    continue;
                }

                const int particle_i = probe.indices_ptr[begin + lhs_local];
                const int particle_j = probe.indices_ptr[begin + rhs_local];
                const std::size_t species_i = probe.species_ptr[particle_i];
                const std::size_t species_j = probe.species_ptr[particle_j];

                Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];
                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T sigma_g = kernel.sigma_g(
                    probe.properties_ptr,
                    species_i,
                    species_j,
                    relative_speed_squared);

                const int count_i = DsmcPiclasSolver<T>::species_count(probe, begin, end, species_i);
                const int count_j = species_i == species_j
                    ? count_i
                    : DsmcPiclasSolver<T>::species_count(probe, begin, end, species_j);
                const int case_count = DsmcPiclasSolver<T>::pair_case_count(
                    probe,
                    cell,
                    begin,
                    count,
                    species_i,
                    species_j,
                    probe.collision_seed + static_cast<std::uint64_t>(cell));

                const T probability = PiclasVhsKernel<T>::collision_probability(
                    sigma_g,
                    static_cast<T>(count_i),
                    static_cast<T>(count_j),
                    species_i == species_j,
                    case_count,
                    probe.statistical_weight,
                    dt,
                    probe.cell_volume);

                const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(local_pair);
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

template <typename T>
int
DsmcPiclasSolver<T>::species_count(const DsmcProbe<T>& probe,
                                   const int begin,
                                   const int end,
                                   const std::size_t species) noexcept {
    int count = 0;
    for (int sorted = begin; sorted < end; ++sorted) {
        const int particle = probe.indices_ptr[sorted];
        if (probe.species_ptr[particle] == species) {
            ++count;
        }
    }
    return count;
}

template <typename T>
int
DsmcPiclasSolver<T>::pair_case_count(const DsmcProbe<T>& probe,
                                     const int cell,
                                     const int begin,
                                     const int count,
                                     const std::size_t species_i,
                                     const std::size_t species_j,
                                     const std::uint64_t seed) noexcept {
    const int pair_count = count / 2;
    int case_count = 0;

    for (int local_pair = 0; local_pair < pair_count; ++local_pair) {
        int lhs_local = -1;
        int rhs_local = -1;
        PiclasPairing::select_pair_offsets(lhs_local, rhs_local, local_pair, count, cell, seed);
        if (lhs_local < 0 || rhs_local < 0) {
            continue;
        }

        const int particle_i = probe.indices_ptr[begin + lhs_local];
        const int particle_j = probe.indices_ptr[begin + rhs_local];
        const std::size_t lhs_species = probe.species_ptr[particle_i];
        const std::size_t rhs_species = probe.species_ptr[particle_j];

        if ((lhs_species == species_i && rhs_species == species_j)
            || (lhs_species == species_j && rhs_species == species_i)) {
            ++case_count;
        }
    }

    return case_count;
}

} // namespace atlas::system
