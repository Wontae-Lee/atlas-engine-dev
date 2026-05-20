#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
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
                          const DsmcKernelType kernel_type) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type)) {

    ensure_universe_states();
}

template <typename T>
typename DsmcSolver<T>::Builder
DsmcSolver<T>::builder() noexcept {
    return Builder {};
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
        reset_collision_data();
        return;
    }

    ensure_universe_states();
    this->_searcher->build();

    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    DsmcSolverProbe probe;

    if (!make_probe(allocated_solver, probe)) {
        reset_collision_data();
        return;
    }

    if (!measure_cell_collision_statistics(probe, index, dt)) {
        return;
    }

    apply_cell_sequential_collisions(probe, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_universe_states() {

    if (!this->_universe) {
        return;
    }

    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    if (auto* state = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>();
        state == nullptr) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxSigmaGState<T>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

    if (auto* state = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();
        state == nullptr) {
        this->_universe->template emplace_state<atlas::universe::UniverseCollisionCountState<int>>(
            number_of_cells);
    } else if (state->data().size() != number_of_cells) {
        state->data().resize(number_of_cells);
    }

}

template <typename T>
void
DsmcSolver<T>::reset_collision_data() {

    if (!this->_universe) {
        return;
    }

    const auto num_of_cells = this->_universe->number_of_cells();

    if (num_of_cells <= 0) {
        return;
    }

    ensure_universe_states();

    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (number_particle_state != nullptr) {
        auto& buffer = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    if (max_relative_speed_state != nullptr) {
        auto& buffer = max_relative_speed_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    if (max_sigma_g_state != nullptr) {
        auto& buffer = max_sigma_g_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    if (collision_count_state != nullptr) {
        auto& buffer = collision_count_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), 0);
    }
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

    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

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

    probe.velocity_ptr   = atlas::raw_pointer_cast(velocities.data());
    probe.species_ptr    = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr = atlas::raw_pointer_cast(properties.data());

    probe.number_particle_ptr    = atlas::raw_pointer_cast(number_particle.data());
    probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
    probe.max_sigma_g_ptr        = atlas::raw_pointer_cast(max_sigma_g.data());
    probe.collision_count_ptr    = atlas::raw_pointer_cast(collision_count.data());

    probe.indices_ptr    = this->_searcher->indices();
    probe.cell_start_ptr = this->_searcher->cell_start();
    probe.cell_end_ptr   = this->_searcher->cell_end();

    probe.allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    probe.particle_count               = static_cast<int>(this->_fluid->particle_count());
    probe.num_of_cells                 = this->_universe->number_of_cells();
    probe.cell_volume                  = this->_universe->cell_volume();
    probe.statistical_weight           = this->_fluid->statistical_weight();
    probe.kernel                       = _kernel;
    probe.collision_seed               = _collision_seed++;

    return true;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DsmcSolverProbe& probe,
                                                 const int index,
                                                 const T dt) {

    if (probe.particle_count < 2 || probe.num_of_cells <= 0
        || probe.properties_ptr == nullptr || !(probe.cell_volume > T(0))) {
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

            const int count               = end - begin;
            T max_relative_squared        = T(0);
            T max_sigma_g                 = T(0);

            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];

                for (int b = a + 1; b < end; ++b) {
                    const int particle_j = probe.indices_ptr[b];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed_squared     = relative_velocity.length_squared();

                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = DsmcSolver<T>::sigma_g(
                        probe.kernel,
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
                probe.collision_count_ptr[cell] = max_collision_count;
                return;
            }

            const T base_count = std::floor(ntc_count);
            int collisions     = static_cast<int>(base_count);
            const T remainder  = ntc_count - base_count;

            if (remainder > T(0)
                && atlas::sampling::sample_hashed_unit_interval<T>(cell, probe.collision_seed) < remainder) {
                ++collisions;
            }

            probe.collision_count_ptr[cell] = collisions > 0 ? collisions : 0;
        });

    return true;
}

template <typename T>
void
DsmcSolver<T>::apply_cell_sequential_collisions(const DsmcSolverProbe& probe,
                                                const int index,
                                                const T) {
    if (probe.num_of_cells <= 0 || probe.collision_count_ptr == nullptr
        || probe.velocity_ptr == nullptr || probe.species_ptr == nullptr
        || probe.properties_ptr == nullptr
        || probe.indices_ptr == nullptr
        || probe.cell_start_ptr == nullptr || probe.cell_end_ptr == nullptr
        || probe.number_particle_ptr == nullptr || probe.max_sigma_g_ptr == nullptr) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                return;
            }

            const int collisions = probe.collision_count_ptr[cell];
            const int count      = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g  = probe.max_sigma_g_ptr[cell];

            if (collisions <= 0 || count < 2 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                return;
            }

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(local_collision);

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

                const int particle_i = DsmcSolver<T>::nth_valid_particle(
                    lhs_local,
                    begin,
                    end,
                    probe.particle_count,
                    probe.indices_ptr);

                const int particle_j = DsmcSolver<T>::nth_valid_particle(
                    rhs_local,
                    begin,
                    end,
                    probe.particle_count,
                    probe.indices_ptr);

                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                const std::size_t species_i = probe.species_ptr[particle_i];
                const std::size_t species_j = probe.species_ptr[particle_j];

                Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T sigma_g                = DsmcSolver<T>::sigma_g(
                    probe.kernel,
                    probe.properties_ptr,
                    species_i,
                    species_j,
                    relative_speed_squared);

                if (!(sigma_g > T(0))) {
                    continue;
                }

                T accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }

                const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                    cell,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

                if (accept_sample >= accept_probability) {
                    continue;
                }

                probe.kernel(
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
T
DsmcSolver<T>::sigma_g(const DsmcKernel<T>& kernel,
                       const MaterialProperties<T>* properties_ptr,
                       const std::size_t species_i,
                       const std::size_t species_j,
                       const T relative_speed_squared) noexcept {
    if (!(relative_speed_squared > T(0))) {
        return T(0);
    }

    const T relative_speed = static_cast<T>(std::sqrt(static_cast<double>(relative_speed_squared)));
    return DsmcKernel<T>::cross_section(
               kernel.type,
               properties_ptr[species_i],
               properties_ptr[species_j],
               relative_speed)
        * relative_speed;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {

    return _kernel.type;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel = DsmcKernel<T>(kernel_type);
    return *this;
}

template <typename T>
void
DsmcSolver<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcSolver<T>
DsmcSolver<T>::Builder::build() const {
    validate();
    return DsmcSolver<T>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type);
}

template <typename T>
atlas::host_shared_ptr<DsmcSolver<T>>
DsmcSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type);
}

}
