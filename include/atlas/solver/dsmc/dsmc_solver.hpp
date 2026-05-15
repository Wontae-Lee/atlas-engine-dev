#pragma once

#include <atlas/math/constants.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/shuffle/shuffle_operator.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcSolver<T>::DsmcSolver(UniverseHostPtr<T> universe,
                          FluidHostPtr<T> fluid,
                          SpatialHashingSearcherHostPtr<T> searcher,
                          const DsmcKernelType kernel_type,
                          const DsmcApplyMode apply_mode,
                          const DsmcMajorantMode majorant_mode,
                          const int majorant_sample_count,
                          const T majorant_safety_factor,
                          const T majorant_decay_factor,
                          const bool use_pre_collision_snapshot,
                          const bool diagnostics_enabled) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type))
    , _kernel_type(kernel_type)
    , _apply_mode(apply_mode)
    , _majorant_mode(majorant_mode)
    , _majorant_sample_count(majorant_sample_count > 0 ? majorant_sample_count : 1)
    , _majorant_safety_factor(majorant_safety_factor >= T(1) ? majorant_safety_factor : T(1))
    , _majorant_decay_factor(majorant_decay_factor > T(0) && majorant_decay_factor <= T(1)
                                 ? majorant_decay_factor
                                 : T(1))
    , _use_pre_collision_snapshot(use_pre_collision_snapshot)
    , _diagnostics_enabled(diagnostics_enabled) {

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

    if (_diagnostics_enabled) {
        _majorant_violation_count.resize(1);
        _accepted_collision_count.resize(1);
        atlas::parallel_fill<ExecutionPolicy::device>(
            _majorant_violation_count.begin(),
            _majorant_violation_count.end(),
            0);
        atlas::parallel_fill<ExecutionPolicy::device>(
            _accepted_collision_count.begin(),
            _accepted_collision_count.end(),
            0);
    } else {
        _majorant_violation_count.resize(0);
        _accepted_collision_count.resize(0);
    }

    if (_apply_mode == DsmcApplyMode::flattened_atomic) {
        _particle_locks.resize(this->_fluid->particle_count());
        atlas::parallel_fill<ExecutionPolicy::device>(
            _particle_locks.begin(),
            _particle_locks.end(),
            0);
    } else {
        _particle_locks.resize(0);
    }

    if (!make_probe(allocated_solver, probe)) {
        reset_collision_data();
        return;
    }

    if (!build_collision_workload(probe, index, dt)) {
        return;
    }

    if (_apply_mode == DsmcApplyMode::cell_sequential) {
        apply_cell_sequential_collisions(probe, index, dt);
        _flattened_collision_cells.resize(0);
        return;
    }

    if (!build_flattened_collision_workload()) {
        return;
    }

    probe.collision_offsets_ptr         = atlas::raw_pointer_cast(_collision_offsets.data());
    probe.flattened_collision_cells_ptr = atlas::raw_pointer_cast(_flattened_collision_cells.data());
    probe.flattened_collision_count     = static_cast<int>(_flattened_collision_cells.size());

    apply_collisions(probe, index, dt);
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

    _collision_offsets.resize(number_of_cells);
}

template <typename T>
void
DsmcSolver<T>::reset_collision_data() {

    if (!this->_universe) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        _particle_locks.resize(0);
        _pre_collision_velocities.resize(0);
        _species_pair_models.resize(0);
        _majorant_violation_count.resize(0);
        _accepted_collision_count.resize(0);
        _species_pair_models_dirty         = true;
        _species_pair_model_property_count = 0;
        return;
    }

    const auto num_of_cells = this->_universe->number_of_cells();

    if (num_of_cells <= 0) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        _particle_locks.resize(0);
        _pre_collision_velocities.resize(0);
        _species_pair_models.resize(0);
        _majorant_violation_count.resize(0);
        _accepted_collision_count.resize(0);
        _species_pair_models_dirty         = true;
        _species_pair_model_property_count = 0;
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

    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        _collision_offsets.begin(),
        _collision_offsets.end(),
        0);

    _flattened_collision_cells.resize(0);
    _particle_locks.resize(0);
    _pre_collision_velocities.resize(0);
    _species_pair_models.resize(0);
    _majorant_violation_count.resize(0);
    _accepted_collision_count.resize(0);
    _species_pair_models_dirty         = true;
    _species_pair_model_property_count = 0;
}

template <typename T>
bool
DsmcSolver<T>::build_collision_workload(const DsmcSolverProbe& probe,
                                        const int index,
                                        const T dt) {

    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    return measure_cell_collision_statistics(probe, index, dt);
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

    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || max_sigma_g_state == nullptr || collision_count_state == nullptr) {
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

    rebuild_species_pair_models();

    if (_apply_mode == DsmcApplyMode::flattened_atomic && _use_pre_collision_snapshot) {
        _pre_collision_velocities = velocities;
    } else {
        _pre_collision_velocities.resize(0);
    }

    probe.velocity_ptr               = atlas::raw_pointer_cast(velocities.data());
    probe.pre_collision_velocity_ptr = _pre_collision_velocities.empty()
        ? nullptr
        : atlas::raw_pointer_cast(_pre_collision_velocities.data());
    probe.species_ptr                = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr             = atlas::raw_pointer_cast(properties.data());
    probe.species_pair_model_ptr     = _species_pair_models.empty()
            ? nullptr
            : atlas::raw_pointer_cast(_species_pair_models.data());

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
    probe.num_of_properties            = static_cast<int>(properties.size());
    probe.cell_volume                  = this->_universe->cell_volume();
    probe.statistical_weight           = this->_fluid->statistical_weight();
    probe.kernel_type                  = _kernel_type;
    probe.kernel                       = _kernel;
    probe.collision_seed               = _collision_seed++;
    probe.majorant_mode                = _majorant_mode;
    probe.majorant_sample_count        = _majorant_sample_count;
    probe.majorant_safety_factor       = _majorant_safety_factor;
    probe.majorant_decay_factor        = _majorant_decay_factor;
    probe.particle_lock_ptr            = _particle_locks.empty() ? nullptr : atlas::raw_pointer_cast(_particle_locks.data());
    probe.use_pre_collision_snapshot   = _use_pre_collision_snapshot;
    probe.majorant_violation_count_ptr = _majorant_violation_count.empty()
        ? nullptr
        : atlas::raw_pointer_cast(_majorant_violation_count.data());
    probe.accepted_collision_count_ptr = _accepted_collision_count.empty()
        ? nullptr
        : atlas::raw_pointer_cast(_accepted_collision_count.data());

    return true;
}

template <typename T>
void
DsmcSolver<T>::rebuild_species_pair_models() {
    if (!this->_fluid) {
        _species_pair_models.resize(0);
        _species_pair_models_dirty         = true;
        _species_pair_model_property_count = 0;
        return;
    }

    auto& properties = this->_fluid->particle_properties();
    if (properties.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("DsmcSolver: species property count exceeds int range.");
    }

    const int count = static_cast<int>(properties.size());

    if (count <= 0) {
        _species_pair_models.resize(0);
        _species_pair_models_dirty         = true;
        _species_pair_model_property_count = 0;
        return;
    }

    if (count > 0 && count > std::numeric_limits<int>::max() / count) {
        throw std::overflow_error("DsmcSolver: species-pair model count exceeds int range.");
    }

    const int model_count = count * count;

    if (!_species_pair_models_dirty
        && _species_pair_model_property_count == count
        && _species_pair_model_kernel_type == _kernel_type
        && _species_pair_models.size() == static_cast<std::size_t>(model_count)) {
        return;
    }

    _species_pair_models.resize(static_cast<std::size_t>(model_count));

    auto* properties_ptr = atlas::raw_pointer_cast(properties.data());
    auto* models_ptr     = atlas::raw_pointer_cast(_species_pair_models.data());
    const auto type      = _kernel_type;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        model_count,
        [=] ATLAS_DEVICE(const int pair_index) {
            const int lhs_index = pair_index / count;
            const int rhs_index = pair_index - lhs_index * count;

            const auto& lhs = properties_ptr[lhs_index];
            const auto& rhs = properties_ptr[rhs_index];

            DsmcSpeciesPairModel<T> model {};

            if (type == DsmcKernelType::hard_sphere) {
                if (lhs.reference_diameter.has_value() && rhs.reference_diameter.has_value()) {
                    const T diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
                    if (diameter > T(0)) {
                        model.coefficient = static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
                        model.exponent    = T(0.5);
                    }
                }
            } else {
                if (lhs.reference_diameter.has_value() && rhs.reference_diameter.has_value()
                    && lhs.reference_temperature.has_value() && rhs.reference_temperature.has_value()) {
                    const T lhs_mass = lhs.molecular_mass;
                    const T rhs_mass = rhs.molecular_mass;
                    const T mass_sum = lhs_mass + rhs_mass;

                    const T reference_diameter    = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
                    const T reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * T(0.5);
                    const T viscosity_index       = (lhs.viscosity_index.value_or(T(0.5)) + rhs.viscosity_index.value_or(T(0.5))) * T(0.5);
                    const T gamma_argument        = T(2.5) - viscosity_index;

                    if (lhs_mass > T(0) && rhs_mass > T(0) && mass_sum > T(0)
                        && reference_diameter > T(0) && reference_temperature > T(0)
                        && gamma_argument > T(0)) {
                        const T reduced_mass = lhs_mass * rhs_mass / mass_sum;
                        const T thermal_base = (T(2) * static_cast<T>(atlas::boltzmann_constant) * reference_temperature)
                            / reduced_mass;
                        const T reference_area = static_cast<T>(std::numbers::pi_v<double>) * reference_diameter * reference_diameter;
                        const T gamma_value    = static_cast<T>(std::tgamma(static_cast<double>(gamma_argument)));

                        if (thermal_base > T(0) && gamma_value > T(0)) {
                            const T thermal_exponent = viscosity_index - T(0.5);
                            model.coefficient        = reference_area
                                * static_cast<T>(std::pow(static_cast<double>(thermal_base), static_cast<double>(thermal_exponent)))
                                / gamma_value;
                            model.exponent = T(1) - viscosity_index;
                        }
                    }
                }
            }

            models_ptr[pair_index] = model;
        });

    _species_pair_models_dirty         = false;
    _species_pair_model_property_count = count;
    _species_pair_model_kernel_type    = _kernel_type;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DsmcSolverProbe& probe,
                                                 const int index,
                                                 const T dt) {

    if (probe.particle_count < 2 || probe.num_of_cells <= 0 || probe.num_of_properties <= 0
        || probe.species_pair_model_ptr == nullptr || !(probe.cell_volume > T(0))) {
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
            const T previous_max_relative = probe.max_relative_speed_ptr[cell];
            const T previous_max_sigma_g  = probe.max_sigma_g_ptr[cell];

            const auto pair_count         = static_cast<std::int64_t>(count) * static_cast<std::int64_t>(count - 1) / 2;
            const bool use_exact_majorant = probe.majorant_mode == DsmcMajorantMode::exact_all_pairs
                || pair_count <= static_cast<std::int64_t>(probe.majorant_sample_count)
                || !(previous_max_sigma_g > T(0));

            if (use_exact_majorant) {

                for (int a = begin; a < end; ++a) {
                    const int particle_i = probe.indices_ptr[a];

                    for (int b = a + 1; b < end; ++b) {
                        const int particle_j = probe.indices_ptr[b];

                        const std::size_t species_i = probe.species_ptr[particle_i];
                        const std::size_t species_j = probe.species_ptr[particle_j];

                        if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                            || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                            continue;
                        }

                        const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                        const T relative_speed_squared     = relative_velocity.length_squared();

                        if (relative_speed_squared > max_relative_squared) {
                            max_relative_squared = relative_speed_squared;
                        }

                        const T sigma_g = DsmcSolver<T>::sigma_g_from_pair_model(
                            probe.species_pair_model_ptr[species_i * static_cast<std::size_t>(probe.num_of_properties) + species_j],
                            relative_speed_squared);

                        if (sigma_g > max_sigma_g) {
                            max_sigma_g = sigma_g;
                        }
                    }
                }
            } else {

                const int samples = probe.majorant_sample_count > 0 ? probe.majorant_sample_count : 1;

                for (int sample = 0; sample < samples; ++sample) {
                    const auto stream = static_cast<std::uint64_t>(cell) * 0x9e3779b97f4a7c15ull
                        + static_cast<std::uint64_t>(sample);

                    const int lhs_local = DsmcSolver<T>::hashed_index(
                        cell,
                        count,
                        probe.collision_seed + stream + 0x5bf03635c6f2b8d9ull);

                    int rhs_local = DsmcSolver<T>::hashed_index(
                        cell,
                        count - 1,
                        probe.collision_seed + stream + 0x9ddfea08eb382d69ull);
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

                    if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                        || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                        continue;
                    }

                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed_squared     = relative_velocity.length_squared();

                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = DsmcSolver<T>::sigma_g_from_pair_model(
                        probe.species_pair_model_ptr[species_i * static_cast<std::size_t>(probe.num_of_properties) + species_j],
                        relative_speed_squared);

                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }

                const T previous_max_relative_squared = previous_max_relative * previous_max_relative;
                if (max_relative_squared < previous_max_relative_squared) {
                    max_relative_squared = previous_max_relative_squared;
                }

                const T inflated_sampled_max = max_sigma_g * probe.majorant_safety_factor;
                const T decayed_previous_max = previous_max_sigma_g * probe.majorant_decay_factor;
                max_sigma_g                  = decayed_previous_max > inflated_sampled_max ? decayed_previous_max : inflated_sampled_max;
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
                && DsmcSolver<T>::hashed_unit_interval(cell, probe.collision_seed) < remainder) {
                ++collisions;
            }

            probe.collision_count_ptr[cell] = collisions > 0 ? collisions : 0;
        });

    return true;
}

template <typename T>
bool
DsmcSolver<T>::build_flattened_collision_workload() {
    auto* collision_count_state = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

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

    const auto last_cell  = static_cast<std::size_t>(num_of_cells - 1);
    const int last_offset = _collision_offsets[last_cell];
    const int last_count  = collision_count[last_cell];

    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcSolver: flattened collision workload exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;

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
void
DsmcSolver<T>::apply_cell_sequential_collisions(const DsmcSolverProbe& probe,
                                                const int index,
                                                const T) {
    if (probe.num_of_cells <= 0 || probe.collision_count_ptr == nullptr
        || probe.velocity_ptr == nullptr || probe.species_ptr == nullptr
        || probe.properties_ptr == nullptr || probe.species_pair_model_ptr == nullptr
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
                const auto stream = static_cast<std::uint64_t>(cell) * 0x9e3779b97f4a7c15ull
                    + static_cast<std::uint64_t>(local_collision);

                const int lhs_local = DsmcSolver<T>::hashed_index(
                    cell,
                    count,
                    probe.collision_seed + stream + 0x632be59bd9b4e019ull);

                int rhs_local = DsmcSolver<T>::hashed_index(
                    cell,
                    count - 1,
                    probe.collision_seed + stream + 0x85157af5ull);
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

                if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                    || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                    continue;
                }

                Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T sigma_g                = DsmcSolver<T>::sigma_g_from_pair_model(
                    probe.species_pair_model_ptr[species_i * static_cast<std::size_t>(probe.num_of_properties) + species_j],
                    relative_speed_squared);

                if (sigma_g > max_sigma_g) {
                    DsmcSolver<T>::increment_diagnostic_counter(probe.majorant_violation_count_ptr);
                }

                if (!(sigma_g > T(0))) {
                    continue;
                }

                T accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }

                const T accept_sample = DsmcSolver<T>::hashed_unit_interval(
                    cell,
                    probe.collision_seed + stream + 0xda942042e4dd58b5ull);

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
                DsmcSolver<T>::increment_diagnostic_counter(probe.accepted_collision_count_ptr);
            }
        });
}

template <typename T>
void
DsmcSolver<T>::apply_collisions(const DsmcSolverProbe& probe,
                                const int index,
                                const T) {
    if (probe.flattened_collision_count <= 0 || probe.flattened_collision_cells_ptr == nullptr
        || probe.collision_offsets_ptr == nullptr || probe.particle_lock_ptr == nullptr
        || probe.species_pair_model_ptr == nullptr) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            const int cell            = probe.flattened_collision_cells_ptr[work_index];
            const int local_collision = work_index - probe.collision_offsets_ptr[cell];

            const int count     = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g = probe.max_sigma_g_ptr[cell];

            if (count < 2 || local_collision < 0 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            const auto stream = static_cast<std::uint64_t>(cell) * 0x9e3779b97f4a7c15ull
                + static_cast<std::uint64_t>(local_collision);

            const int lhs_local = DsmcSolver<T>::hashed_index(
                cell,
                count,
                probe.collision_seed + stream + 0x632be59bd9b4e019ull);

            int rhs_local = DsmcSolver<T>::hashed_index(
                cell,
                count - 1,
                probe.collision_seed + stream + 0x85157af5ull);
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
                return;
            }

            const std::size_t species_i = probe.species_ptr[particle_i];
            const std::size_t species_j = probe.species_ptr[particle_j];

            if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                return;
            }

            const T accept_sample = DsmcSolver<T>::hashed_unit_interval(
                cell,
                probe.collision_seed + stream + 0xda942042e4dd58b5ull);

            if (probe.use_pre_collision_snapshot && probe.pre_collision_velocity_ptr != nullptr) {
                const Vector3<T> snapshot_lhs_velocity  = probe.pre_collision_velocity_ptr[particle_i];
                const Vector3<T> snapshot_rhs_velocity  = probe.pre_collision_velocity_ptr[particle_j];
                const T snapshot_relative_speed_squared = (snapshot_lhs_velocity - snapshot_rhs_velocity).length_squared();
                const T snapshot_sigma_g                = DsmcSolver<T>::sigma_g_from_pair_model(
                    probe.species_pair_model_ptr[species_i * static_cast<std::size_t>(probe.num_of_properties) + species_j],
                    snapshot_relative_speed_squared);

                if (!(snapshot_sigma_g > T(0))) {
                    return;
                }

                T snapshot_accept_probability = snapshot_sigma_g / max_sigma_g;
                if (snapshot_accept_probability > T(1)) {
                    snapshot_accept_probability = T(1);
                }

                if (accept_sample >= snapshot_accept_probability) {
                    return;
                }
            }

            DsmcSolver<T>::acquire_particle_pair_locks(probe.particle_lock_ptr, particle_i, particle_j);

            Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
            Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

            const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
            const T sigma_g                = DsmcSolver<T>::sigma_g_from_pair_model(
                probe.species_pair_model_ptr[species_i * static_cast<std::size_t>(probe.num_of_properties) + species_j],
                relative_speed_squared);

            if (sigma_g > max_sigma_g) {
                DsmcSolver<T>::increment_diagnostic_counter(probe.majorant_violation_count_ptr);
            }

            T accept_probability = T(0);
            if (sigma_g > T(0)) {
                accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }
            }

            if (accept_sample < accept_probability) {
                probe.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.properties_ptr[species_i],
                    probe.properties_ptr[species_j]);

                probe.velocity_ptr[particle_i] = lhs_velocity;
                probe.velocity_ptr[particle_j] = rhs_velocity;
                DsmcSolver<T>::increment_diagnostic_counter(probe.accepted_collision_count_ptr);
            }

            DsmcSolver<T>::release_particle_pair_locks(probe.particle_lock_ptr, particle_i, particle_j);
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
DsmcSolver<T>::hashed_unit_interval(const int index,
                                    const std::uint64_t seed) noexcept {
    const std::uint64_t value = atlas::system::ShuffleOperator {}(index, seed);
    constexpr double scale    = 1.0 / 9007199254740992.0;
    return static_cast<T>(value >> 11) * T(scale);
}

template <typename T>
int
DsmcSolver<T>::hashed_index(const int index,
                            const int upper_bound,
                            const std::uint64_t seed) noexcept {
    if (upper_bound <= 0) {
        return 0;
    }

    const std::uint64_t value = atlas::system::ShuffleOperator {}(index, seed);
    return static_cast<int>(value % static_cast<std::uint64_t>(upper_bound));
}

template <typename T>
T
DsmcSolver<T>::sigma_g_from_pair_model(const DsmcSpeciesPairModel<T>& model,
                                       const T relative_speed_squared) noexcept {
    if (!(model.coefficient > T(0)) || !(relative_speed_squared > T(0))) {
        return T(0);
    }

    if (model.exponent == T(1)) {
        return model.coefficient * relative_speed_squared;
    }

    if (model.exponent == T(0)) {
        return model.coefficient;
    }

    if (model.exponent == T(0.5)) {
        return model.coefficient * static_cast<T>(std::sqrt(static_cast<double>(relative_speed_squared)));
    }

    return model.coefficient
        * static_cast<T>(std::pow(
            static_cast<double>(relative_speed_squared),
            static_cast<double>(model.exponent)));
}

template <typename T>
void
DsmcSolver<T>::increment_diagnostic_counter(int* counter) noexcept {
    if (counter == nullptr) {
        return;
    }

#if defined(ATLAS_TASKING_CUDA) && defined(__CUDA_ARCH__)
    atomicAdd(counter, 1);
#else
    std::atomic_ref<int> atomic_counter(*counter);
    atomic_counter.fetch_add(1, std::memory_order_relaxed);
#endif
}

template <typename T>
void
DsmcSolver<T>::acquire_particle_pair_locks(int* locks,
                                           const int particle_i,
                                           const int particle_j) noexcept {
    if (locks == nullptr || particle_i == particle_j) {
        return;
    }

    const int first  = particle_i < particle_j ? particle_i : particle_j;
    const int second = particle_i < particle_j ? particle_j : particle_i;

#if defined(ATLAS_TASKING_CUDA)
    while (atomicCAS(&locks[first], 0, 1) != 0) { }
    while (atomicCAS(&locks[second], 0, 1) != 0) { }
#else
    std::atomic_ref<int> first_lock(locks[first]);
    int expected = 0;
    while (!first_lock.compare_exchange_weak(expected, 1, std::memory_order_acquire)) {
        expected = 0;
    }

    std::atomic_ref<int> second_lock(locks[second]);
    expected = 0;
    while (!second_lock.compare_exchange_weak(expected, 1, std::memory_order_acquire)) {
        expected = 0;
    }
#endif
}

template <typename T>
void
DsmcSolver<T>::release_particle_pair_locks(int* locks,
                                           const int particle_i,
                                           const int particle_j) noexcept {
    if (locks == nullptr || particle_i == particle_j) {
        return;
    }

    const int first  = particle_i < particle_j ? particle_i : particle_j;
    const int second = particle_i < particle_j ? particle_j : particle_i;

#if defined(ATLAS_TASKING_CUDA)
    atomicExch(&locks[second], 0);
    atomicExch(&locks[first], 0);
#else
    std::atomic_ref<int> second_lock(locks[second]);
    second_lock.store(0, std::memory_order_release);

    std::atomic_ref<int> first_lock(locks[first]);
    first_lock.store(0, std::memory_order_release);
#endif
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {

    return _kernel_type;
}

template <typename T>
DsmcApplyMode
DsmcSolver<T>::apply_mode() const noexcept {

    return _apply_mode;
}

template <typename T>
DsmcMajorantMode
DsmcSolver<T>::majorant_mode() const noexcept {

    return _majorant_mode;
}

template <typename T>
int
DsmcSolver<T>::majorant_sample_count() const noexcept {

    return _majorant_sample_count;
}

template <typename T>
T
DsmcSolver<T>::majorant_safety_factor() const noexcept {

    return _majorant_safety_factor;
}

template <typename T>
T
DsmcSolver<T>::majorant_decay_factor() const noexcept {

    return _majorant_decay_factor;
}

template <typename T>
bool
DsmcSolver<T>::use_pre_collision_snapshot() const noexcept {

    return _use_pre_collision_snapshot;
}

template <typename T>
bool
DsmcSolver<T>::diagnostics_enabled() const noexcept {

    return _diagnostics_enabled;
}

template <typename T>
int
DsmcSolver<T>::majorant_violation_count() const noexcept {

    return _majorant_violation_count.empty() ? 0 : _majorant_violation_count[0];
}

template <typename T>
int
DsmcSolver<T>::accepted_collision_count() const noexcept {

    return _accepted_collision_count.empty() ? 0 : _accepted_collision_count[0];
}

template <typename T>
void
DsmcSolver<T>::invalidate_species_pair_models() noexcept {

    _species_pair_models_dirty = true;
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
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_apply_mode(const DsmcApplyMode apply_mode) noexcept {
    _apply_mode = apply_mode;
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_majorant_mode(const DsmcMajorantMode majorant_mode) noexcept {
    _majorant_mode = majorant_mode;
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_majorant_sample_count(const int majorant_sample_count) noexcept {
    _majorant_sample_count = majorant_sample_count > 0 ? majorant_sample_count : 1;
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_majorant_safety_factor(const T majorant_safety_factor) noexcept {
    _majorant_safety_factor = majorant_safety_factor >= T(1) ? majorant_safety_factor : T(1);
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_majorant_decay_factor(const T majorant_decay_factor) noexcept {
    _majorant_decay_factor = majorant_decay_factor > T(0) && majorant_decay_factor <= T(1)
        ? majorant_decay_factor
        : T(1);
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_pre_collision_snapshot(const bool use_pre_collision_snapshot) noexcept {
    _use_pre_collision_snapshot = use_pre_collision_snapshot;
    return *this;
}

template <typename T>
typename DsmcSolver<T>::Builder&
DsmcSolver<T>::Builder::with_diagnostics_enabled(const bool diagnostics_enabled) noexcept {
    _diagnostics_enabled = diagnostics_enabled;
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
        _kernel_type,
        _apply_mode,
        _majorant_mode,
        _majorant_sample_count,
        _majorant_safety_factor,
        _majorant_decay_factor,
        _use_pre_collision_snapshot,
        _diagnostics_enabled);
}

template <typename T>
atlas::host_shared_ptr<DsmcSolver<T>>
DsmcSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel_type,
        _apply_mode,
        _majorant_mode,
        _majorant_sample_count,
        _majorant_safety_factor,
        _majorant_decay_factor,
        _use_pre_collision_snapshot,
        _diagnostics_enabled);
}

}