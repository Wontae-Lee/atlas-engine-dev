#pragma once

#include <atlas/atomic/atomic.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace atlas::system {

template <typename T>
HybridDsmcSphSolver<T>::HybridDsmcSphSolver(UniverseHostPtr<T> universe,
                                            FluidHostPtr<T> fluid,
                                            SpatialHashingSearcherHostPtr<T> searcher,
                                            const T grouping_length,
                                            const int sph_particle_threshold,
                                            const SphKernelType sph_kernel_type,
                                            const DsmcKernelType dsmc_kernel_type,
                                            const bool pairing_without_replacement) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _sph_kernel(sph_kernel_type)
    , _dsmc_kernel(dsmc_kernel_type)
    , _grouping_length(grouping_length)
    , _sph_particle_threshold(sph_particle_threshold > 0 ? sph_particle_threshold : 5)
    , _pairing_without_replacement(pairing_without_replacement) {
    ensure_states();
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder
HybridDsmcSphSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
HybridDsmcSphSolver<T>::solve(const T dt) {
    solve(nullptr, 0, dt);
}

template <typename T>
void
HybridDsmcSphSolver<T>::solve(const DeviceBuffer<int>*, const int, const T dt) {
    if (!initialize_context()) {
        return;
    }

    if (!(dt > T(0))) {
        throw std::invalid_argument("HybridDsmcSphSolver: dt must be positive.");
    }

    if (!prepare_fields() || !make_probe()) {
        reset_states();
        return;
    }

    classify_particles();
    build_sph_groups();
    estimate_group_density_and_pressure();
    update_group_motion(dt);
    scatter_group_states_to_particles();
    build_dsmc_groups();
    apply_grouped_dsmc(dt);
}

template <typename T>
T
HybridDsmcSphSolver<T>::grouping_length() const noexcept {
    return _grouping_length;
}

template <typename T>
int
HybridDsmcSphSolver<T>::sph_particle_threshold() const noexcept {
    return _sph_particle_threshold;
}

template <typename T>
SphKernelType
HybridDsmcSphSolver<T>::sph_kernel_type() const noexcept {
    return _sph_kernel.type;
}

template <typename T>
DsmcKernelType
HybridDsmcSphSolver<T>::dsmc_kernel_type() const noexcept {
    return _dsmc_kernel.type;
}

template <typename T>
bool
HybridDsmcSphSolver<T>::pairing_without_replacement() const noexcept {
    return _pairing_without_replacement;
}

template <typename T>
void
HybridDsmcSphSolver<T>::ensure_states() {
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

    if (!this->_universe->template has_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(number_of_cells);
    }

    if (!this->_universe->template has_state<atlas::universe::UniverseMaxSigmaGState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxSigmaGState<T>>(number_of_cells);
    }

    if (!this->_universe->template has_state<atlas::universe::UniverseCollisionCountState<int>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseCollisionCountState<int>>(number_of_cells);
    }
}

template <typename T>
bool
HybridDsmcSphSolver<T>::initialize_context() noexcept {
    if (!this->_universe || !this->_fluid || !this->_searcher || !(_grouping_length > T(0))) {
        reset_states();
        return false;
    }

    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr) {
        reset_states();
        return false;
    }

    ensure_states();
    this->_searcher->build();
    return true;
}

template <typename T>
bool
HybridDsmcSphSolver<T>::make_probe() noexcept {
    _probe = {};

    auto* position_state = this->_fluid->template state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* field_force_state        = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr
        || number_particle_state == nullptr || field_force_state == nullptr
        || max_relative_speed_state == nullptr || max_sigma_g_state == nullptr
        || collision_count_state == nullptr) {
        return false;
    }

    auto& positions          = position_state->data();
    auto& velocities         = velocity_state->data();
    auto& species            = species_state->data();
    auto& properties         = this->_fluid->particle_properties();
    auto& number_particle    = number_particle_state->data();
    auto& field_force        = field_force_state->data();
    auto& max_relative_speed = max_relative_speed_state->data();
    auto& max_sigma_g        = max_sigma_g_state->data();
    auto& collision_count    = collision_count_state->data();

    _probe.sph.position_ptr        = atlas::raw_pointer_cast(positions.data());
    _probe.sph.velocity_ptr        = atlas::raw_pointer_cast(velocities.data());
    _probe.sph.species_ptr         = atlas::raw_pointer_cast(species.data());
    _probe.sph.properties_ptr      = atlas::raw_pointer_cast(properties.data());
    _probe.sph.number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());
    _probe.sph.field_force_ptr     = atlas::raw_pointer_cast(field_force.data());
    _probe.sph.indices_ptr         = this->_searcher->indices();
    _probe.sph.cell_start_ptr      = this->_searcher->cell_start();
    _probe.sph.cell_end_ptr        = this->_searcher->cell_end();
    _probe.sph.lower_corner        = this->_searcher->lower_corner();
    _probe.sph.grid_size           = this->_searcher->grid_size();
    _probe.sph.inverse_cell_size   = this->_searcher->inverse_cell_size();
    _probe.sph.cell_size           = this->_searcher->cell_size();
    _probe.sph.particle_count      = static_cast<int>(this->_fluid->particle_count());
    _probe.sph.num_of_cells        = this->_universe->number_of_cells();
    _probe.sph.num_of_properties   = static_cast<int>(properties.size());
    _probe.sph.kernel              = _sph_kernel;

    _probe.dsmc.velocity_ptr            = atlas::raw_pointer_cast(velocities.data());
    _probe.dsmc.species_ptr             = atlas::raw_pointer_cast(species.data());
    _probe.dsmc.properties_ptr          = atlas::raw_pointer_cast(properties.data());
    _probe.dsmc.number_particle_ptr     = atlas::raw_pointer_cast(number_particle.data());
    _probe.dsmc.max_relative_speed_ptr  = atlas::raw_pointer_cast(max_relative_speed.data());
    _probe.dsmc.max_sigma_g_ptr         = atlas::raw_pointer_cast(max_sigma_g.data());
    _probe.dsmc.collision_count_ptr     = atlas::raw_pointer_cast(collision_count.data());
    _probe.dsmc.indices_ptr             = this->_searcher->indices();
    _probe.dsmc.cell_start_ptr          = this->_searcher->cell_start();
    _probe.dsmc.cell_end_ptr            = this->_searcher->cell_end();
    _probe.dsmc.particle_count          = static_cast<int>(this->_fluid->particle_count());
    _probe.dsmc.num_of_cells            = this->_universe->number_of_cells();
    _probe.dsmc.cell_volume             = this->_universe->cell_volume();
    _probe.dsmc.statistical_weight      = this->_fluid->statistical_weight();
    _probe.dsmc.kernel                  = _dsmc_kernel;
    _probe.dsmc.collision_seed          = _collision_seed;
    _probe.dsmc_piclas_scheduler = _pairing_without_replacement;

    _probe.grouping_length         = _grouping_length;
    _probe.sph_particle_threshold  = _sph_particle_threshold;
    _probe.collision_seed          = _collision_seed++;

    return true;
}

template <typename T>
bool
HybridDsmcSphSolver<T>::prepare_fields() {
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells   = this->_universe->number_of_cells();

    if (particle_count <= 0 || num_of_cells <= 0) {
        _sph_candidate.resize(0);
        _group_owner.resize(0);
        _group_member_count.resize(0);
        _dsmc_particle_count.resize(0);
        _dsmc_group_owner.resize(0);
        _dsmc_group_member_count.resize(0);
        _dsmc_collision_count.resize(0);
        _dsmc_max_relative_speed.resize(0);
        _dsmc_max_sigma_g.resize(0);
        _group_position.resize(0);
        _group_velocity.resize(0);
        _group_updated_velocity.resize(0);
        _group_mass.resize(0);
        _group_density.resize(0);
        _group_pressure.resize(0);
        _group_species.resize(0);
        reset_states();
        return false;
    }

    const auto size = static_cast<std::size_t>(particle_count);
    _sph_candidate.resize(size);
    _group_owner.resize(size);
    _group_member_count.resize(size);
    _dsmc_particle_count.resize(1);
    _dsmc_group_owner.resize(size);
    _dsmc_group_member_count.resize(size);
    _dsmc_collision_count.resize(size);
    _dsmc_max_relative_speed.resize(size);
    _dsmc_max_sigma_g.resize(size);
    _group_position.resize(size);
    _group_velocity.resize(size);
    _group_updated_velocity.resize(size);
    _group_mass.resize(size);
    _group_density.resize(size);
    _group_pressure.resize(size);
    _group_species.resize(size);

    atlas::parallel_fill<ExecutionPolicy::device>(_sph_candidate.begin(), _sph_candidate.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_group_owner.begin(), _group_owner.end(), -1);
    atlas::parallel_fill<ExecutionPolicy::device>(_group_member_count.begin(), _group_member_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_particle_count.begin(), _dsmc_particle_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_group_owner.begin(), _dsmc_group_owner.end(), -1);
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_group_member_count.begin(), _dsmc_group_member_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_collision_count.begin(), _dsmc_collision_count.end(), 0);
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_max_relative_speed.begin(), _dsmc_max_relative_speed.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_dsmc_max_sigma_g.begin(), _dsmc_max_sigma_g.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_position.begin(), _group_position.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_velocity.begin(), _group_velocity.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_updated_velocity.begin(), _group_updated_velocity.end(), Vector3<T>(T(0), T(0), T(0)));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_mass.begin(), _group_mass.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_density.begin(), _group_density.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_pressure.begin(), _group_pressure.end(), T(0));
    atlas::parallel_fill<ExecutionPolicy::device>(_group_species.begin(), _group_species.end(), std::size_t(0));

    reset_states();
    return true;
}

template <typename T>
void
HybridDsmcSphSolver<T>::reset_states() {
    if (!this->_universe) {
        return;
    }

    ensure_states();

    if (auto* state = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>()) {
        state->reset();
    }
    if (auto* state = this->_universe->template state<atlas::universe::UniverseFieldForceState<T>>()) {
        state->reset();
    }
    if (auto* state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>()) {
        state->reset();
    }
    if (auto* state = this->_universe->template state<atlas::universe::UniverseMaxSigmaGState<T>>()) {
        state->reset();
    }
    if (auto* state = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>()) {
        state->reset();
    }
}

template <typename T>
void
HybridDsmcSphSolver<T>::classify_particles() {
    const auto probe = _probe;
    auto* sph_candidate_ptr     = atlas::raw_pointer_cast(_sph_candidate.data());
    auto* group_owner_ptr       = atlas::raw_pointer_cast(_group_owner.data());
    auto* dsmc_particle_count   = atlas::raw_pointer_cast(_dsmc_particle_count.data());
    const T grouping_length2    = probe.grouping_length * probe.grouping_length;
    const int grouping_radius   = search_radius_for(probe.grouping_length, probe.sph.cell_size);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            const Vector3<T> pos = probe.sph.position_ptr[particle];
            const auto center    = particle_cell(pos, probe.sph.lower_corner, probe.sph.inverse_cell_size, probe.sph.grid_size);
            int count            = 0;

            for (int z = -grouping_radius; z <= grouping_radius; ++z) {
                for (int y = -grouping_radius; y <= grouping_radius; ++y) {
                    for (int x = -grouping_radius; x <= grouping_radius; ++x) {
                        const Vector3<int> cell = center + Vector3<int>(x, y, z);
                        if (!is_valid_neighbor_cell(cell, probe.sph.grid_size)) {
                            continue;
                        }

                        const int flat = (cell.z * probe.sph.grid_size.y + cell.y) * probe.sph.grid_size.x + cell.x;
                        const int begin = probe.sph.cell_start_ptr[flat];
                        const int end   = probe.sph.cell_end_ptr[flat];
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted = begin; sorted < end; ++sorted) {
                            const int other = probe.sph.indices_ptr[sorted];
                            const Vector3<T> delta = pos - probe.sph.position_ptr[other];
                            if (delta.length_squared() <= grouping_length2) {
                                ++count;
                            }
                        }
                    }
                }
            }

            if (count >= probe.sph_particle_threshold) {
                sph_candidate_ptr[particle] = count;
            } else {
                group_owner_ptr[particle] = -1;
                atlas::atomic_fetch_add_relaxed(dsmc_particle_count, 1);
            }
        });

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            if (sph_candidate_ptr[particle] == 0) {
                return;
            }

            const Vector3<T> pos = probe.sph.position_ptr[particle];
            const auto center    = particle_cell(pos, probe.sph.lower_corner, probe.sph.inverse_cell_size, probe.sph.grid_size);
            int owner            = particle;
            int owner_heat       = sph_candidate_ptr[particle];

            for (int z = -grouping_radius; z <= grouping_radius; ++z) {
                for (int y = -grouping_radius; y <= grouping_radius; ++y) {
                    for (int x = -grouping_radius; x <= grouping_radius; ++x) {
                        const Vector3<int> cell = center + Vector3<int>(x, y, z);
                        if (!is_valid_neighbor_cell(cell, probe.sph.grid_size)) {
                            continue;
                        }

                        const int flat = (cell.z * probe.sph.grid_size.y + cell.y) * probe.sph.grid_size.x + cell.x;
                        const int begin = probe.sph.cell_start_ptr[flat];
                        const int end   = probe.sph.cell_end_ptr[flat];
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted = begin; sorted < end; ++sorted) {
                            const int other = probe.sph.indices_ptr[sorted];
                            if (sph_candidate_ptr[other] == 0) {
                                continue;
                            }

                            const Vector3<T> delta = pos - probe.sph.position_ptr[other];
                            if (delta.length_squared() > grouping_length2) {
                                continue;
                            }

                            const int other_heat = sph_candidate_ptr[other];
                            if (other_heat > owner_heat || (other_heat == owner_heat && other < owner)) {
                                owner = other;
                                owner_heat = other_heat;
                            }
                        }
                    }
                }
            }

            group_owner_ptr[particle] = owner;
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::build_sph_groups() {
    const auto probe = _probe;
    const auto* group_owner_ptr = atlas::raw_pointer_cast(_group_owner.data());
    auto* group_member_count_ptr = atlas::raw_pointer_cast(_group_member_count.data());
    auto* group_position_ptr = atlas::raw_pointer_cast(_group_position.data());
    auto* group_velocity_ptr = atlas::raw_pointer_cast(_group_velocity.data());
    auto* group_updated_velocity_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());
    auto* group_mass_ptr = atlas::raw_pointer_cast(_group_mass.data());
    auto* group_species_ptr = atlas::raw_pointer_cast(_group_species.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            const int owner = group_owner_ptr[particle];
            if (owner < 0) {
                return;
            }

            const int previous_count = atlas::atomic_fetch_add_relaxed(&group_member_count_ptr[owner], 1);

            if (previous_count == 0) {
                group_species_ptr[owner] = probe.sph.species_ptr[particle];
            }
        });

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            if (group_member_count_ptr[particle] <= 0) {
                return;
            }

            Vector3<T> position_sum(T(0), T(0), T(0));
            Vector3<T> velocity_sum(T(0), T(0), T(0));
            T mass_sum = T(0);
            int count = 0;

            for (int member = 0; member < probe.sph.particle_count; ++member) {
                if (group_owner_ptr[member] != particle) {
                    continue;
                }

                position_sum += probe.sph.position_ptr[member];
                velocity_sum += probe.sph.velocity_ptr[member];
                ++count;

                const auto species = probe.sph.species_ptr[member];
                if (species < static_cast<std::size_t>(probe.sph.num_of_properties)) {
                    mass_sum += probe.sph.properties_ptr[species].mass;
                }
            }

            if (count <= 0) {
                return;
            }

            group_position_ptr[particle] = position_sum / static_cast<T>(count);
            group_velocity_ptr[particle] = velocity_sum / static_cast<T>(count);
            group_updated_velocity_ptr[particle] = group_velocity_ptr[particle];
            group_mass_ptr[particle] = mass_sum;
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::estimate_group_density_and_pressure() {
    const auto probe = _probe;
    const auto* group_member_count_ptr = atlas::raw_pointer_cast(_group_member_count.data());
    const auto* group_position_ptr = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_mass_ptr = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_species_ptr = atlas::raw_pointer_cast(_group_species.data());
    auto* group_density_ptr = atlas::raw_pointer_cast(_group_density.data());
    auto* group_pressure_ptr = atlas::raw_pointer_cast(_group_pressure.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int lhs) {
            if (group_member_count_ptr[lhs] <= 0) {
                return;
            }

            const auto species = group_species_ptr[lhs];
            if (species >= static_cast<std::size_t>(probe.sph.num_of_properties)) {
                return;
            }

            const auto& property = probe.sph.properties_ptr[species];
            const T smoothing_length = smoothing_length_for(property, probe.sph.cell_size);
            const T rest_density = rest_density_for(property);
            const T pressure_coeff = pressure_coefficient_for(property);
            const auto center = particle_cell(group_position_ptr[lhs], probe.sph.lower_corner, probe.sph.inverse_cell_size, probe.sph.grid_size);
            const int search_radius = search_radius_for(smoothing_length, probe.sph.cell_size);

            T density = T(0);

            for (int z = -search_radius; z <= search_radius; ++z) {
                for (int y = -search_radius; y <= search_radius; ++y) {
                    for (int x = -search_radius; x <= search_radius; ++x) {
                        const Vector3<int> cell = center + Vector3<int>(x, y, z);
                        if (!is_valid_neighbor_cell(cell, probe.sph.grid_size)) {
                            continue;
                        }

                        const int flat = (cell.z * probe.sph.grid_size.y + cell.y) * probe.sph.grid_size.x + cell.x;
                        const int begin = probe.sph.cell_start_ptr[flat];
                        const int end = probe.sph.cell_end_ptr[flat];
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted = begin; sorted < end; ++sorted) {
                            const int rhs = probe.sph.indices_ptr[sorted];
                            if (group_member_count_ptr[rhs] <= 0) {
                                continue;
                            }

                            const Vector3<T> delta = group_position_ptr[lhs] - group_position_ptr[rhs];
                            const T radius = delta.length();
                            density += group_mass_ptr[rhs] * probe.sph.kernel.density_weight(radius, smoothing_length);
                        }
                    }
                }
            }

            if (!(density > T(0))) {
                density = rest_density;
            }

            group_density_ptr[lhs] = density;
            group_pressure_ptr[lhs] = pressure_coeff * (density - rest_density);
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::update_group_motion(const T dt) {
    const auto probe = _probe;
    const auto* group_member_count_ptr = atlas::raw_pointer_cast(_group_member_count.data());
    const auto* group_position_ptr = atlas::raw_pointer_cast(_group_position.data());
    const auto* group_velocity_ptr = atlas::raw_pointer_cast(_group_velocity.data());
    const auto* group_mass_ptr = atlas::raw_pointer_cast(_group_mass.data());
    const auto* group_density_ptr = atlas::raw_pointer_cast(_group_density.data());
    const auto* group_pressure_ptr = atlas::raw_pointer_cast(_group_pressure.data());
    const auto* group_species_ptr = atlas::raw_pointer_cast(_group_species.data());
    auto* group_updated_velocity_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int lhs) {
            if (group_member_count_ptr[lhs] <= 0) {
                return;
            }

            const auto species = group_species_ptr[lhs];
            if (species >= static_cast<std::size_t>(probe.sph.num_of_properties)) {
                return;
            }

            const auto& property = probe.sph.properties_ptr[species];
            const T smoothing_length = smoothing_length_for(property, probe.sph.cell_size);
            const T viscosity = property.dynamic_viscosity.value_or(T(0));
            const auto center = particle_cell(group_position_ptr[lhs], probe.sph.lower_corner, probe.sph.inverse_cell_size, probe.sph.grid_size);
            const int search_radius = search_radius_for(smoothing_length, probe.sph.cell_size);
            Vector3<T> acceleration(T(0), T(0), T(0));

            for (int z = -search_radius; z <= search_radius; ++z) {
                for (int y = -search_radius; y <= search_radius; ++y) {
                    for (int x = -search_radius; x <= search_radius; ++x) {
                        const Vector3<int> cell = center + Vector3<int>(x, y, z);
                        if (!is_valid_neighbor_cell(cell, probe.sph.grid_size)) {
                            continue;
                        }

                        const int flat = (cell.z * probe.sph.grid_size.y + cell.y) * probe.sph.grid_size.x + cell.x;
                        const int begin = probe.sph.cell_start_ptr[flat];
                        const int end = probe.sph.cell_end_ptr[flat];
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted = begin; sorted < end; ++sorted) {
                            const int rhs = probe.sph.indices_ptr[sorted];
                            if (rhs == lhs || group_member_count_ptr[rhs] <= 0 || !(group_density_ptr[rhs] > T(0))) {
                                continue;
                            }

                            const Vector3<T> delta = group_position_ptr[lhs] - group_position_ptr[rhs];
                            const T radius = delta.length();
                            if (!(radius > T(0)) || radius > smoothing_length) {
                                continue;
                            }

                            const Vector3<T> grad = probe.sph.kernel.pressure_gradient(delta, radius, smoothing_length);
                            const T pressure_term = (group_pressure_ptr[lhs] + group_pressure_ptr[rhs])
                                / (static_cast<T>(2) * group_density_ptr[rhs]);
                            acceleration -= grad * (group_mass_ptr[rhs] * pressure_term);

                            if (viscosity > T(0)) {
                                const T laplacian = probe.sph.kernel.viscosity_laplacian(radius, smoothing_length);
                                acceleration += (group_velocity_ptr[rhs] - group_velocity_ptr[lhs])
                                    * (viscosity * group_mass_ptr[rhs] * laplacian / group_density_ptr[rhs]);
                            }
                        }
                    }
                }
            }

            group_updated_velocity_ptr[lhs] = group_velocity_ptr[lhs] + acceleration * dt;
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::scatter_group_states_to_particles() {
    const auto probe = _probe;
    const auto* group_owner_ptr = atlas::raw_pointer_cast(_group_owner.data());
    const auto* group_updated_velocity_ptr = atlas::raw_pointer_cast(_group_updated_velocity.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            const int owner = group_owner_ptr[particle];
            if (owner >= 0) {
                probe.sph.velocity_ptr[particle] = group_updated_velocity_ptr[owner];
            }
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::build_dsmc_groups() {
    const auto probe = _probe;
    const auto* group_owner_ptr = atlas::raw_pointer_cast(_group_owner.data());
    auto* dsmc_group_owner_ptr = atlas::raw_pointer_cast(_dsmc_group_owner.data());
    auto* dsmc_group_member_count_ptr = atlas::raw_pointer_cast(_dsmc_group_member_count.data());
    const int dsmc_count = _dsmc_particle_count.empty() ? 0 : _dsmc_particle_count[0];

    if (dsmc_count <= 0) {
        return;
    }

    const T grouping_length2 = probe.grouping_length * probe.grouping_length;
    const int grouping_radius = search_radius_for(probe.grouping_length, probe.sph.cell_size);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            if (group_owner_ptr[particle] >= 0) {
                return;
            }

            const Vector3<T> pos = probe.sph.position_ptr[particle];
            const auto center = particle_cell(pos, probe.sph.lower_corner, probe.sph.inverse_cell_size, probe.sph.grid_size);
            int owner = particle;

            for (int z = -grouping_radius; z <= grouping_radius; ++z) {
                for (int y = -grouping_radius; y <= grouping_radius; ++y) {
                    for (int x = -grouping_radius; x <= grouping_radius; ++x) {
                        const Vector3<int> cell = center + Vector3<int>(x, y, z);
                        if (!is_valid_neighbor_cell(cell, probe.sph.grid_size)) {
                            continue;
                        }

                        const int flat = (cell.z * probe.sph.grid_size.y + cell.y) * probe.sph.grid_size.x + cell.x;
                        const int begin = probe.sph.cell_start_ptr[flat];
                        const int end = probe.sph.cell_end_ptr[flat];
                        if (begin < 0 || end <= begin) {
                            continue;
                        }

                        for (int sorted = begin; sorted < end; ++sorted) {
                            const int other = probe.sph.indices_ptr[sorted];
                            if (group_owner_ptr[other] >= 0) {
                                continue;
                            }

                            const Vector3<T> delta = pos - probe.sph.position_ptr[other];
                            if (delta.length_squared() <= grouping_length2 && other < owner) {
                                owner = other;
                            }
                        }
                    }
                }
            }

            dsmc_group_owner_ptr[particle] = owner;
        });

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.sph.particle_count,
        [=] ATLAS_DEVICE(const int particle) {
            const int owner = dsmc_group_owner_ptr[particle];
            if (owner >= 0) {
                atlas::atomic_fetch_add_relaxed(&dsmc_group_member_count_ptr[owner], 1);
            }
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::apply_grouped_dsmc(const T dt) {
    measure_grouped_dsmc_statistics(dt);

    if (_probe.dsmc_piclas_scheduler) {
        apply_grouped_dsmc_collisions_without_replacement();
    } else {
        apply_random_grouped_dsmc_collisions();
    }
}

template <typename T>
void
HybridDsmcSphSolver<T>::measure_grouped_dsmc_statistics(const T dt) {
    const auto probe = _probe;
    const auto* dsmc_group_owner_ptr = atlas::raw_pointer_cast(_dsmc_group_owner.data());
    auto* dsmc_group_member_count_ptr = atlas::raw_pointer_cast(_dsmc_group_member_count.data());
    auto* dsmc_collision_count_ptr = atlas::raw_pointer_cast(_dsmc_collision_count.data());
    auto* max_relative_speed_ptr = atlas::raw_pointer_cast(_dsmc_max_relative_speed.data());
    auto* max_sigma_g_ptr = atlas::raw_pointer_cast(_dsmc_max_sigma_g.data());
    const int dsmc_count = _dsmc_particle_count.empty() ? 0 : _dsmc_particle_count[0];

    if (dsmc_count < 2 || probe.dsmc.properties_ptr == nullptr) {
        return;
    }

    const T group_volume = probe.grouping_length * probe.grouping_length * probe.grouping_length;
    if (!(group_volume > T(0))) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.dsmc.particle_count,
        [=] ATLAS_DEVICE(const int owner) {
            const int count = dsmc_group_member_count_ptr[owner];
            if (count < 2) {
                return;
            }

            T max_relative_squared = T(0);
            T max_sigma_g = T(0);

            for (int lhs = 0; lhs < probe.dsmc.particle_count; ++lhs) {
                if (dsmc_group_owner_ptr[lhs] != owner) {
                    continue;
                }

                for (int rhs = lhs + 1; rhs < probe.dsmc.particle_count; ++rhs) {
                    if (dsmc_group_owner_ptr[rhs] != owner) {
                        continue;
                    }

                    const auto species_i = probe.dsmc.species_ptr[lhs];
                    const auto species_j = probe.dsmc.species_ptr[rhs];
                    const T relative_speed_squared = (probe.dsmc.velocity_ptr[lhs] - probe.dsmc.velocity_ptr[rhs]).length_squared();
                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = probe.dsmc.kernel.sigma_g(probe.dsmc.properties_ptr,
                        species_i,
                        species_j,
                        relative_speed_squared);
                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            max_relative_speed_ptr[owner] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);
            max_sigma_g_ptr[owner] = max_sigma_g;

            if (!(max_sigma_g > T(0))) {
                return;
            }

            const T pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T expected_collisions = pair_count * max_sigma_g * probe.dsmc.statistical_weight * dt / group_volume;
            if (expected_collisions >= static_cast<T>(std::numeric_limits<int>::max())) {
                dsmc_collision_count_ptr[owner] = probe.dsmc_piclas_scheduler
                    ? count / 2
                    : std::numeric_limits<int>::max();
                return;
            }

            int collision_count = static_cast<int>(std::floor(expected_collisions));
            const T remainder = expected_collisions - static_cast<T>(collision_count);
            if (remainder > T(0)
                && atlas::sampling::sample_hashed_unit_interval<T>(owner, probe.collision_seed) < remainder) {
                ++collision_count;
            }

            if (probe.dsmc_piclas_scheduler) {
                const int max_unique_pairs = count / 2;
                if (collision_count > max_unique_pairs) {
                    collision_count = max_unique_pairs;
                }
            }

            dsmc_collision_count_ptr[owner] = collision_count > 0 ? collision_count : 0;
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::apply_random_grouped_dsmc_collisions() {
    const auto probe = _probe;
    const auto* dsmc_group_owner_ptr = atlas::raw_pointer_cast(_dsmc_group_owner.data());
    auto* dsmc_group_member_count_ptr = atlas::raw_pointer_cast(_dsmc_group_member_count.data());
    auto* dsmc_collision_count_ptr = atlas::raw_pointer_cast(_dsmc_collision_count.data());
    auto* max_sigma_g_ptr = atlas::raw_pointer_cast(_dsmc_max_sigma_g.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.dsmc.particle_count,
        [=] ATLAS_DEVICE(const int owner) {
            const int count = dsmc_group_member_count_ptr[owner];
            const int collision_count = dsmc_collision_count_ptr[owner];
            const T max_sigma_g = max_sigma_g_ptr[owner];

            if (count < 2 || collision_count <= 0 || !(max_sigma_g > T(0))) {
                return;
            }

            for (int collision = 0; collision < collision_count; ++collision) {
                const auto stream = static_cast<std::uint64_t>(owner) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(collision);

                int lhs_local = -1;
                int rhs_local = -1;

                lhs_local = atlas::sampling::sample_hashed_index(
                    owner,
                    count,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);
                rhs_local = atlas::sampling::sample_hashed_index(
                    owner,
                    count - 1,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);
                if (rhs_local >= lhs_local) {
                    ++rhs_local;
                }

                int lhs_particle = -1;
                int rhs_particle = -1;
                int local_index = 0;

                for (int particle = 0; particle < probe.dsmc.particle_count; ++particle) {
                    if (dsmc_group_owner_ptr[particle] != owner) {
                        continue;
                    }

                    if (local_index == lhs_local) {
                        lhs_particle = particle;
                    }
                    if (local_index == rhs_local) {
                        rhs_particle = particle;
                    }
                    ++local_index;
                }

                if (lhs_particle < 0 || rhs_particle < 0) {
                    continue;
                }

                const auto species_i = probe.dsmc.species_ptr[lhs_particle];
                const auto species_j = probe.dsmc.species_ptr[rhs_particle];
                Vector3<T> lhs_velocity = probe.dsmc.velocity_ptr[lhs_particle];
                Vector3<T> rhs_velocity = probe.dsmc.velocity_ptr[rhs_particle];
                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T sigma_g = probe.dsmc.kernel.sigma_g(probe.dsmc.properties_ptr,
                    species_i,
                    species_j,
                    relative_speed_squared);

                T accept_probability = T(0);
                if (sigma_g > T(0)) {
                    accept_probability = sigma_g / max_sigma_g;
                    if (accept_probability > T(1)) {
                        accept_probability = T(1);
                    }
                }

                const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                    owner,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

                if (!(accept_sample < accept_probability)) {
                    continue;
                }

                probe.dsmc.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.dsmc.properties_ptr[species_i],
                    probe.dsmc.properties_ptr[species_j]);

                probe.dsmc.velocity_ptr[lhs_particle] = lhs_velocity;
                probe.dsmc.velocity_ptr[rhs_particle] = rhs_velocity;
            }
        });
}

template <typename T>
void
HybridDsmcSphSolver<T>::apply_grouped_dsmc_collisions_without_replacement() {
    const auto probe = _probe;
    const auto* dsmc_group_owner_ptr = atlas::raw_pointer_cast(_dsmc_group_owner.data());
    auto* dsmc_group_member_count_ptr = atlas::raw_pointer_cast(_dsmc_group_member_count.data());
    auto* dsmc_collision_count_ptr = atlas::raw_pointer_cast(_dsmc_collision_count.data());
    auto* max_sigma_g_ptr = atlas::raw_pointer_cast(_dsmc_max_sigma_g.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.dsmc.particle_count,
        [=] ATLAS_DEVICE(const int owner) {
            const int count = dsmc_group_member_count_ptr[owner];
            const int collision_count = dsmc_collision_count_ptr[owner];
            const T max_sigma_g = max_sigma_g_ptr[owner];

            if (count < 2 || collision_count <= 0 || !(max_sigma_g > T(0))) {
                return;
            }

            for (int collision = 0; collision < collision_count; ++collision) {
                const auto stream = static_cast<std::uint64_t>(owner) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(collision);

                int lhs_local = -1;
                int rhs_local = -1;
                atlas::scheduler::DsmcPiclasScheduler<T>::select_pair_offsets(
                    lhs_local,
                    rhs_local,
                    collision,
                    count,
                    owner,
                    probe.collision_seed + static_cast<std::uint64_t>(owner));

                int lhs_particle = -1;
                int rhs_particle = -1;
                int local_index = 0;

                for (int particle = 0; particle < probe.dsmc.particle_count; ++particle) {
                    if (dsmc_group_owner_ptr[particle] != owner) {
                        continue;
                    }

                    if (local_index == lhs_local) {
                        lhs_particle = particle;
                    }
                    if (local_index == rhs_local) {
                        rhs_particle = particle;
                    }
                    ++local_index;
                }

                if (lhs_particle < 0 || rhs_particle < 0) {
                    continue;
                }

                const auto species_i = probe.dsmc.species_ptr[lhs_particle];
                const auto species_j = probe.dsmc.species_ptr[rhs_particle];
                Vector3<T> lhs_velocity = probe.dsmc.velocity_ptr[lhs_particle];
                Vector3<T> rhs_velocity = probe.dsmc.velocity_ptr[rhs_particle];
                const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
                const T sigma_g = probe.dsmc.kernel.sigma_g(probe.dsmc.properties_ptr,
                    species_i,
                    species_j,
                    relative_speed_squared);

                T accept_probability = T(0);
                if (sigma_g > T(0)) {
                    accept_probability = sigma_g / max_sigma_g;
                    if (accept_probability > T(1)) {
                        accept_probability = T(1);
                    }
                }

                const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                    owner,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

                if (!(accept_sample < accept_probability)) {
                    continue;
                }

                probe.dsmc.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.dsmc.properties_ptr[species_i],
                    probe.dsmc.properties_ptr[species_j]);

                probe.dsmc.velocity_ptr[lhs_particle] = lhs_velocity;
                probe.dsmc.velocity_ptr[rhs_particle] = rhs_velocity;
            }
        });
}

template <typename T>
T
HybridDsmcSphSolver<T>::smoothing_length_for(const MaterialProperties<T>& property,
                                             const T cell_size) noexcept {
    if (property.smoothing_length.has_value() && *property.smoothing_length > T(0)) {
        return *property.smoothing_length;
    }
    return cell_size;
}

template <typename T>
T
HybridDsmcSphSolver<T>::rest_density_for(const MaterialProperties<T>& property) noexcept {
    if (property.rest_density.has_value() && *property.rest_density > T(0)) {
        return *property.rest_density;
    }
    return T(1);
}

template <typename T>
T
HybridDsmcSphSolver<T>::pressure_coefficient_for(const MaterialProperties<T>& property) noexcept {
    return property.pressure_coefficient.value_or(T(0));
}

template <typename T>
int
HybridDsmcSphSolver<T>::search_radius_for(const T length,
                                          const T cell_size) noexcept {
    return static_cast<int>(std::ceil(length / cell_size));
}

template <typename T>
Vector3<int>
HybridDsmcSphSolver<T>::particle_cell(const Vector3<T>& position,
                                      const Vector3<T>& lower_corner,
                                      const T inverse_cell_size,
                                      const Vector3<int>& grid_size) noexcept {
    auto cell = atlas::math::floor((position - lower_corner) * inverse_cell_size).template cast_to<int>();
    return atlas::math::clamp(cell, Vector3<int>(0, 0, 0), grid_size - Vector3<int>(1, 1, 1));
}

template <typename T>
bool
HybridDsmcSphSolver<T>::is_valid_neighbor_cell(const Vector3<int>& cell,
                                               const Vector3<int>& grid_size) noexcept {
    return cell.x >= 0 && cell.y >= 0 && cell.z >= 0
        && cell.x < grid_size.x && cell.y < grid_size.y && cell.z < grid_size.z;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_grouping_length(const T grouping_length) noexcept {
    _grouping_length = grouping_length;
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_sph_particle_threshold(const int threshold) noexcept {
    _sph_particle_threshold = threshold;
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_sph_kernel_type(const SphKernelType kernel_type) noexcept {
    _sph_kernel = SphKernel<T>(kernel_type);
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_dsmc_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _dsmc_kernel = DsmcKernel<T>(kernel_type);
    return *this;
}

template <typename T>
typename HybridDsmcSphSolver<T>::Builder&
HybridDsmcSphSolver<T>::Builder::with_pairing_without_replacement(const bool enabled) noexcept {
    _pairing_without_replacement = enabled;
    return *this;
}

template <typename T>
void
HybridDsmcSphSolver<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("HybridDsmcSphSolver::Builder: universe must not be null.");
    }
    if (!_fluid) {
        throw std::runtime_error("HybridDsmcSphSolver::Builder: fluid must not be null.");
    }
    if (!_searcher) {
        throw std::runtime_error("HybridDsmcSphSolver::Builder: searcher must not be null.");
    }
    if (!(_grouping_length > T(0))) {
        throw std::runtime_error("HybridDsmcSphSolver::Builder: grouping_length must be positive.");
    }
    if (_sph_particle_threshold <= 0) {
        throw std::runtime_error("HybridDsmcSphSolver::Builder: sph_particle_threshold must be positive.");
    }
}

template <typename T>
HybridDsmcSphSolver<T>
HybridDsmcSphSolver<T>::Builder::build() const {
    validate();
    return HybridDsmcSphSolver<T>(
        _universe,
        _fluid,
        _searcher,
        _grouping_length,
        _sph_particle_threshold,
        _sph_kernel.type,
        _dsmc_kernel.type,
        _pairing_without_replacement);
}

template <typename T>
atlas::host_shared_ptr<HybridDsmcSphSolver<T>>
HybridDsmcSphSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<HybridDsmcSphSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _grouping_length,
        _sph_particle_threshold,
        _sph_kernel.type,
        _dsmc_kernel.type,
        _pairing_without_replacement);
}

} // namespace atlas::system
