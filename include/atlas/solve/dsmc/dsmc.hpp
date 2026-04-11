#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/scan/exclusive_scan.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Dsmc<T>::Dsmc(FluidHostPtr<T> fluid,
              DsmcOperator<T> op,
              DsmcSolveMode mode)
    : _fluid(std::move(fluid))
    , _operator(std::move(op))
    , _solve_mode(mode) {
    rebuild_pair_tables();
}

template <typename T>
typename Dsmc<T>::Builder
Dsmc<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Dsmc<T>::solve(DomainDeviceProbe<T> domain,
               SpatialHashingProbe<T> searcher,
               FluidDeviceProbe<T> particle,
               CodecDeviceProbe<T>) {
    if (particle.particle_count <= 0 || particle.particle_property == nullptr || particle.vel == nullptr || particle.species == nullptr) {
        return;
    }

    apply_field_force(domain, particle);

    switch (_solve_mode) {
    case DsmcSolveMode::ntc:
        solve_ntc(domain, searcher, particle);
        return;
    case DsmcSolveMode::disjoint_pair:
    default:
        solve_disjoint_pair(domain, searcher, particle);
        return;
    }
}

template <typename T>
void
Dsmc<T>::set_operator(DsmcOperator<T> op) noexcept {
    _operator = std::move(op);
}

template <typename T>
void
Dsmc<T>::set_fluid(FluidHostPtr<T> fluid) {
    _fluid = std::move(fluid);
    rebuild_pair_tables();
}

template <typename T>
void
Dsmc<T>::set_solve_mode(DsmcSolveMode mode) noexcept {
    _solve_mode = mode;
}

template <typename T>
const FluidHostPtr<T>&
Dsmc<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
const DsmcOperator<T>&
Dsmc<T>::collision_operator() const noexcept {
    return _operator;
}

template <typename T>
DsmcOperator<T>&
Dsmc<T>::collision_operator() noexcept {
    return _operator;
}

template <typename T>
DsmcSolveMode
Dsmc<T>::solve_mode() const noexcept {
    return _solve_mode;
}

template <typename T>
void
Dsmc<T>::apply_field_force(DomainDeviceProbe<T> domain,
                           FluidDeviceProbe<T> particle) const {
    if (domain.field_force == nullptr || particle.pos == nullptr || particle.vel == nullptr || particle.particle_count <= 0) {
        return;
    }

    auto* positions         = particle.pos;
    auto* velocities        = particle.vel;
    const auto* field_force = domain.field_force;
    const Vector3<T> lower_corner = domain.lower_corner;
    const Vector3<int> grid_size  = domain.grid_size;
    const T inv_h                 = domain.inv_h;
    const Vector3<int> lo { 0, 0, 0 };
    const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle.particle_count,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<T> rel = (positions[i] - lower_corner) * inv_h;
            Vector3<int> ijk     = atlas::math::floor(rel).template cast_to<int>();
            ijk                  = atlas::math::clamp(ijk, lo, hi);

            const int cell = ijk.x + ijk.y * grid_size.x + ijk.z * grid_size.x * grid_size.y;
            velocities[i] += field_force[cell];
        });
}

template <typename T>
void
Dsmc<T>::solve_disjoint_pair(DomainDeviceProbe<T> domain,
                             SpatialHashingProbe<T> searcher,
                             FluidDeviceProbe<T> particle) noexcept {
    const auto collision_operator               = _operator;
    auto* velocities                            = particle.vel;
    const auto* species                         = particle.species;
    const auto* properties                      = particle.particle_property;
    const auto* cell_start                      = searcher.cell_start;
    const auto* cell_end                        = searcher.cell_end;
    const auto* indices                         = searcher.indices;
    const auto* field_temperature               = domain.field_temperature;
    const auto* effective_collision_diameters   = atlas::raw_pointer_cast(_effective_collision_diameters.data());
    const auto* effective_viscosity_indices     = atlas::raw_pointer_cast(_effective_viscosity_indices.data());
    const auto* effective_scattering_parameters = atlas::raw_pointer_cast(_effective_scattering_parameters.data());
    const auto* reduced_masses                  = atlas::raw_pointer_cast(_reduced_masses.data());
    const auto pair_indexer                     = _pair_indexer;
    const int num_cells                         = domain.num_of_cells;
    const int species_count                     = _species_count;

    if (num_cells <= 0 || cell_start == nullptr || cell_end == nullptr || indices == nullptr || species_count <= 0) {
        return;
    }

    _cell_pair_counts.resize(static_cast<std::size_t>(num_cells));
    _pair_offsets.resize(static_cast<std::size_t>(num_cells + 1));

    auto* cell_pair_counts_ptr = atlas::raw_pointer_cast(_cell_pair_counts.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start[cell];
            if (begin < 0) {
                cell_pair_counts_ptr[cell] = 0;
                return;
            }

            const int end              = cell_end[cell];
            const int pair_count       = (end - begin) / 2;
            cell_pair_counts_ptr[cell] = (pair_count > 0) ? pair_count : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        _cell_pair_counts.begin(),
        _cell_pair_counts.end(),
        _pair_offsets.begin(),
        0);

    int last_pair_offset = 0;
    int last_pair_count  = 0;
    atlas::copy_device_to_host(atlas::raw_pointer_cast(_pair_offsets.data()) + (num_cells - 1), &last_pair_offset, 1);
    atlas::copy_device_to_host(atlas::raw_pointer_cast(_cell_pair_counts.data()) + (num_cells - 1), &last_pair_count, 1);

    const int total_pairs = last_pair_offset + last_pair_count;
    _pair_offsets[static_cast<std::size_t>(num_cells)] = total_pairs;

    if (total_pairs <= 0) {
        return;
    }

    const auto* pair_offsets_ptr = atlas::raw_pointer_cast(_pair_offsets.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        total_pairs,
        [=] ATLAS_DEVICE(const int work_index) {
            int left  = 0;
            int right = num_cells;

            while (left + 1 < right) {
                const int mid = left + (right - left) / 2;
                if (pair_offsets_ptr[mid] <= work_index) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            const int cell        = left;
            const int pair_offset = work_index - pair_offsets_ptr[cell];
            const int pair_begin  = cell_start[cell] + 2 * pair_offset;
            const int p           = indices[pair_begin];
            const int q           = indices[pair_begin + 1];

            if (p < 0 || q < 0 || p >= particle.particle_count || q >= particle.particle_count) {
                return;
            }

            const size_t species_p        = species[p];
            const size_t species_q        = species[q];
            const int material_pair_index = pair_indexer.pair_index(
                static_cast<int>(species_p),
                static_cast<int>(species_q),
                species_count);

            const Vector3<T> vp         = velocities[p];
            const Vector3<T> vq         = velocities[q];
            const Vector3<T> g          = vq - vp;
            const T local_temperature   = field_temperature != nullptr ? field_temperature[cell] : T(0);
            const std::uint64_t pair_id = static_cast<std::uint64_t>(cell) * static_cast<std::uint64_t>(particle.buffer_size)
                + static_cast<std::uint64_t>(pair_begin);

            if (!collision_operator.should_collide(
                    effective_collision_diameters[material_pair_index],
                    reduced_masses[material_pair_index],
                    effective_viscosity_indices[material_pair_index],
                    effective_scattering_parameters[material_pair_index],
                    local_temperature,
                    g,
                    pair_id)) {
                return;
            }

            const T mass_p     = properties[species_p].mass;
            const T mass_q     = properties[species_q].mass;
            const T total_mass = mass_p + mass_q;
            if (!(mass_p > T(0)) || !(mass_q > T(0)) || !(total_mass > T(0))) {
                return;
            }

            const Vector3<T> center_of_mass_velocity = (vp * mass_p + vq * mass_q) / total_mass;
            const Vector3<T> scattered_g             = collision_operator.scatter_relative_velocity(
                effective_scattering_parameters[material_pair_index],
                g,
                pair_id);

            velocities[p] = center_of_mass_velocity - scattered_g * (mass_q / total_mass);
            velocities[q] = center_of_mass_velocity + scattered_g * (mass_p / total_mass);
        });
}

template <typename T>
void
Dsmc<T>::solve_ntc(DomainDeviceProbe<T> domain,
                   SpatialHashingProbe<T> searcher,
                   FluidDeviceProbe<T> particle) noexcept {
    const auto collision_operator               = _operator;
    auto* velocities                            = particle.vel;
    const auto* species                         = particle.species;
    const auto* properties                      = particle.particle_property;
    const auto* cell_start                      = searcher.cell_start;
    const auto* cell_end                        = searcher.cell_end;
    const auto* indices                         = searcher.indices;
    const auto* field_temperature               = domain.field_temperature;
    const auto* effective_collision_diameters   = atlas::raw_pointer_cast(_effective_collision_diameters.data());
    const auto* effective_viscosity_indices     = atlas::raw_pointer_cast(_effective_viscosity_indices.data());
    const auto* effective_scattering_parameters = atlas::raw_pointer_cast(_effective_scattering_parameters.data());
    const auto* reduced_masses                  = atlas::raw_pointer_cast(_reduced_masses.data());
    const auto pair_indexer                     = _pair_indexer;
    const int num_cells                         = domain.num_of_cells;
    const int species_count                     = _species_count;
    unsigned int base_seed                      = 0u;

    switch (collision_operator.type) {
    case DsmcModelType::hard_sphere:
        base_seed = collision_operator.hard_sphere.seed;
        break;
    case DsmcModelType::vhs:
        base_seed = collision_operator.vhs.seed;
        break;
    case DsmcModelType::vss:
        base_seed = collision_operator.vss.seed;
        break;
    default:
        break;
    }

    if (num_cells <= 0 || cell_start == nullptr || cell_end == nullptr || indices == nullptr || species_count <= 0) {
        return;
    }

    _cell_trial_counts.resize(static_cast<std::size_t>(num_cells));
    _trial_offsets.resize(static_cast<std::size_t>(num_cells + 1));

    auto* cell_trial_counts_ptr = atlas::raw_pointer_cast(_cell_trial_counts.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start[cell];
            if (begin < 0) {
                cell_trial_counts_ptr[cell] = 0;
                return;
            }

            const int end = cell_end[cell];
            const int count = end - begin;
            cell_trial_counts_ptr[cell] = count >= 2 ? (count * (count - 1)) / 2 : 0;
        });

    atlas::exclusive_scan<ExecutionPolicy::device>(
        _cell_trial_counts.begin(),
        _cell_trial_counts.end(),
        _trial_offsets.begin(),
        0);

    int last_trial_offset = 0;
    int last_trial_count  = 0;
    atlas::copy_device_to_host(atlas::raw_pointer_cast(_trial_offsets.data()) + (num_cells - 1), &last_trial_offset, 1);
    atlas::copy_device_to_host(atlas::raw_pointer_cast(_cell_trial_counts.data()) + (num_cells - 1), &last_trial_count, 1);

    const int total_trials = last_trial_offset + last_trial_count;
    _trial_offsets[static_cast<std::size_t>(num_cells)] = total_trials;

    if (total_trials <= 0) {
        return;
    }

    const auto* trial_offsets_ptr = atlas::raw_pointer_cast(_trial_offsets.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        total_trials,
        [=] ATLAS_DEVICE(const int work_index) {
            int left  = 0;
            int right = num_cells;

            while (left + 1 < right) {
                const int mid = left + (right - left) / 2;
                if (trial_offsets_ptr[mid] <= work_index) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            const int cell        = left;
            const int local_trial = work_index - trial_offsets_ptr[cell];
            const int begin       = cell_start[cell];
            const int end         = cell_end[cell];
            const int count       = end - begin;

            if (begin < 0 || count < 2) {
                return;
            }

            auto engine = atlas::default_random_engine<T>(
                static_cast<unsigned int>(base_seed + 1664525u * static_cast<unsigned int>(work_index) + 1013904223u));
            atlas::uniform_real_distribution<T> dist(T(0), T(1));

            const int local_p = std::min(static_cast<int>(dist(engine) * static_cast<T>(count)), count - 1);
            int local_q = std::min(static_cast<int>(dist(engine) * static_cast<T>(count - 1)), count - 2);
            if (local_q >= local_p) {
                ++local_q;
            }

            const int p = indices[begin + local_p];
            const int q = indices[begin + local_q];
            if (p < 0 || q < 0 || p >= particle.particle_count || q >= particle.particle_count || p == q) {
                return;
            }

            const size_t species_p        = species[p];
            const size_t species_q        = species[q];
            const int material_pair_index = pair_indexer.pair_index(
                static_cast<int>(species_p),
                static_cast<int>(species_q),
                species_count);

            const Vector3<T> vp         = velocities[p];
            const Vector3<T> vq         = velocities[q];
            const Vector3<T> g          = vq - vp;
            const T local_temperature   = field_temperature != nullptr ? field_temperature[cell] : T(0);
            const std::uint64_t pair_id = static_cast<std::uint64_t>(cell) * static_cast<std::uint64_t>(particle.buffer_size)
                + static_cast<std::uint64_t>(local_trial);

            if (!collision_operator.should_collide(
                    effective_collision_diameters[material_pair_index],
                    reduced_masses[material_pair_index],
                    effective_viscosity_indices[material_pair_index],
                    effective_scattering_parameters[material_pair_index],
                    local_temperature,
                    g,
                    pair_id)) {
                return;
            }

            const T mass_p     = properties[species_p].mass;
            const T mass_q     = properties[species_q].mass;
            const T total_mass = mass_p + mass_q;
            if (!(mass_p > T(0)) || !(mass_q > T(0)) || !(total_mass > T(0))) {
                return;
            }

            const Vector3<T> center_of_mass_velocity = (vp * mass_p + vq * mass_q) / total_mass;
            const Vector3<T> scattered_g             = collision_operator.scatter_relative_velocity(
                effective_scattering_parameters[material_pair_index],
                g,
                pair_id);

            velocities[p] = center_of_mass_velocity - scattered_g * (mass_q / total_mass);
            velocities[q] = center_of_mass_velocity + scattered_g * (mass_p / total_mass);
        });
}

template <typename T>
void
Dsmc<T>::rebuild_pair_tables() {
    if (!_fluid) {
        throw std::runtime_error("Dsmc: fluid must not be null.");
    }

    _species_count = _fluid->size();
    if (_species_count <= 0) {
        _effective_collision_diameters.clear();
        _effective_viscosity_indices.clear();
        _effective_scattering_parameters.clear();
        _reduced_masses.clear();
        return;
    }

    const std::size_t pair_count = static_cast<std::size_t>(_species_count) * static_cast<std::size_t>(_species_count + 1) / 2;

    HostBuffer<MatrialProperties<T>> particles_host(static_cast<std::size_t>(_species_count));
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(_fluid->particles().data()),
        particles_host.data(),
        particles_host.size());

    HostBuffer<T> effective_collision_diameters_host(pair_count, T(0));
    HostBuffer<T> effective_viscosity_indices_host(pair_count, T(1));
    HostBuffer<T> effective_scattering_parameters_host(pair_count, T(1));
    HostBuffer<T> reduced_masses_host(pair_count, T(0));

    for (int s = 0; s < _species_count; ++s) {
        for (int r = s; r < _species_count; ++r) {
            const int pair_index = _pair_indexer.pair_index(s, r, _species_count);
            const auto& a        = particles_host[static_cast<std::size_t>(s)];
            const auto& b        = particles_host[static_cast<std::size_t>(r)];

            const T da    = a.collision_diameter.has_value() ? *a.collision_diameter : T(0);
            const T db    = b.collision_diameter.has_value() ? *b.collision_diameter : T(0);
            const T wa    = a.viscosity_index.has_value() ? *a.viscosity_index : T(1);
            const T wb    = b.viscosity_index.has_value() ? *b.viscosity_index : T(1);
            const T aa    = a.scattering_parameter.has_value() ? *a.scattering_parameter : T(1);
            const T ab    = b.scattering_parameter.has_value() ? *b.scattering_parameter : T(1);
            const T denom = a.mass + b.mass;

            effective_collision_diameters_host[static_cast<std::size_t>(pair_index)]   = T(0.5) * (da + db);
            effective_viscosity_indices_host[static_cast<std::size_t>(pair_index)]     = T(0.5) * (wa + wb);
            effective_scattering_parameters_host[static_cast<std::size_t>(pair_index)] = T(0.5) * (aa + ab);
            reduced_masses_host[static_cast<std::size_t>(pair_index)]                  = (a.mass > T(0) && b.mass > T(0) && denom > T(0)) ? ((a.mass * b.mass) / denom) : T(0);
        }
    }

    _effective_collision_diameters = DeviceBuffer<T>(
        effective_collision_diameters_host.begin(),
        effective_collision_diameters_host.end());
    _effective_viscosity_indices = DeviceBuffer<T>(
        effective_viscosity_indices_host.begin(),
        effective_viscosity_indices_host.end());
    _effective_scattering_parameters = DeviceBuffer<T>(
        effective_scattering_parameters_host.begin(),
        effective_scattering_parameters_host.end());
    _reduced_masses = DeviceBuffer<T>(
        reduced_masses_host.begin(),
        reduced_masses_host.end());
}

template <typename T>
typename Dsmc<T>::Builder&
Dsmc<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Dsmc<T>::Builder&
Dsmc<T>::Builder::with_operator(const DsmcOperator<T>& op) noexcept {
    _operator = op;
    return *this;
}

template <typename T>
typename Dsmc<T>::Builder&
Dsmc<T>::Builder::with_solve_mode(DsmcSolveMode mode) noexcept {
    _solve_mode = mode;
    return *this;
}

template <typename T>
void
Dsmc<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Dsmc::Builder: fluid must not be null.");
    }
}

template <typename T>
Dsmc<T>
Dsmc<T>::Builder::build() const {
    validate();
    return Dsmc<T>(_fluid, _operator, _solve_mode);
}

template <typename T>
atlas::host_shared_ptr<Dsmc<T>>
Dsmc<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<Dsmc<T>>(_fluid, _operator, _solve_mode);
}

}
