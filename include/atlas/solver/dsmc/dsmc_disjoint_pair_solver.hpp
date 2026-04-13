#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcDisjointPairSolver<T>::DsmcDisjointPairSolver(FluidHostPtr<T> fluid,
                                                  DsmcKernel<T> op)
    : DsmcSolver<T>(std::move(fluid), std::move(op)) { }

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder
DsmcDisjointPairSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcDisjointPairSolver<T>::solve(DomainDeviceProbe<T>& domain,
                                 SpatialHashingProbe<T>& searcher,
                                 FluidDeviceProbe<T>& particle,
                                 CodecDeviceProbe<T>&) {
    if (particle.particle_count <= 0 || particle.particle_property == nullptr || particle.vel == nullptr || particle.species == nullptr) {
        return;
    }

    this->apply_field_force(domain, particle);
    const auto collision_operator               = this->_operator;
    auto* velocities                            = particle.vel;
    const auto* species                         = particle.species;
    const auto* properties                      = particle.particle_property;
    const auto* cell_start                      = searcher.cell_start;
    const auto* cell_end                        = searcher.cell_end;
    const auto* indices                         = searcher.indices;
    const auto* field_temperature               = domain.field_temperature;
    const auto* effective_collision_diameters   = atlas::raw_pointer_cast(this->_effective_collision_diameters.data());
    const auto* effective_viscosity_indices     = atlas::raw_pointer_cast(this->_effective_viscosity_indices.data());
    const auto* effective_scattering_parameters = atlas::raw_pointer_cast(this->_effective_scattering_parameters.data());
    const auto* reduced_masses                  = atlas::raw_pointer_cast(this->_reduced_masses.data());
    const auto pair_indexer                     = this->_pair_indexer;
    const int num_cells                         = domain.num_of_cells;
    const int species_count                     = this->_species_count;

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

    const int total_pairs                              = last_pair_offset + last_pair_count;
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
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_operator(const DsmcKernel<T>& op) noexcept {
    _operator = op;
    return *this;
}

template <typename T>
void
DsmcDisjointPairSolver<T>::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: fluid must not be null.");
    }
}

template <typename T>
DsmcDisjointPairSolver<T>
DsmcDisjointPairSolver<T>::Builder::build() const {
    validate();
    return DsmcDisjointPairSolver<T>(_fluid, _operator);
}

template <typename T>
atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
DsmcDisjointPairSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcDisjointPairSolver<T>>(_fluid, _operator);
}

}
