#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder
DsmcEnergyExchangeSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcEnergyExchangeSolver<T>::apply_flattened_collision(const DeviceBuffer<int>* allocated_solver,
                                                       const int index) {
    const auto probe                = this->_probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    if (!this->_flatten_workload.build(probe.collision_count_ptr, probe.num_of_cells, allocated_solver_ptr, index)) {
        return;
    }

    const int* collision_offsets_ptr    = atlas::raw_pointer_cast(this->_flatten_workload.collision_offsets.data());
    const int* collision_cells_ptr      = atlas::raw_pointer_cast(this->_flatten_workload.collision_cells.data());
    const int flattened_collision_count = this->_flatten_workload.flattened_collision_count;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            const int cell            = collision_cells_ptr[work_index];
            const int local_collision = work_index - collision_offsets_ptr[cell];
            const int count           = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g       = probe.max_sigma_g_ptr[cell];
            const int begin           = probe.cell_start_ptr[cell];
            const int end             = probe.cell_end_ptr[cell];
            if (count < 2 || !(max_sigma_g > T(0)) || begin < 0 || end <= begin) {
                return;
            }
            if (end - begin < count) {
                return;
            }

            const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(local_collision);
            int lhs_local = 0;
            int rhs_local = 0;
            DsmcSolver<T>::sample_distinct_pair(
                lhs_local,
                rhs_local,
                cell,
                count,
                probe.collision_seed,
                stream);

            DsmcEnergyExchangeSolver<T>::collide_indexed_pair(
                probe,
                cell,
                local_collision,
                stream,
                probe.indices_ptr[begin + lhs_local],
                probe.indices_ptr[begin + rhs_local],
                max_sigma_g);
        });
}

template <typename T>
void
DsmcEnergyExchangeSolver<T>::apply_collision(const DeviceBuffer<int>* allocated_solver,
                                             const int index,
                                             const T) {
    if (this->_workload_type == DsmcCollisionWorkloadType::flatten) {
        apply_flattened_collision(allocated_solver, index);
        return;
    }

    const auto probe                = this->_probe;
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
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
            if (end - begin < count) {
                return;
            }
            const auto stream_base = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER;

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                const auto stream = stream_base + static_cast<std::uint64_t>(local_collision);
                int lhs_local     = 0;
                int rhs_local     = 0;
                DsmcSolver<T>::sample_distinct_pair(
                    lhs_local,
                    rhs_local,
                    cell,
                    count,
                    probe.collision_seed,
                    stream);

                DsmcEnergyExchangeSolver<T>::collide_indexed_pair(
                    probe,
                    cell,
                    local_collision,
                    stream,
                    probe.indices_ptr[begin + lhs_local],
                    probe.indices_ptr[begin + rhs_local],
                    max_sigma_g);
            }
        });
}

template <typename T>
bool
DsmcEnergyExchangeSolver<T>::collide_pair(const Probe& probe,
                                          const int cell,
                                          const int local_collision,
                                          const int begin,
                                          const int end,
                                          const int lhs_local,
                                          const int rhs_local,
                                          const T max_sigma_g) noexcept {
    const int particle_i = DsmcSolver<T>::particle_at(lhs_local, begin, end, probe.particle_count, probe.indices_ptr);
    const int particle_j = DsmcSolver<T>::particle_at(rhs_local, begin, end, probe.particle_count, probe.indices_ptr);

    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return DsmcEnergyExchangeSolver<T>::collide_indexed_pair(
        probe,
        cell,
        local_collision,
        stream,
        particle_i,
        particle_j,
        max_sigma_g);
}

template <typename T>
bool
DsmcEnergyExchangeSolver<T>::collide_indexed_pair(const Probe& probe,
                                                  const int cell,
                                                  const int local_collision,
                                                  const std::uint64_t stream,
                                                  const int particle_i,
                                                  const int particle_j,
                                                  const T max_sigma_g) noexcept {
    if (particle_i < 0 || particle_j < 0 || particle_i == particle_j) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];
    if (species_i >= static_cast<std::size_t>(probe.species_count)
        || species_j >= static_cast<std::size_t>(probe.species_count)) {
        return false;
    }

    Vector3<T> lhs_velocity        = probe.velocity_ptr[particle_i];
    Vector3<T> rhs_velocity        = probe.velocity_ptr[particle_j];
    const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const T sigma_g                = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);
    if (!(sigma_g > T(0))) {
        return false;
    }

    T local_max_sigma_g = max_sigma_g;
    if (sigma_g > local_max_sigma_g) {
        local_max_sigma_g           = sigma_g;
        probe.max_sigma_g_ptr[cell] = sigma_g;
    }

    T accept_probability = sigma_g / local_max_sigma_g;
    if (accept_probability > T(1)) {
        accept_probability = T(1);
    }

    const T accept_sample = atlas::sample_hashed_unit_interval<T>(
        cell,
        probe.collision_seed + stream + atlas::DSMC_COLLISION_ACCEPT_SALT);
    if (accept_sample >= accept_probability) {
        return false;
    }

    const T post_translational_energy = DsmcEnergyExchangeSolver<T>::exchange_internal_energy(
        probe,
        cell,
        local_collision,
        particle_i,
        particle_j,
        species_i,
        species_j,
        relative_speed_squared);

    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    DsmcEnergyExchangeSolver<T>::rescale_relative_velocity(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j],
        post_translational_energy);

    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;
    return true;
}

template <typename T>
T
DsmcEnergyExchangeSolver<T>::sample_unit(const int cell,
                                         const int local_collision,
                                         const std::uint64_t seed,
                                         const std::uint64_t salt) noexcept {
    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return atlas::sample_hashed_unit_interval<T>(cell, seed + stream + salt);
}

template <typename T>
T
DsmcEnergyExchangeSolver<T>::sample_bl(const T exp_1,
                                       const T exp_2,
                                       const int cell,
                                       const int local_collision,
                                       const std::uint64_t seed,
                                       const std::uint64_t salt) noexcept {
    if (!(exp_1 > T(0)) || !(exp_2 > T(0))) {
        return T(0);
    }

    const T exp_sum = exp_1 + exp_2;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const T x = DsmcEnergyExchangeSolver<T>::sample_unit(
            cell,
            local_collision,
            seed,
            salt + static_cast<std::uint64_t>(attempt) * 2u);
        const T y = std::pow(x * exp_sum / exp_1, exp_1)
            * std::pow((T(1) - x) * exp_sum / exp_2, exp_2);
        const T accept = DsmcEnergyExchangeSolver<T>::sample_unit(
            cell,
            local_collision,
            seed,
            salt + static_cast<std::uint64_t>(attempt) * 2u + 1u);
        if (accept <= y) {
            return x;
        }
    }
    return T(0.5);
}

template <typename T>
T
DsmcEnergyExchangeSolver<T>::rotational_relaxation_probability(const MaterialProperties<T>& material,
                                                               const T collision_energy,
                                                               const T omega) noexcept {
    const int dof = material.rotational_dof.value_or(0);
    if (dof <= 0) {
        return T(0);
    }

    if (material.rotational_relaxation_c1.has_value()
        && material.rotational_relaxation_c2.has_value()
        && material.rotational_relaxation_c3.has_value()
        && collision_energy > T(0)) {
        const T denominator = static_cast<T>(atlas::boltzmann_constant)
            * (T(2.5) - omega + static_cast<T>(dof) * T(0.5));
        if (denominator > T(0)) {
            const T tr = collision_energy / denominator;
            if (tr > T(0)) {
                const T probability = (T(1)
                                       + material.rotational_relaxation_c2.value() / atlas::sqrt_nonnegative(tr)
                                       + material.rotational_relaxation_c3.value() / tr)
                    / material.rotational_relaxation_c1.value();
                return probability < T(0) ? T(0) : (probability > T(1) ? T(1) : probability);
            }
        }
    }

    return material.rotational_relaxation_probability.value_or(T(0));
}

template <typename T>
T
DsmcEnergyExchangeSolver<T>::vibrational_relaxation_probability(const MaterialProperties<T>& material,
                                                                const T collision_energy,
                                                                const T omega) noexcept {
    if (material.vibrational_dof.value_or(0) <= 0) {
        return T(0);
    }

    if (material.vibrational_relaxation_c1.has_value()
        && material.vibrational_relaxation_c2.has_value()
        && collision_energy > T(0)) {
        const T denominator = static_cast<T>(atlas::boltzmann_constant) * (T(3.5) - omega);
        if (denominator > T(0)) {
            const T tr = collision_energy / denominator;
            if (tr > T(0)) {
                const T probability = T(1)
                    / (material.vibrational_relaxation_c1.value() / std::pow(tr, omega)
                       * std::exp(material.vibrational_relaxation_c2.value() / std::pow(tr, T(1) / T(3))));
                return probability < T(0) ? T(0) : (probability > T(1) ? T(1) : probability);
            }
        }
    }

    return material.vibrational_relaxation_probability.value_or(T(0));
}

template <typename T>
T
DsmcEnergyExchangeSolver<T>::exchange_internal_energy(const Probe& probe,
                                                      const int cell,
                                                      const int local_collision,
                                                      const int particle_i,
                                                      const int particle_j,
                                                      const std::size_t species_i,
                                                      const std::size_t species_j,
                                                      const T relative_speed_squared) noexcept {
    const auto& lhs_material = probe.properties_ptr[species_i];
    const auto& rhs_material = probe.properties_ptr[species_j];
    const auto pair          = DsmcKernel<T>::pair_parameters(lhs_material, rhs_material);
    if (!pair.valid) {
        return T(0);
    }

    T e_dispose = T(0.5) * pair.reduced_mass * relative_speed_squared;
    if (probe.internal_energy_ptr == nullptr) {
        return e_dispose;
    }

    DsmcEnergyExchangeSolver<T>::exchange_particle_internal_energy(
        probe,
        cell,
        local_collision,
        particle_i,
        lhs_material,
        pair.viscosity_index,
        0x7f4a7c15ull,
        e_dispose);
    DsmcEnergyExchangeSolver<T>::exchange_particle_internal_energy(
        probe,
        cell,
        local_collision,
        particle_j,
        rhs_material,
        pair.viscosity_index,
        0x94d049bbull,
        e_dispose);

    return e_dispose > T(0) ? e_dispose : T(0);
}

template <typename T>
void
DsmcEnergyExchangeSolver<T>::exchange_particle_internal_energy(const Probe& probe,
                                                               const int cell,
                                                               const int local_collision,
                                                               const int particle,
                                                               const MaterialProperties<T>& material,
                                                               const T omega,
                                                               const std::uint64_t salt_base,
                                                               T& e_dispose) noexcept {
    auto energy = probe.internal_energy_ptr[particle];

    const int rot_dof       = material.rotational_dof.value_or(0);
    const T rot_probability = DsmcEnergyExchangeSolver<T>::rotational_relaxation_probability(
        material,
        e_dispose + energy.rotational,
        omega);
    if (rot_dof > 0
        && DsmcEnergyExchangeSolver<T>::sample_unit(cell, local_collision, probe.collision_seed, salt_base) <= rot_probability) {
        e_dispose += energy.rotational;
        if (rot_dof == 2) {
            const T exponent  = T(2.5) - omega;
            const T u         = DsmcEnergyExchangeSolver<T>::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 1u);
            energy.rotational = exponent > T(0)
                ? (T(1) - std::pow(u, T(1) / exponent)) * e_dispose
                : T(0);
        } else {
            energy.rotational = e_dispose * DsmcEnergyExchangeSolver<T>::sample_bl(static_cast<T>(rot_dof) * T(0.5) - T(1), T(1.5) - omega, cell, local_collision, probe.collision_seed, salt_base + 2u);
        }
        e_dispose -= energy.rotational;
    } else if (rot_dof <= 0) {
        energy.rotational = T(0);
    }

    const int vib_dof       = material.vibrational_dof.value_or(0);
    const T vib_probability = DsmcEnergyExchangeSolver<T>::vibrational_relaxation_probability(
        material,
        e_dispose + energy.vibrational,
        omega);
    if (vib_dof > 0
        && DsmcEnergyExchangeSolver<T>::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 17u) <= vib_probability) {
        e_dispose += energy.vibrational;
        if (vib_dof == 2) {
            const T exponent   = T(2.5) - omega;
            const T u          = DsmcEnergyExchangeSolver<T>::sample_unit(cell, local_collision, probe.collision_seed, salt_base + 18u);
            energy.vibrational = exponent > T(0)
                ? (T(1) - std::pow(u, T(1) / exponent)) * e_dispose
                : T(0);
        } else {
            energy.vibrational = e_dispose * DsmcEnergyExchangeSolver<T>::sample_bl(static_cast<T>(vib_dof) * T(0.5) - T(1), T(1.5) - omega, cell, local_collision, probe.collision_seed, salt_base + 19u);
        }
        e_dispose -= energy.vibrational;
    } else if (vib_dof <= 0) {
        energy.vibrational = T(0);
    }

    energy.translational                = e_dispose;
    probe.internal_energy_ptr[particle] = energy;
}

template <typename T>
void
DsmcEnergyExchangeSolver<T>::rescale_relative_velocity(Vector3<T>& lhs_velocity,
                                                       Vector3<T>& rhs_velocity,
                                                       const MaterialProperties<T>& lhs,
                                                       const MaterialProperties<T>& rhs,
                                                       const T translational_energy) noexcept {
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))
        || !(translational_energy > T(0))) {
        return;
    }

    const T reduced_mass = lhs_mass * rhs_mass / mass_sum;
    if (!(reduced_mass > T(0))) {
        return;
    }

    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();
    if (!(speed > T(0))) {
        return;
    }

    const T target_speed                = atlas::sqrt_nonnegative(T(2) * translational_energy / reduced_mass);
    const Vector3<T> scattered_relative = relative * (target_speed / speed);
    const Vector3<T> center             = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder&
DsmcEnergyExchangeSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder&
DsmcEnergyExchangeSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder&
DsmcEnergyExchangeSolver<T>::Builder::with_searcher(SearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder&
DsmcEnergyExchangeSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename DsmcEnergyExchangeSolver<T>::Builder&
DsmcEnergyExchangeSolver<T>::Builder::with_workload_type(const DsmcCollisionWorkloadType workload_type) noexcept {
    _workload_type = workload_type;
    return *this;
}

template <typename T>
void
DsmcEnergyExchangeSolver<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: universe must not be null.");
    }
    if (!_fluid) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: fluid must not be null.");
    }
    if (!_searcher) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcEnergyExchangeSolver<T>
DsmcEnergyExchangeSolver<T>::Builder::build() const {
    validate();
    return DsmcEnergyExchangeSolver<T>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcEnergyExchangeSolver<T>>
DsmcEnergyExchangeSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcEnergyExchangeSolver<T>>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

}