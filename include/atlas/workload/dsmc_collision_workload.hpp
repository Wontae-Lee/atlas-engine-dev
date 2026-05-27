#pragma once

#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/dsmc_probe.h>

#include <cstdint>

namespace atlas::workload {

template <typename T>
bool
DsmcCollisionWorkload<T>::execute_collision_pair(const Probe& probe,
                                                  const int cell,
                                                  const int local_collision,
                                                  const int begin,
                                                  const int end,
                                                  const int lhs_local,
                                                  const int rhs_local,
                                                  const T max_sigma_g) noexcept {
    const int particle_i = particle_at(
        lhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    const int particle_j = particle_at(
        rhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    if (particle_i < 0 || particle_j < 0) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];

    Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
    Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

    const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const T sigma_g = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);

    if (!(sigma_g > T(0))) {
        return false;
    }

    T accept_probability = sigma_g / max_sigma_g;
    if (accept_probability > T(1)) {
        accept_probability = T(1);
    }

    const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);

    const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
        cell,
        probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

    if (accept_sample >= accept_probability) {
        return false;
    }

    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;

    return true;
}

template <typename T>
int
DsmcCollisionWorkload<T>::particle_at(const int nth,
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

}
