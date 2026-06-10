#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>
#include <cstddef>

namespace atlas::system {

template <typename T>
bool
DsmcStatistics<T>::measure(const DsmcProbe<T>& probe,
                           const DeviceBuffer<int>* allocated_solver,
                           const int index,
                           const T dt) const {
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end = probe.cell_end_ptr[cell];
            const int count = end - begin;

            T max_relative_squared = T(0);
            T measured_max_sigma_g = T(0);

            for (int lhs_sorted_index = begin; lhs_sorted_index < end; ++lhs_sorted_index) {
                const int particle_i = probe.indices_ptr[lhs_sorted_index];

                for (int rhs_sorted_index = lhs_sorted_index + 1; rhs_sorted_index < end; ++rhs_sorted_index) {
                    const int particle_j = probe.indices_ptr[rhs_sorted_index];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed_squared = relative_velocity.length_squared();

                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    const T sigma_g = probe.kernel.sigma_g(
                        probe.properties_ptr,
                        species_i,
                        species_j,
                        relative_speed_squared);

                    if (sigma_g > measured_max_sigma_g) {
                        measured_max_sigma_g = sigma_g;
                    }
                }
            }

            probe.number_particle_ptr[cell] = static_cast<T>(count);

            probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);

            const T stored_max_sigma_g = probe.max_sigma_g_ptr[cell];
            const T max_sigma_g = measured_max_sigma_g > stored_max_sigma_g
                ? measured_max_sigma_g
                : stored_max_sigma_g;
            probe.max_sigma_g_ptr[cell] = max_sigma_g;

            if (count < 2 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const T ntc_pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T cell_volume = probe.universe_volume_ptr != nullptr
                ? probe.universe_volume_ptr[cell]
                : probe.cell_volume;

            if (!(cell_volume > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const T expected_count = ntc_pair_count
                * max_sigma_g
                * probe.statistical_weight
                * dt
                / cell_volume;

            if (!(expected_count > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const T ntc_count = expected_count + probe.collision_remainder_ptr[cell];
            const T base_count = std::floor(ntc_count);
            const int collisions = static_cast<int>(base_count);
            probe.collision_remainder_ptr[cell] = ntc_count - base_count;

            probe.collision_count_ptr[cell] = collisions;
        });

    return true;
}

} // namespace atlas::system
