#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>

namespace atlas::system {

template <typename T>
bool
DsmcSimpleStatistics<T>::measure(const DsmcProbe<T>& probe,
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
            for (int lhs_sorted_index = begin; lhs_sorted_index < end; ++lhs_sorted_index) {
                const int particle_i = probe.indices_ptr[lhs_sorted_index];
                for (int rhs_sorted_index = lhs_sorted_index + 1; rhs_sorted_index < end; ++rhs_sorted_index) {
                    const int particle_j = probe.indices_ptr[rhs_sorted_index];
                    const T relative_speed_squared =
                        (probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j]).length_squared();
                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }
                }
            }

            const T max_relative_speed = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);

            probe.number_particle_ptr[cell] = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative_speed;

            const T stored_max_sigma_g = probe.max_sigma_g_ptr[cell];
            const T max_sigma_g = max_relative_speed > stored_max_sigma_g
                ? max_relative_speed
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
