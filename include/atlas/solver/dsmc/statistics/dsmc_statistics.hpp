#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

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

            probe.number_particle_ptr[cell] = static_cast<T>(count);

            // Load the persistent SPARTA-style NTC majorant; the collision hot
            // path only ever grows it, so max(stored, sampled) preserves it.
            T max_sigma_g = probe.max_sigma_g_ptr[cell];

            // Keep tiny cells exact, but avoid the serial O(N^2) scan once the
            // cell is large enough for the pair loop to dominate a GPU thread.
            if (count >= 2) {
                constexpr int SAMPLE_PAIRS = 8;
                T max_relative_squared = T(0);
                T sampled_max_sigma_g  = T(0);

                if (count < 5) {
                    for (int lhs_local = 0; lhs_local < count; ++lhs_local) {
                        const int pi = probe.indices_ptr[begin + lhs_local];
                        const std::size_t si = probe.species_ptr[pi];
                        for (int rhs_local = lhs_local + 1; rhs_local < count; ++rhs_local) {
                            const int pj = probe.indices_ptr[begin + rhs_local];
                            const std::size_t sj = probe.species_ptr[pj];
                            const T rel2 = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                            if (rel2 > max_relative_squared) max_relative_squared = rel2;

                            const T sg = probe.kernel.sigma_g(probe.properties_ptr, si, sj, rel2);
                            if (sg > sampled_max_sigma_g) sampled_max_sigma_g = sg;
                        }
                    }
                } else {
                    for (int k = 0; k < SAMPLE_PAIRS; ++k) {
                        const auto k64 = static_cast<std::uint64_t>(k);
                        const int lhs_local = atlas::sampling::sample_hashed_index(
                            cell, count,
                            probe.collision_seed + k64 * 2u + atlas::seed::DSMC_COLLISION_LHS_SALT);
                        int rhs_local = atlas::sampling::sample_hashed_index(
                            cell, count - 1,
                            probe.collision_seed + k64 * 2u + 1u + atlas::seed::DSMC_COLLISION_RHS_SALT);
                        if (rhs_local >= lhs_local) ++rhs_local;

                        const int pi = probe.indices_ptr[begin + lhs_local];
                        const int pj = probe.indices_ptr[begin + rhs_local];
                        const std::size_t si = probe.species_ptr[pi];
                        const std::size_t sj = probe.species_ptr[pj];
                        const T rel2 = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                        if (rel2 > max_relative_squared) max_relative_squared = rel2;

                        const T sg = probe.kernel.sigma_g(probe.properties_ptr, si, sj, rel2);
                        if (sg > sampled_max_sigma_g) sampled_max_sigma_g = sg;
                    }
                }

                probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                    ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                    : T(0);

                // NTC majorant: take max of the persistent value and the current sample.
                if (sampled_max_sigma_g > max_sigma_g) {
                    max_sigma_g = sampled_max_sigma_g;
                    probe.max_sigma_g_ptr[cell] = max_sigma_g;
                }
            }

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
