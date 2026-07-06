#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstdint>

namespace atlas {

bool
DsmcSimpleStatistics::measure(const DsmcProbe& probe,
                              const DeviceBuffer<int>* allocated_solver,
                              const int index,
                              const float dt) const {
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    measure_cells(probe, allocated_solver_ptr, index, dt);

    return true;
}

void
DsmcSimpleStatistics::measure_cells(const DsmcProbe& probe,
                                    const int* allocated_solver_ptr,
                                    const int index,
                                    const float dt) {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            const int count = end - begin;

            probe.number_particle_ptr[cell] = static_cast<float>(count);

            float max_sigma_g = probe.max_sigma_g_ptr[cell];

            if (count >= 2) {
                constexpr int SAMPLE_PAIRS = 8;
                float max_relative_squared = 0.0f;
                if (count < 5) {
                    for (int lhs_local = 0; lhs_local < count; ++lhs_local) {
                        const int pi = probe.indices_ptr[begin + lhs_local];
                        for (int rhs_local = lhs_local + 1; rhs_local < count; ++rhs_local) {
                            const int pj     = probe.indices_ptr[begin + rhs_local];
                            const float rel2 = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                            if (rel2 > max_relative_squared) max_relative_squared = rel2;
                        }
                    }
                } else {
                    for (int k = 0; k < SAMPLE_PAIRS; ++k) {
                        const auto k64      = static_cast<std::uint64_t>(k);
                        const int lhs_local = atlas::sample_hashed_index(
                            cell,
                            count,
                            probe.collision_seed + k64 * 2u + atlas::DSMC_COLLISION_LHS_SALT);
                        int rhs_local = atlas::sample_hashed_index(
                            cell,
                            count - 1,
                            probe.collision_seed + k64 * 2u + 1u + atlas::DSMC_COLLISION_RHS_SALT);
                        if (rhs_local >= lhs_local) ++rhs_local;

                        const int pi     = probe.indices_ptr[begin + lhs_local];
                        const int pj     = probe.indices_ptr[begin + rhs_local];
                        const float rel2 = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                        if (rel2 > max_relative_squared) max_relative_squared = rel2;
                    }
                }

                const float max_relative_speed = atlas::sqrt_nonnegative(max_relative_squared);

                probe.max_relative_speed_ptr[cell] = max_relative_speed;

                // Unlike DsmcStatistics, this skips DsmcKernel::sigma_g
                // entirely and uses raw relative speed as the max_sigma_g
                // bound. NTC's accept/reject step in DsmcSolver still keeps
                // this bound self-correcting toward the true sigma*g over
                // time, so this only trades cross-section fidelity for a
                // cheaper per-cell measurement, not correctness — see
                // dsmc_simple_solver.h's top-of-file documentation.
                if (max_relative_speed > max_sigma_g) {
                    max_sigma_g                 = max_relative_speed;
                    probe.max_sigma_g_ptr[cell] = max_sigma_g;
                }
            }

            if (count < 2 || !(max_sigma_g > 0.0f)) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const float ntc_pair_count = static_cast<float>(count) * static_cast<float>(count - 1) * 0.5f;
            const float cell_volume    = probe.universe_volume_ptr != nullptr
                   ? probe.universe_volume_ptr[cell]
                   : probe.cell_volume;

            if (!(cell_volume > 0.0f)) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const float expected_count = ntc_pair_count
                * max_sigma_g
                * probe.statistical_weight
                * dt
                / cell_volume;

            if (!(expected_count > 0.0f)) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            const float ntc_count               = expected_count + probe.collision_remainder_ptr[cell];
            const float base_count              = std::floor(ntc_count);
            const int collisions                = static_cast<int>(base_count);
            probe.collision_remainder_ptr[cell] = ntc_count - base_count;

            probe.collision_count_ptr[cell] = collisions;
        });
}

}
