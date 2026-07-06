#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>

#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace atlas {

bool
DsmcStatistics::measure(const DsmcProbe& probe,
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
DsmcStatistics::measure_cells(const DsmcProbe& probe,
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
                // Exhaustive C(count,2) scan is only affordable for small
                // cells; above that, 8 random pairs give a cheap
                // probabilistic estimate of the max instead. Either way this
                // is only an *estimate* of the true per-cell max sigma*g —
                // NTC's accept/reject step in DsmcSolver keeps correcting it
                // upward whenever it sees a larger real value, so an
                // under-sampled max here only costs efficiency, not
                // correctness (see dsmc_solver.h's top-of-file derivation).
                constexpr int SAMPLE_PAIRS = 8;
                float max_relative_squared = 0.0f;
                float sampled_max_sigma_g  = 0.0f;

                if (count < 5) {
                    for (int lhs_local = 0; lhs_local < count; ++lhs_local) {
                        const int pi         = probe.indices_ptr[begin + lhs_local];
                        const std::size_t si = probe.species_ptr[pi];
                        for (int rhs_local = lhs_local + 1; rhs_local < count; ++rhs_local) {
                            const int pj         = probe.indices_ptr[begin + rhs_local];
                            const std::size_t sj = probe.species_ptr[pj];
                            const float rel2     = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                            if (rel2 > max_relative_squared) max_relative_squared = rel2;

                            const float sg = probe.kernel.sigma_g(probe.properties_ptr, si, sj, rel2);
                            if (sg > sampled_max_sigma_g) sampled_max_sigma_g = sg;
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

                        const int pi         = probe.indices_ptr[begin + lhs_local];
                        const int pj         = probe.indices_ptr[begin + rhs_local];
                        const std::size_t si = probe.species_ptr[pi];
                        const std::size_t sj = probe.species_ptr[pj];
                        const float rel2     = (probe.velocity_ptr[pi] - probe.velocity_ptr[pj]).length_squared();
                        if (rel2 > max_relative_squared) max_relative_squared = rel2;

                        const float sg = probe.kernel.sigma_g(probe.properties_ptr, si, sj, rel2);
                        if (sg > sampled_max_sigma_g) sampled_max_sigma_g = sg;
                    }
                }

                probe.max_relative_speed_ptr[cell] = atlas::sqrt_nonnegative(max_relative_squared);

                if (sampled_max_sigma_g > max_sigma_g) {
                    max_sigma_g                 = sampled_max_sigma_g;
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

            // expected_count is rarely an integer; carrying the fractional
            // remainder forward (rather than rounding each step
            // independently) keeps the long-run average collision rate
            // correct instead of systematically under- or over-counting by
            // up to 1 collision every step.
            const float ntc_count               = expected_count + probe.collision_remainder_ptr[cell];
            const float base_count              = std::floor(ntc_count);
            const int collisions                = static_cast<int>(base_count);
            probe.collision_remainder_ptr[cell] = ntc_count - base_count;

            probe.collision_count_ptr[cell] = collisions;
        });
}

}
