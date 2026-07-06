#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>

#include <cstddef>
#include <cstdint>

/**
 * @file dsmc_probe.h
 * @brief Flat, device-copyable view over everything one DSMC step needs:
 *        particle state, per-cell NTC bookkeeping, and the spatial cell
 *        partition. Same non-owning-probe pattern as `ColliderProbe`
 *        (see `collider_probe.h` for why this pattern exists).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `DsmcSolver`'s fluid/universe/searcher
 *        state, passed by value into the NTC selection and collision
 *        kernels. See `detail::DsmcProbeBuilder` for how it is filled
 *        and `dsmc_solver.h` for how each field is used.
 */
struct DsmcProbe {
    /** Particle velocities, owned by `Fluid`; read and updated in place
     *  by collision kernels. */
    Vector3* velocity_ptr {};
    /** Particle internal energy, owned by `Fluid`; `nullptr` if the
     *  fluid does not track it (see `DsmcEnergyExchangeSolver`). */
    FluidInternalEnergy* internal_energy_ptr {};
    /** Particle species indices, owned by `Fluid`. */
    const std::size_t* species_ptr {};
    /** Per-species material properties (indexed by `species_ptr`). */
    const MaterialProperties* properties_ptr {};
    /** Per-cell running mean particle count, used by
     *  `DsmcStatistics`/`DsmcSimpleStatistics` to estimate the NTC
     *  candidate count. */
    float* number_particle_ptr {};
    /** Per-cell running maximum relative speed observed, another NTC
     *  statistic input. */
    float* max_relative_speed_ptr {};
    /** Per-cell running upper bound on `sigma * g` (see
     *  `dsmc_solver.h`'s NTC derivation); raised in place whenever a
     *  sampled candidate pair exceeds it. */
    float* max_sigma_g_ptr {};
    /** Per-cell fractional leftover candidate count carried between
     *  steps (the true `N_candidates` is rarely an integer). */
    float* collision_remainder_ptr {};
    /** Per-cell integer candidate count to draw and test this step. */
    int* collision_count_ptr {};
    /** Searcher's particle indices sorted by cell (see
     *  `atlas::Searcher`); `[cell_start_ptr[c], cell_end_ptr[c])` is
     *  cell `c`'s slice. */
    const int* indices_ptr {};
    /** Per-cell start offset into `indices_ptr`. */
    const int* cell_start_ptr {};
    /** Per-cell (one-past-)end offset into `indices_ptr`. */
    const int* cell_end_ptr {};
    /** Total universe volume, used to derive `cell_volume` when cells
     *  are uniform. */
    const float* universe_volume_ptr {};
    /** Length of `velocity_ptr`/`species_ptr` (and `internal_energy_ptr`
     *  when non-null). */
    int particle_count {};
    /** Length of `properties_ptr`. */
    int species_count {};
    /** Number of spatial cells (length of the per-cell arrays above). */
    int cell_count {};
    /** Volume of one spatial cell (uniform-grid assumption). */
    float cell_volume {};
    /** Real molecules per simulated particle (`F_N` in the NTC
     *  candidate-count formula; see `dsmc_solver.h`). */
    float statistical_weight {};
    /** The active collision model; see `DsmcKernel`. */
    DsmcKernel kernel {};
    /** Per-step random seed mixed into every hashed sample this step
     *  draws (see `dsmc_solver.h`'s stateless-hash-RNG note). */
    std::uint64_t collision_seed {};
};

}
