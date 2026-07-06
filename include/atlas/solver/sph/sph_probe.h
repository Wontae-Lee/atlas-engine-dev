#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_kernel.h>

#include <cstddef>

/**
 * @file sph_probe.h
 * @brief Flat, device-copyable view over everything one SPH step needs:
 *        particle state, the spatial grid, and precomputed neighbor
 *        lists. Same non-owning-probe pattern as `ColliderProbe`/
 *        `DsmcProbe` (see `collider_probe.h`).
 *
 * Unlike `DsmcProbe` (which only needs each cell's own particle range —
 * DSMC collisions are intra-cell), SPH sums over every particle within
 * the kernel's support radius, which generally spans several
 * neighboring cells. `neighbor_offsets_ptr`/`neighbor_indices_ptr` hold
 * a precomputed, per-particle CSR-style neighbor list (particle `i`'s
 * neighbors are `neighbor_indices_ptr[neighbor_offsets_ptr[i] ..
 * neighbor_offsets_ptr[i+1])`), built once per step by the searcher so
 * `estimate_density`/`accelerate` don't have to re-walk the grid's
 * `3x3x3` (or larger) neighbor-cell block on every access.
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of an `SphSolver`'s fluid/universe/searcher
 *        state, passed by value into the density-estimation and
 *        force-accumulation kernels. See `SphSolver::make_probe` for
 *        how it is filled and `sph_solver.h` for how each field is
 *        used.
 */
struct SphProbe {
    /** Particle positions, owned by `Fluid`. */
    const Float3* position_ptr {};
    /** Particle velocities, owned by `Fluid`; updated in place by
     *  `SphSolver::accelerate`. */
    Float3* velocity_ptr {};
    /** Particle species indices, owned by `Fluid`. */
    const std::size_t* species_ptr {};
    /** Per-species material properties (indexed by `species_ptr`). */
    const MaterialProperties* properties_ptr {};
    /** Per-cell particle counts, written by `SphSolver::count_particles`. */
    float* number_particle_ptr {};
    /** Per-cell averaged particle acceleration (mass-weighted mean),
     *  written by `SphSolver::accelerate`'s second pass. */
    Float3* field_force_ptr {};
    /** Searcher's particle indices sorted by cell. */
    const int* indices_ptr {};
    /** Per-cell start offset into `indices_ptr`. */
    const int* cell_start_ptr {};
    /** Per-cell (one-past-)end offset into `indices_ptr`. */
    const int* cell_end_ptr {};
    /** Per-particle CSR offsets into `neighbor_indices_ptr` (length
     *  `particle_count + 1`); see this file's top-of-file documentation. */
    const int* neighbor_offsets_ptr {};
    /** Flattened neighbor-particle indices, sliced per particle by
     *  `neighbor_offsets_ptr`. */
    const int* neighbor_indices_ptr {};
    /** Spatial grid's world-space lower corner. */
    Float3 lower_corner {};
    /** Spatial grid's cell counts along each axis. */
    Int3 grid_size {};
    /** `1 / cell_size`, precomputed for grid-index arithmetic. */
    float inverse_cell_size {};
    /** Grid cell size — also the SPH kernel's support radius `h`. */
    float cell_size {};
    /** Length of `position_ptr`/`velocity_ptr`/`species_ptr`. */
    int particle_count {};
    /** Number of spatial cells. */
    int cell_count {};
    /** Length of `properties_ptr`. */
    int property_count {};
    /** The active SPH smoothing kernel; see `SphKernel`. */
    SphKernel kernel {};
};

using SphSolverProbe = atlas::SphProbe;

}
