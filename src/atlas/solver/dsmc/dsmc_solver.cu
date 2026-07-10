#include <atlas/solver/dsmc/dsmc_solver.h>

#include <atlas/fluid/fluid_view.h>
#include <atlas/logging/logging.h>
#include <atlas/material/material.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/universe/universe_view.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace atlas {

namespace {

    /**
     * @brief Whether cell @p cell is scheduled to this solver.
     *
     * A null ownership buffer is the "single solver owns everything" case, so every cell
     * is owned; otherwise ownership is the exact match `allocated_solver[cell] == index`.
     *
     * @param allocated_solver Per-cell owning-solver index, or null when unpartitioned.
     * @param cell             Cell id to test.
     * @param index            This solver's id.
     * @return `true` when this solver may process @p cell.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    owns_cell(const int* allocated_solver, const int cell, const int index) noexcept {
        return allocated_solver == nullptr || allocated_solver[cell] == index;
    }

    /**
     * @brief Widens a cell's running majorant with one candidate pair.
     *
     * Looks up the two particles at local slots `lhs_local`/`rhs_local` within the cell's
     * sorted range, and raises @p max_relative_squared and @p max_sigma_g in place if this
     * pair exceeds them. Used both by the exhaustive scan and the sampled estimate of the
     * per-cell NTC bound.
     *
     * Kept a free function rather than a lambda inside the kernel: nvcc restricts what an
     * extended device lambda may contain.
     *
     * @param fluid_view          Fluid columns (velocity, species).
     * @param searcher_view       Spatial-hash view giving the sorted particle order.
     * @param kernel              Collision kernel used to evaluate `sigma * g`.
     * @param materials           Device material dictionary.
     * @param begin               Start offset of this cell in the sorted-index array.
     * @param lhs_local           First particle's local slot within the cell.
     * @param rhs_local           Second particle's local slot within the cell.
     * @param max_relative_squared Running max squared relative speed, updated in place.
     * @param max_sigma_g         Running max `sigma * g`, updated in place.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    accumulate_majorant(const FluidDsmcView& fluid_view,
                        const SpatialHashingSearcherView& searcher_view,
                        const DsmcKernel& kernel,
                        const Material* materials,
                        const int begin,
                        const int lhs_local,
                        const int rhs_local,
                        float& max_relative_squared,
                        float& max_sigma_g) noexcept {
        const int lhs = searcher_view.sorted_index[begin + lhs_local];
        const int rhs = searcher_view.sorted_index[begin + rhs_local];

        const float relative_squared
            = (fluid_view.velocity[lhs] - fluid_view.velocity[rhs]).length_squared();

        if (relative_squared > max_relative_squared) {
            max_relative_squared = relative_squared;
        }

        const float sigma_g = kernel.sigma_g(
            materials,
            fluid_view.species[lhs],
            fluid_view.species[rhs],
            relative_squared);

        if (sigma_g > max_sigma_g) {
            max_sigma_g = sigma_g;
        }
    }

    /**
     * @brief Draws two distinct local slots of a cell holding @p count particles.
     *
     * Hashes @p cell and @p stream (with partner-selection salts) to pick a first slot in
     * `[0, count)` and a second in `[0, count - 1)`, then bumps the second past the first
     * whenever they would collide — the standard trick to sample an unordered distinct pair
     * uniformly without rejection. Stateless, so the same `(cell, stream)` always yields the
     * same pair, which is what lets pass 1 and pass 2 agree.
     *
     * @param lhs_local Receives the first partner's local slot in `[0, count)`.
     * @param rhs_local Receives the second partner's local slot, distinct from the first.
     * @param cell      Cell id, hashed into the draw.
     * @param count     Number of particles in the cell; must be >= 2.
     * @param stream    Per-collision stream base (step seed + cell mix + collision index).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sample_distinct_pair(int& lhs_local,
                         int& rhs_local,
                         const int cell,
                         const int count,
                         const std::uint64_t stream) noexcept {
        lhs_local = atlas::sample_hashed_index(cell, count, stream + atlas::DSMC_COLLISION_LHS_SALT);
        rhs_local = atlas::sample_hashed_index(cell, count - 1, stream + atlas::DSMC_COLLISION_RHS_SALT);

        // Shift the second draw up by one if it landed on or after the first, mapping the
        // reduced range back onto the distinct-partner slots.
        if (rhs_local >= lhs_local) {
            ++rhs_local;
        }
    }

    /**
     * @brief Seeds a scatter engine for one accepted collision.
     *
     * Folds the scatter salt into the collision's `(cell, stream)` pair and runs it through
     * the SplitMix64 finalizer, so neighbouring cells and successive collisions within a cell
     * start from well-separated states — an LCG seeded with adjacent values would otherwise
     * emit correlated first draws. The 64-bit key is folded onto 32 bits because the engine's
     * state is 32-bit; @ref atlas::default_random_engine::seed maps a zero fold onto its
     * default seed, so no guard is needed here.
     *
     * @param cell   Cell id owning the collision.
     * @param stream Per-collision stream base (step seed + cell mix + collision index).
     * @return An engine seeded for this collision's scatter draws.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::default_random_engine
    make_scatter_engine(const int cell, const std::uint64_t stream) noexcept {
        const std::uint64_t key
            = atlas::shuffle_key(cell, stream + atlas::DSMC_COLLISION_SCATTER_SALT);

        return atlas::default_random_engine(static_cast<std::uint32_t>(key ^ (key >> 32)));
    }

}

DsmcSolver::DsmcSolver(const DsmcKernelType kernel_type,
                       const int majorant_sample_pairs,
                       const int majorant_exhaustive_limit) noexcept
    : _majorant_sample_pairs(majorant_sample_pairs)
    , _majorant_exhaustive_limit(majorant_exhaustive_limit)
    , _kernel(kernel_type) {
}

DsmcSolver::Builder
DsmcSolver::builder() noexcept {
    return Builder {};
}

int
DsmcSolver::flatten_candidates(const UniverseDsmcView& universe_view, const int index) {
    const int cell_count = universe_view.cell_count;

    if (cell_count <= 0) {
        return 0;
    }

    const auto cells = static_cast<std::size_t>(cell_count);

    if (_candidate_offsets.size() != cells) {
        _candidate_offsets.resize(cells);
    }

    // When several solvers share the grid, mask the other solvers' cells into a
    // scratch copy rather than zeroing the caller's collision counts, which the
    // cell's next owner still needs.
    const int* scheduled = universe_view.collision_count;

    if (universe_view.allocated_solver != nullptr) {
        if (_owned_candidate_counts.size() != cells) {
            _owned_candidate_counts.resize(cells);
        }

        auto* owned           = atlas::raw_pointer_cast(_owned_candidate_counts.data());
        const auto* counts    = universe_view.collision_count;
        const auto* allocated = universe_view.allocated_solver;

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            cell_count,
            [=] ATLAS_ALL_DEVICE(const int cell) {
                owned[cell] = (allocated[cell] == index) ? counts[cell] : 0;
            });

        scheduled = owned;
    }

    atlas::exclusive_scan<ExecutionPolicy::device>(
        scheduled,
        scheduled + cell_count,
        _candidate_offsets.begin(),
        0);

    if (_candidate_total.size() < 1) {
        _candidate_total.resize(1);
    }

    auto* total         = atlas::raw_pointer_cast(_candidate_total.data());
    const auto* offsets = atlas::raw_pointer_cast(_candidate_offsets.data());
    const int last_cell = cell_count - 1;

    // The grand total is the last cell's exclusive offset plus its own count; compute it on
    // the device (one thread) so the counts never round-trip to the host, then read only the
    // single scalar back.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(int) {
            total[0] = offsets[last_cell] + scheduled[last_cell];
        });

    int candidate_count = 0;
    atlas::copy_device_to_host(total, &candidate_count, 1);

    if (candidate_count <= 0) {
        _candidate_cells.resize(0);
        return 0;
    }

    if (_candidate_cells.size() != static_cast<std::size_t>(candidate_count)) {
        _candidate_cells.resize(static_cast<std::size_t>(candidate_count));
    }

    auto* candidate_cells = atlas::raw_pointer_cast(_candidate_cells.data());

    // Recover each flat candidate's owning cell by an upper_bound search over
    // the prefix sums: the last cell whose starting offset is <= work_index.
    // One thread per candidate, and no storage beyond the offsets already built.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        candidate_count,
        [=] ATLAS_ALL_DEVICE(const int work_index) {
            int low  = 0;
            int high = cell_count;

            while (low < high) {
                const int mid = low + (high - low) / 2;

                if (offsets[mid] <= work_index) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            candidate_cells[work_index] = low - 1;
        });

    return candidate_count;
}

void
DsmcSolver::solve(Fluid& fluid,
                  Universe& universe,
                  const SpatialHashingSearcherView& searcher_view,
                  const int index,
                  const float dt) {
    const auto fluid_view    = fluid.view<FluidDsmcView>();
    const auto universe_view = universe.view<UniverseDsmcView>();

    if (!fluid_view.is_complete() || !universe_view.is_complete()) {
        atlas::warn()
            << "DsmcSolver::solve: the fluid or the universe is missing a state the "
            << "collision step needs; no collisions are performed.";
        return;
    }

    if (searcher_view.sorted_index == nullptr
        || searcher_view.cell_start == nullptr
        || searcher_view.cell_end == nullptr) {
        atlas::warn()
            << "DsmcSolver::solve: the searcher has not classified the particles; "
            << "no collisions are performed.";
        return;
    }

    if (!fluid.materials()) {
        atlas::warn()
            << "DsmcSolver::solve: the fluid carries no material dictionary; "
            << "no collisions are performed.";
        return;
    }

    const auto& material_buffer = fluid.materials()->materials();

    if (material_buffer.empty() || fluid_view.particle_count <= 0 || !(dt > 0.0f)) {
        return;
    }

    // System::Builder has already bounded every generator's species ids by the
    // dictionary's length, so the kernels index it unchecked.
    const auto* materials = atlas::raw_pointer_cast(material_buffer.data());

    const std::uint64_t collision_seed = _collision_seed++;

    const DsmcKernel kernel = _kernel;

    const int majorant_sample_pairs     = _majorant_sample_pairs;
    const int majorant_exhaustive_limit = _majorant_exhaustive_limit;

    // Pass 1 — how many candidate pairs each cell draws.
    //
    // NTC: the expected number of candidates in a cell is
    // C(n,2) * (sigma*g)_max * W * dt / V. The majorant only has to be an upper
    // bound; pass 2 raises it whenever it meets a larger real sigma*g, so an
    // under-estimate here costs efficiency, not correctness. The fractional part
    // is simply floored — the engine trades that bias for not carrying per-cell
    // remainder state.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        universe_view.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            if (!owns_cell(universe_view.allocated_solver, cell, index)) {
                return;
            }

            const int begin = searcher_view.cell_start[cell];
            const int end   = searcher_view.cell_end[cell];
            const int count = (begin < 0) ? 0 : end - begin;

            if (count < 2) {
                universe_view.collision_count[cell] = 0;
                return;
            }

            float max_relative_squared = 0.0f;
            float sampled_max_sigma_g  = 0.0f;

            if (count < majorant_exhaustive_limit) {
                for (int lhs_local = 0; lhs_local < count; ++lhs_local) {
                    for (int rhs_local = lhs_local + 1; rhs_local < count; ++rhs_local) {
                        accumulate_majorant(fluid_view, searcher_view, kernel, materials, begin, lhs_local, rhs_local, max_relative_squared, sampled_max_sigma_g);
                    }
                }
            } else {
                for (int k = 0; k < majorant_sample_pairs; ++k) {
                    const auto stream = collision_seed
                        + static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
                        + static_cast<std::uint64_t>(k);

                    int lhs_local = 0;
                    int rhs_local = 0;
                    sample_distinct_pair(lhs_local, rhs_local, cell, count, stream);

                    accumulate_majorant(fluid_view, searcher_view, kernel, materials, begin, lhs_local, rhs_local, max_relative_squared, sampled_max_sigma_g);
                }
            }

            universe_view.max_relative_speed[cell] = atlas::sqrt_nonnegative(max_relative_squared);

            // The majorant persists across steps: start from the cell's stored bound and only
            // raise it, so a rare fast pair keeps protecting the acceptance test afterwards.
            float max_sigma_g = universe_view.max_sigma_g[cell];

            if (sampled_max_sigma_g > max_sigma_g) {
                max_sigma_g                     = sampled_max_sigma_g;
                universe_view.max_sigma_g[cell] = max_sigma_g;
            }

            if (!(max_sigma_g > 0.0f) || !(universe_view.cell_volume > 0.0f)) {
                universe_view.collision_count[cell] = 0;
                return;
            }

            const float candidate_pairs = static_cast<float>(count) * static_cast<float>(count - 1) * 0.5f;

            const float expected = candidate_pairs
                * max_sigma_g
                * fluid_view.statistical_weight
                * dt
                / universe_view.cell_volume;

            universe_view.collision_count[cell] = (expected > 0.0f)
                ? static_cast<int>(std::floor(expected))
                : 0;
        });

    // Pass 2 — one work item per candidate, so cells of wildly different
    // occupancy do not leave threads idle inside a warp.
    const int candidate_count = flatten_candidates(universe_view, index);

    if (candidate_count <= 0) {
        return;
    }

    const auto* candidate_offsets = atlas::raw_pointer_cast(_candidate_offsets.data());
    const auto* candidate_cells   = atlas::raw_pointer_cast(_candidate_cells.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        candidate_count,
        [=] ATLAS_ALL_DEVICE(const int work_index) {
            const int cell = candidate_cells[work_index];

            const int begin = searcher_view.cell_start[cell];
            const int end   = searcher_view.cell_end[cell];
            const int count = (begin < 0) ? 0 : end - begin;

            if (count < 2) {
                return;
            }

            // The candidate's position within its own cell, so the hashed
            // stream matches what a per-cell loop would have drawn.
            const int collision = work_index - candidate_offsets[cell];

            const auto stream = collision_seed
                + static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(collision);

            int lhs_local = 0;
            int rhs_local = 0;
            sample_distinct_pair(lhs_local, rhs_local, cell, count, stream);

            const int lhs = searcher_view.sorted_index[begin + lhs_local];
            const int rhs = searcher_view.sorted_index[begin + rhs_local];

            if (lhs < 0 || rhs < 0 || lhs == rhs) {
                return;
            }

            const std::size_t lhs_species = fluid_view.species[lhs];
            const std::size_t rhs_species = fluid_view.species[rhs];

            Float3 lhs_velocity = fluid_view.velocity[lhs];
            Float3 rhs_velocity = fluid_view.velocity[rhs];

            const float sigma_g = kernel.sigma_g(
                materials,
                lhs_species,
                rhs_species,
                (lhs_velocity - rhs_velocity).length_squared());

            if (!(sigma_g > 0.0f)) {
                return;
            }

            // The majorant is self-correcting: a candidate larger than the
            // current bound raises it for the rest of this step and the next.
            float max_sigma_g = universe_view.max_sigma_g[cell];

            if (sigma_g > max_sigma_g) {
                max_sigma_g                     = sigma_g;
                universe_view.max_sigma_g[cell] = sigma_g;
            }

            const float accept_probability = (sigma_g < max_sigma_g)
                ? sigma_g / max_sigma_g
                : 1.0f;

            const float accept_sample = atlas::sample_hashed_unit_interval(
                cell,
                stream + atlas::DSMC_COLLISION_ACCEPT_SALT);

            if (accept_sample >= accept_probability) {
                return;
            }

            // Seeded per collision, so a replay of the same step reproduces the same scatter
            // while the angle stays independent of the pair's velocities.
            auto engine = make_scatter_engine(cell, stream);

            kernel(lhs_velocity, rhs_velocity, materials[lhs_species], materials[rhs_species], engine);

            fluid_view.velocity[lhs] = lhs_velocity;
            fluid_view.velocity[rhs] = rhs_velocity;
        });
}

DsmcSolver::Builder&
DsmcSolver::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

DsmcSolver::Builder&
DsmcSolver::Builder::with_majorant_sample_pairs(const int majorant_sample_pairs) noexcept {
    _majorant_sample_pairs = majorant_sample_pairs;
    return *this;
}

DsmcSolver::Builder&
DsmcSolver::Builder::with_majorant_exhaustive_limit(const int majorant_exhaustive_limit) noexcept {
    _majorant_exhaustive_limit = majorant_exhaustive_limit;
    return *this;
}

void
DsmcSolver::Builder::validate() const {
    if (_majorant_sample_pairs < 1) {
        throw std::runtime_error(
            "DsmcSolver::Builder: majorant_sample_pairs must be at least 1.");
    }

    // A cell needs two particles before a pair exists at all, so an exhaustive
    // scan below that bound would sample nothing.
    if (_majorant_exhaustive_limit < 2) {
        throw std::runtime_error(
            "DsmcSolver::Builder: majorant_exhaustive_limit must be at least 2.");
    }
}

DsmcSolver
DsmcSolver::Builder::build() const {
    validate();

    return DsmcSolver(_kernel_type, _majorant_sample_pairs, _majorant_exhaustive_limit);
}

atlas::host_shared_ptr<DsmcSolver>
DsmcSolver::Builder::make_host_shared() const {
    return atlas::make_host_shared<DsmcSolver>(build());
}

}
