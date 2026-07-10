#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe_view.h>

#include <cstdint>

namespace atlas {

/**
 * @brief Direct Simulation Monte Carlo (DSMC) collision solver over the spatial grid.
 *
 * Implements the No-Time-Counter (NTC) scheme: for every grid cell it owns, it draws a
 * number of candidate collision pairs proportional to the cell's occupancy and its
 * running majorant `(sigma*g)_max`, then accepts each pair with probability
 * `sigma*g / (sigma*g)_max` and, on acceptance, scatters the pair through the selected
 * @ref DsmcKernel. Only velocities change; positions are integrated elsewhere.
 *
 * @ref solve runs in two device passes. Pass 1 estimates a per-cell candidate count and
 * refreshes the majorant; pass 2 flattens those counts (@ref flatten_candidates) into a
 * one-work-item-per-candidate list so warps stay busy even when cell occupancy varies
 * wildly, then evaluates and applies each candidate. The scratch @c DeviceBuffer members
 * make this a @ref HostVariant-style owner (move/host only) rather than a device-capturable
 * leaf; the solver itself is host-owned and dispatched virtually by the System.
 *
 * Randomness is stateless: every draw is a hash of the cell id, a per-step
 * @c _collision_seed, and a purpose salt, so a run is reproducible and restartable.
 *
 * @note Multiple DsmcSolvers can share one grid; each processes only the cells the codec
 *       assigned to it (`allocated_solver[cell] == index`). A null ownership buffer means
 *       this solver owns every cell.
 */
class DsmcSolver final : public Solver {
public:
    /** @brief Fluent builder that validates the majorant parameters before constructing. */
    class Builder;

public:
    /** @brief Default-constructs a solver with the default kernel and majorant settings. */
    DsmcSolver() = default;

    /**
     * @brief Constructs a solver with explicit kernel and majorant-sampling parameters.
     *
     * Preconditions on the arguments are enforced by @ref Builder::validate, not here;
     * this constructor trusts its inputs and cannot throw.
     *
     * @param kernel_type               Which collision kernel leaf to activate.
     * @param majorant_sample_pairs     Number of random pairs sampled to estimate a
     *                                  cell's majorant when its occupancy is large.
     * @param majorant_exhaustive_limit Occupancy below which every pair is scanned
     *                                  exactly instead of sampled.
     */
    ATLAS_HOST
    DsmcSolver(DsmcKernelType kernel_type,
               int majorant_sample_pairs,
               int majorant_exhaustive_limit) noexcept;

    /** @brief Defaulted destructor; the device scratch buffers free themselves. */
    ~DsmcSolver() override = default;

    /**
     * @brief Returns a fresh @ref Builder for fluent configuration.
     * @return A default-initialized builder.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Identifies this solver as the DSMC kind.
     * @return Always @ref SolverType::dsmc.
     */
    ATLAS_NODISCARD ATLAS_HOST SolverType
    type() const noexcept override {
        return SolverType::dsmc;
    }

    /**
     * @brief Performs one DSMC collision sub-step over the cells this solver owns.
     *
     * Gathers @ref FluidDsmcView and @ref UniverseDsmcView on the host, validates that
     * the fluid, grid, searcher, and material dictionary are all present, then launches
     * the two NTC passes. Silently no-ops (with a warning) when a required input is
     * missing, and returns without effect when `dt <= 0`, the fluid is empty, or no
     * candidate pairs were scheduled. Overrides @ref Solver::solve.
     *
     * @param fluid         Particle population; only the velocity column is written.
     * @param universe      Grid whose per-cell majorant and collision counters are updated.
     * @param searcher_view Classified spatial-hash view giving per-cell particle ranges.
     * @param index         This solver's id, matched against the cell ownership buffer.
     * @param dt            Sub-step duration in seconds; must be positive.
     */
    ATLAS_HOST void
    solve(Fluid& fluid,
          Universe& universe,
          const SpatialHashingSearcherView& searcher_view,
          int index,
          float dt) override;

    /**
     * @brief The collision kernel kind currently selected.
     * @return The active @ref DsmcKernel's discriminator tag.
     */
    ATLAS_NODISCARD ATLAS_HOST DsmcKernelType
    kernel_type() const noexcept {
        return _kernel.type;
    }

    /**
     * @brief Number of random pairs used to estimate a large cell's majorant.
     * @return The configured sample count (>= 1).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    majorant_sample_pairs() const noexcept {
        return _majorant_sample_pairs;
    }

    /**
     * @brief Occupancy threshold below which the majorant is computed exactly.
     * @return The configured exhaustive limit (>= 2).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    majorant_exhaustive_limit() const noexcept {
        return _majorant_exhaustive_limit;
    }

    /**
     * @brief Turns the per-cell candidate counts into a flat candidate work list.
     *
     * Exclusive-scans the per-cell collision counts into @c _candidate_offsets, reads the
     * grand total back to the host to size @c _candidate_cells, then fills @c _candidate_cells
     * so entry `w` names the cell owning the `w`-th candidate (found by binary search over
     * the offsets). When several solvers share the grid, the other solvers' counts are first
     * masked into @c _owned_candidate_counts so their scheduling is left untouched. All
     * scratch buffers grow as needed and are reused across steps.
     *
     * @param universe_view Grid view supplying `cell_count`, the per-cell `collision_count`,
     *                      and the optional `allocated_solver` ownership buffer.
     * @param index         This solver's id, used to select the cells it owns.
     * @return The total number of scheduled candidate pairs (0 when there is no work).
     *
     * @note Public despite being an internal step: it launches an extended
     *       `__host__ __device__` lambda, which nvcc forbids inside a private/protected
     *       member function.
     */
    ATLAS_NODISCARD ATLAS_HOST int
    flatten_candidates(const UniverseDsmcView& universe_view, int index);

private:
    int _majorant_sample_pairs = 8; ///< Random pairs sampled per large cell to bound the majorant.

    int _majorant_exhaustive_limit = 5; ///< Occupancy below which the majorant is scanned exactly.

    DsmcKernel _kernel {}; ///< The active cross-section/scatter kernel (tagged union of leaves).

    std::uint64_t _collision_seed = 0; ///< Monotonic per-step stream base; incremented once per solve.

    DeviceBuffer<int> _candidate_offsets; ///< Exclusive prefix sums of per-cell candidate counts.

    DeviceBuffer<int> _candidate_cells; ///< For each flat candidate, its owning cell id.

    DeviceBuffer<int> _owned_candidate_counts; ///< Scratch counts with other solvers' cells masked to 0.

    DeviceBuffer<int> _candidate_total; ///< Single-element device buffer holding the grand candidate total.
};

/**
 * @brief Fluent builder for @ref DsmcSolver that validates the majorant parameters.
 *
 * Setters return `*this` for chaining; @ref build calls @ref validate (which throws on
 * out-of-range parameters) before constructing. Defaults mirror @ref DsmcSolver's own
 * member defaults so an unconfigured builder yields a usable solver.
 */
class DsmcSolver::Builder final {
public:
    /** @brief Default-constructs a builder holding the default kernel and majorant settings. */
    Builder() = default;

    /**
     * @brief Selects the collision kernel leaf.
     * @param kernel_type The kernel kind to activate.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    /**
     * @brief Sets how many random pairs estimate a large cell's majorant.
     * @param majorant_sample_pairs Sample count; @ref validate requires >= 1.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_majorant_sample_pairs(int majorant_sample_pairs) noexcept;

    /**
     * @brief Sets the occupancy below which the majorant is computed exactly.
     * @param majorant_exhaustive_limit Threshold; @ref validate requires >= 2.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_majorant_exhaustive_limit(int majorant_exhaustive_limit) noexcept;

    /**
     * @brief Validates the parameters and constructs the solver by value.
     * @return A fully configured @ref DsmcSolver.
     * @throws std::runtime_error if a majorant parameter is out of range.
     */
    ATLAS_NODISCARD ATLAS_HOST DsmcSolver
    build() const;

    /**
     * @brief Builds the solver and wraps it in a shared host handle.
     * @return A @ref DsmcSolverHostPtr owning the newly built solver.
     * @throws std::runtime_error if @ref build's validation fails.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<DsmcSolver>
    make_host_shared() const;

private:
    /**
     * @brief Enforces the parameter invariants used by the majorant estimation.
     * @throws std::runtime_error if `majorant_sample_pairs < 1` or
     *         `majorant_exhaustive_limit < 2`.
     */
    ATLAS_HOST void
    validate() const;

private:
    int _majorant_sample_pairs = 8; ///< Pending sample-pair count; must be >= 1.

    int _majorant_exhaustive_limit = 5; ///< Pending exhaustive-scan threshold; must be >= 2.

    DsmcKernelType _kernel_type = DsmcKernelType::hard_sphere; ///< Pending kernel selection.
};

/** @brief Shared-ownership host handle to a @ref DsmcSolver. */
using DsmcSolverHostPtr = atlas::host_shared_ptr<DsmcSolver>;

/** @brief Device-resident shared handle to a @ref DsmcSolver; provided for symmetry. */
using DsmcSolverDevicePtr = atlas::device_shared_ptr<DsmcSolver>;

}