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

// Direct Simulation Monte Carlo. Per cell it draws the NTC number of candidate
// pairs, accepts each with probability sigma*g / (sigma*g)_max, and scatters the
// accepted ones through its DsmcKernel.
//
// The candidates are flattened into one work item each before they are drawn,
// so a dense cell does not stall a whole warp while its sparse neighbours idle.
// Cell occupancy is wildly uneven whenever the grid is not tuned to the flow,
// which is the common case.
class DsmcSolver final : public Solver {
public:
    class Builder;

public:
    DsmcSolver() = default;

    ATLAS_HOST
    DsmcSolver(DsmcKernelType kernel_type,
               int majorant_sample_pairs,
               int majorant_exhaustive_limit) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_HOST SolverType
    type() const noexcept override {
        return SolverType::dsmc;
    }

    ATLAS_HOST void
    solve(Fluid& fluid,
          Universe& universe,
          const SpatialHashingSearcherView& searcher_view,
          int index,
          float dt) override;

    ATLAS_NODISCARD ATLAS_HOST DsmcKernelType
    kernel_type() const noexcept {
        return _kernel.type;
    }

    // How many candidate pairs a cell samples when estimating its majorant.
    ATLAS_NODISCARD ATLAS_HOST int
    majorant_sample_pairs() const noexcept {
        return _majorant_sample_pairs;
    }

    // At or above this occupancy the exhaustive C(n,2) scan gives way to that
    // sample.
    ATLAS_NODISCARD ATLAS_HOST int
    majorant_exhaustive_limit() const noexcept {
        return _majorant_exhaustive_limit;
    }

    // Turns the per-cell candidate counts into one work item per candidate.
    // Returns how many there are; zero means there is nothing to collide.
    //
    // Public only because nvcc refuses an extended __host__ __device__ lambda
    // inside a private member function.
    ATLAS_NODISCARD ATLAS_HOST int
    flatten_candidates(const UniverseDsmcView& universe_view, int index);

private:
    int _majorant_sample_pairs = 8;

    int _majorant_exhaustive_limit = 5;

    DsmcKernel _kernel {};

    // Bumped every step, so the hashed sample streams differ between steps.
    std::uint64_t _collision_seed = 0;

    // Scratch for flatten_candidates(): the exclusive prefix sum of the per-cell
    // candidate counts, the owning cell of every flattened candidate, the counts
    // masked to this solver's cells, and a one-element device total.
    DeviceBuffer<int> _candidate_offsets;

    DeviceBuffer<int> _candidate_cells;

    DeviceBuffer<int> _owned_candidate_counts;

    DeviceBuffer<int> _candidate_total;
};

class DsmcSolver::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST Builder&
    with_majorant_sample_pairs(int majorant_sample_pairs) noexcept;

    ATLAS_HOST Builder&
    with_majorant_exhaustive_limit(int majorant_exhaustive_limit) noexcept;

    ATLAS_NODISCARD ATLAS_HOST DsmcSolver
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<DsmcSolver>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    int _majorant_sample_pairs = 8;

    int _majorant_exhaustive_limit = 5;

    DsmcKernelType _kernel_type = DsmcKernelType::hard_sphere;
};

using DsmcSolverHostPtr = atlas::host_shared_ptr<DsmcSolver>;

using DsmcSolverDevicePtr = atlas::device_shared_ptr<DsmcSolver>;

}
