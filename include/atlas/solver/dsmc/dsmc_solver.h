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

    ATLAS_NODISCARD ATLAS_HOST int
    majorant_sample_pairs() const noexcept {
        return _majorant_sample_pairs;
    }

    ATLAS_NODISCARD ATLAS_HOST int
    majorant_exhaustive_limit() const noexcept {
        return _majorant_exhaustive_limit;
    }

    ATLAS_NODISCARD ATLAS_HOST int
    flatten_candidates(const UniverseDsmcView& universe_view, int index);

private:
    int _majorant_sample_pairs = 8;

    int _majorant_exhaustive_limit = 5;

    DsmcKernel _kernel {};

    std::uint64_t _collision_seed = 0;

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