#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher_view.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

#include <cstddef>

namespace atlas {

class Orchestrator final {
public:
    class Builder;

public:
    Orchestrator() = default;

    ATLAS_HOST
    Orchestrator(FluidHostPtr fluid,
                 UniverseHostPtr universe,
                 HostBuffer<SolverHostPtr> solvers);

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    initialize_states();

    ATLAS_HOST void
    orchestrate(const SpatialHashingSearcherView& searcher_view, float dt);

    ATLAS_NODISCARD ATLAS_HOST const FluidHostPtr&
    fluid() const noexcept {
        return _fluid;
    }

    ATLAS_NODISCARD ATLAS_HOST const UniverseHostPtr&
    universe() const noexcept {
        return _universe;
    }

    ATLAS_NODISCARD ATLAS_HOST const HostBuffer<SolverHostPtr>&
    solvers() const noexcept {
        return _solvers;
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    solver_count() const noexcept {
        return _solvers.size();
    }

private:
    ATLAS_HOST void
    initialize_dsmc_states();

private:
    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    HostBuffer<SolverHostPtr> _solvers;
};

class Orchestrator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_solver(SolverHostPtr solver);

    ATLAS_NODISCARD ATLAS_HOST Orchestrator
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Orchestrator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    HostBuffer<SolverHostPtr> _solvers;
};

using OrchestratorHostPtr = atlas::host_shared_ptr<Orchestrator>;

using OrchestratorDevicePtr = atlas::device_shared_ptr<Orchestrator>;

}