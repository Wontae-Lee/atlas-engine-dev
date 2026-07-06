#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

#include <optional>

namespace atlas {

class Orchestrator final {
public:
    class Builder;

public:
    Orchestrator() = default;

    ~Orchestrator() = default;

    ATLAS_HOST
    Orchestrator(UniverseHostPtr universe,
                 FluidHostPtr fluid,
                 SearcherHostPtr searcher,
                 CodecHostPtr codec,
                 MeasurerHostPtr measurer,
                 HostBuffer<SolveHostPtr> solvers) noexcept;

    ATLAS_HOST void
    search();

    ATLAS_HOST void
    classify();

    ATLAS_HOST void
    measure();

    ATLAS_HOST void
    measure(float dt);

    ATLAS_HOST void
    solve(float dt);

    ATLAS_HOST void
    update(float dt);

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    orchestrate(float dt);

    ATLAS_HOST void
    set_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST void
    set_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST void
    set_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST void
    set_codec(CodecHostPtr codec) noexcept;

    ATLAS_HOST void
    set_measurer(MeasurerHostPtr measurer) noexcept;

    ATLAS_HOST void
    add_solver(SolveHostPtr solver) noexcept;

    ATLAS_NODISCARD ATLAS_HOST const SearcherHostPtr&
    searcher() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const UniverseHostPtr&
    universe() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const FluidHostPtr&
    fluid() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const CodecHostPtr&
    codec() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const MeasurerHostPtr&
    measurer() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const HostBuffer<SolveHostPtr>&
    solvers() const noexcept;

    ATLAS_HOST void
    apply_forces(const OrchestratorProbe& probe, float dt);

    ATLAS_HOST void
    apply_gravity(const OrchestratorProbe& probe, float dt);

    ATLAS_HOST void
    apply_field_force(const OrchestratorProbe& probe, float dt);

private:
    ATLAS_HOST void
    apply_probe_forces(float dt);

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    CodecHostPtr _codec {};

    MeasurerHostPtr _measurer {};

    HostBuffer<SolveHostPtr> _solvers {};

    OrchestratorProbe _probe {};
};

class Orchestrator::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_searcher(SearcherHostPtr searcher) noexcept;

    ATLAS_HOST Builder&
    with_codec(CodecHostPtr codec) noexcept;

    ATLAS_HOST Builder&
    with_measurer(MeasurerHostPtr measurer) noexcept;

    ATLAS_HOST Builder&
    with_gravity(const Float3& gravity) noexcept;

    ATLAS_HOST Builder&
    with_solver(SolveHostPtr solver) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Orchestrator
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Orchestrator>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

    ATLAS_HOST void
    ensure_gravity_state() const;

private:
    UniverseHostPtr _universe {};

    FluidHostPtr _fluid {};

    SearcherHostPtr _searcher {};

    CodecHostPtr _codec {};

    MeasurerHostPtr _measurer {};

    std::optional<Float3> _gravity {};

    HostBuffer<SolveHostPtr> _solvers {};
};

using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::Orchestrator>;

using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::Orchestrator>;

}
