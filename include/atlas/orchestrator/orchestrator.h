#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/measure/measurer.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/detail/orchestrator_force_applier.h>
#include <atlas/orchestrator/detail/orchestrator_pipeline.h>
#include <atlas/orchestrator/detail/orchestrator_probe_builder.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

#include <optional>

namespace atlas {

template <typename T>
class Orchestrator final {
    template <typename>
    friend struct detail::OrchestratorForceStage;

public:
    class Builder;

public:
    Orchestrator() = default;

    ~Orchestrator() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Orchestrator(UniverseHostPtr<T> universe,
                 FluidHostPtr<T> fluid,
                 SearcherHostPtr<T> searcher,
                 CodecHostPtr<T> codec,
                 MeasurerHostPtr<T> measurer,
                 HostBuffer<SolveHostPtr<T>> solvers) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    search();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    classify();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    measure(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(CodecHostPtr<T> codec) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_measurer(MeasurerHostPtr<T> measurer) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SearcherHostPtr<T>&
    searcher() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const UniverseHostPtr<T>&
    universe() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MeasurerHostPtr<T>&
    measurer() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SolveHostPtr<T>>&
    solvers() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_forces(const OrchestratorProbe<T>& probe, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_gravity(const OrchestratorProbe<T>& probe, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(const OrchestratorProbe<T>& probe, T dt);

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_probe_forces(T dt);

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    CodecHostPtr<T> _codec {};

    MeasurerHostPtr<T> _measurer {};

    HostBuffer<SolveHostPtr<T>> _solvers {};

    detail::OrchestratorPipeline<T> _pipeline {};

    OrchestratorProbe<T> _probe {};

    detail::OrchestratorProbeBuilder<T> _probe_builder {};

    detail::OrchestratorForceApplier<T> _force_applier {};
};

template <typename T>
class Orchestrator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(CodecHostPtr<T> codec) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_measurer(MeasurerHostPtr<T> measurer) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_gravity(const Vector3<T>& gravity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_gravity_state() const;

private:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    SearcherHostPtr<T> _searcher {};

    CodecHostPtr<T> _codec {};

    MeasurerHostPtr<T> _measurer {};

    std::optional<Vector3<T>> _gravity {};

    HostBuffer<SolveHostPtr<T>> _solvers {};
};

}

namespace atlas {

template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::Orchestrator<T>>;

template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::Orchestrator<T>>;

}

#include <atlas/orchestrator/orchestrator.hpp>