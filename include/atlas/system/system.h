#pragma once

#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

namespace atlas {

class System final {
public:
    class Builder;

public:
    System() = default;

    ATLAS_HOST
    System(FluidHostPtr fluid,
           UniverseHostPtr universe,
           SourceHostPtr source,
           SinkHostPtr sink,
           ColliderHostPtr collider,
           OrchestratorHostPtr orchestrator,
           float dt = 0.01f);

    ~System() = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    update();

    ATLAS_HOST void
    emit();

    ATLAS_HOST void
    orchestrate();

    ATLAS_HOST void
    advect();

    ATLAS_HOST void
    remove();

    ATLAS_HOST void
    time_integration();

    ATLAS_NODISCARD ATLAS_HOST float
    dt() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const FluidHostPtr&
    fluid() const noexcept;

private:
    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    SourceHostPtr _source {};

    SinkHostPtr _sink {};

    ColliderHostPtr _collider {};

    OrchestratorHostPtr _orchestrator {};

    float _dt { 0.01f };

    FluidPositionState* _cached_position_state {};
    FluidVelocityState* _cached_velocity_state {};
};

class System::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_fluid(const FluidHostPtr& fluid) noexcept;

    ATLAS_HOST Builder&
    with_domain(const UniverseHostPtr& universe) noexcept;

    ATLAS_HOST Builder&
    with_source(const SourceHostPtr& source) noexcept;

    ATLAS_HOST Builder&
    with_sink(const SinkHostPtr& sink) noexcept;

    ATLAS_HOST Builder&
    with_collider(const ColliderHostPtr& collider) noexcept;

    ATLAS_HOST Builder&
    with_solver(const OrchestratorHostPtr& orchestrator) noexcept;

    ATLAS_HOST Builder&
    with_dt(float dt) noexcept;

    ATLAS_NODISCARD ATLAS_HOST System
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<System>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    SourceHostPtr _source {};

    SinkHostPtr _sink {};

    ColliderHostPtr _collider {};

    OrchestratorHostPtr _orchestrator {};

    float _dt { 0.01f };
};

using SystemHostPtr = atlas::host_shared_ptr<System>;

}
