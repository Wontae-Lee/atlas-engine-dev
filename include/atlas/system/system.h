#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generator.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <filesystem>
#include <string>

namespace atlas {

class System final {
public:
    class Builder;

public:
    System() = default;

    ATLAS_HOST
    System(FluidHostPtr fluid,
           UniverseHostPtr universe,
           OrchestratorHostPtr orchestrator,
           HostBuffer<SourceHostPtr> sources,
           HostBuffer<GeneratorHostPtr> generators,
           const HostBuffer<Collider>& colliders,
           const HostBuffer<Sink>& sinks,
           CodecHostPtr codec,
           ObserverHostPtr observer,
           float dt);

    System(const System&) = default;

    System(System&&) noexcept = default;

    ~System() = default;

    System&
    operator=(const System&)
        = default;

    System&
    operator=(System&&) noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    update();

    ATLAS_HOST void
    emit();

    ATLAS_HOST void
    search();

    ATLAS_HOST void
    allocate();

    ATLAS_HOST void
    orchestrate();

    ATLAS_HOST void
    advect();

    ATLAS_HOST void
    remove();

    ATLAS_NODISCARD ATLAS_HOST float
    dt() const noexcept {
        return _dt;
    }

    ATLAS_NODISCARD ATLAS_HOST const FluidHostPtr&
    fluid() const noexcept {
        return _fluid;
    }

    ATLAS_NODISCARD ATLAS_HOST const UniverseHostPtr&
    universe() const noexcept {
        return _universe;
    }

    ATLAS_NODISCARD ATLAS_HOST const SpatialHashingSearcherHostPtr&
    searcher() const noexcept {
        return _searcher;
    }

    ATLAS_NODISCARD ATLAS_HOST const OrchestratorHostPtr&
    orchestrator() const noexcept {
        return _orchestrator;
    }

    ATLAS_NODISCARD ATLAS_HOST const CodecHostPtr&
    codec() const noexcept {
        return _codec;
    }

    ATLAS_NODISCARD ATLAS_HOST const ObserverHostPtr&
    observer() const noexcept {
        return _observer;
    }

    ATLAS_HOST void
    save(const std::filesystem::path& directory) const;

    ATLAS_NODISCARD ATLAS_HOST static std::string
    snapshot_directory_name(std::size_t step);

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    step() const noexcept {
        return _step;
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    source_count() const noexcept {
        return _sources.size();
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    collider_count() const noexcept {
        return _colliders.size();
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    sink_count() const noexcept {
        return _sinks.size();
    }

    ATLAS_HOST void
    mark_survivors(int particle_count);

private:
    float _dt = 0.01f;

    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    SpatialHashingSearcherHostPtr _searcher {};

    std::size_t _step = 0;

    OrchestratorHostPtr _orchestrator {};

    CodecHostPtr _codec {};

    ObserverHostPtr _observer {};

    HostBuffer<SourceHostPtr> _sources;

    HostBuffer<GeneratorHostPtr> _generators;

    DeviceBuffer<Collider> _colliders;

    DeviceBuffer<Sink> _sinks;
};

class System::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_orchestrator(OrchestratorHostPtr orchestrator) noexcept;

    ATLAS_HOST Builder&
    with_emitter(SourceHostPtr source, GeneratorHostPtr generator);

    ATLAS_HOST Builder&
    with_collider(const Collider& collider);

    ATLAS_HOST Builder&
    with_sink(const Sink& sink);

    ATLAS_HOST Builder&
    with_codec(CodecHostPtr codec) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

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
    float _dt = 0.01f;

    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    OrchestratorHostPtr _orchestrator {};

    CodecHostPtr _codec {};

    ObserverHostPtr _observer {};

    HostBuffer<SourceHostPtr> _sources;

    HostBuffer<GeneratorHostPtr> _generators;

    HostBuffer<Collider> _colliders;

    HostBuffer<Sink> _sinks;
};

using SystemHostPtr = atlas::host_shared_ptr<System>;

}