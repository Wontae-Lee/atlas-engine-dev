#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generator.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/source/source.h>
#include <atlas/universe/universe.h>

#include <cstddef>

namespace atlas {

// Drives one simulation step. Only the emission stage exists so far: every
// source spawns particle positions into the fluid and its paired generator
// fills the remaining per-particle states.
class System final {
public:
    class Builder;

public:
    System() = default;

    ATLAS_HOST
    System(FluidHostPtr fluid,
           UniverseHostPtr universe,
           HostBuffer<SourceHostPtr> sources,
           HostBuffer<GeneratorHostPtr> generators,
           const HostBuffer<Collider>& colliders,
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
    emit();

    // Sorts the live particles into the universe's grid. The per-cell particle
    // counts land in UniverseNumberParticleState when the universe carries one.
    ATLAS_HOST void
    search();

    // Moves every particle over one step. A particle whose swept segment hits a
    // collider is reflected off the nearest one instead of being integrated;
    // the colliders' own units then advance.
    ATLAS_HOST void
    advect();

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

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    source_count() const noexcept {
        return _sources.size();
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    collider_count() const noexcept {
        return _colliders.size();
    }

private:
    float _dt = 0.01f;

    FluidHostPtr _fluid {};

    UniverseHostPtr _universe {};

    // Built from the universe's grid when the system is constructed.
    SpatialHashingSearcherHostPtr _searcher {};

    // Parallel to _generators: source i emits positions, generator i fills the
    // states that spawn() does not write.
    HostBuffer<SourceHostPtr> _sources;

    HostBuffer<GeneratorHostPtr> _generators;

    // Trivially copyable, so the colliders live on the device and every
    // particle walks the whole set. Each one caches its own world AABB.
    DeviceBuffer<Collider> _colliders;
};

class System::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    // Appends one source and the generator that populates the particles it
    // spawns. The two are stored together.
    ATLAS_HOST Builder&
    with_emitter(SourceHostPtr source, GeneratorHostPtr generator);

    ATLAS_HOST Builder&
    with_collider(const Collider& collider);

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

    HostBuffer<SourceHostPtr> _sources;

    HostBuffer<GeneratorHostPtr> _generators;

    HostBuffer<Collider> _colliders;
};

using SystemHostPtr = atlas::host_shared_ptr<System>;

}
