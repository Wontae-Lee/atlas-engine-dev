#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/sink/despawn.h>
#include <atlas/sink/sink_probe.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstddef>

namespace atlas {

class Sink final {
public:
    class Builder;

public:
    Sink() = default;

    Sink(const Sink&) = delete;

    Sink(Sink&&) noexcept = default;

    ~Sink() = default;

    Sink&
    operator=(const Sink&)
        = delete;

    Sink&
    operator=(Sink&&) noexcept = default;

    ATLAS_HOST
    Sink(UniverseHostPtr universe,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<Despawn> despawn_operators,
         atlas::host_shared_ptr<atlas::Fluid> fluid,
         bool flip                = false,
         float tolerance          = 0.0f,
         ObserverHostPtr observer = nullptr) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    update(float dt);

    ATLAS_HOST void
    sink(float dt = 0.0f);

    ATLAS_HOST void
    compact_fluid_particles();

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe(float dt = 0.0f) noexcept;

    ATLAS_HOST void
    refresh_unit_bounds() noexcept;

    ATLAS_HOST void
    despawn_particles(const SinkProbe& probe, int* removed_unit_indices);

private:
    UniverseHostPtr _universe;

    DeviceBuffer<atlas::AABB> _unit_bounds;

    DeviceBuffer<DespawnType> _despawn_types;

    DeviceBuffer<Despawn> _despawn_operators;

    atlas::host_shared_ptr<atlas::Fluid> _fluid;

    ObserverHostPtr _observer {};

    bool _flip = false;

    float _tolerance = 0.0f;

    DeviceBuffer<std::size_t> _keep;

    DeviceBuffer<std::size_t> _offsets;

    DeviceBuffer<std::size_t> _compact_indices;

    DeviceBuffer<int> _despawned_unit_indices;

    DeviceBuffer<std::size_t> _total_count_buffer {};

    SinkProbe _probe {};

    std::size_t _step_index = 0;
};

class Sink::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Sink
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Sink>
    make_host_shared();

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    ATLAS_HOST Builder&
    with_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    ATLAS_HOST Builder&
    with_despawn_operator(const Despawn& despawn_operator) noexcept;

    ATLAS_HOST Builder&
    with_despawn_operators(const HostBuffer<Despawn>& despawn_operators);

    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    HostBuffer<DespawnType> _despawn_types;

    HostBuffer<Despawn> _despawn_operators;

    bool _flip = false;

    float _tolerance = 0.0f;
};

using SinkHostPtr = atlas::host_shared_ptr<Sink>;

using SinkDevicePtr = atlas::device_shared_ptr<Sink>;

}
