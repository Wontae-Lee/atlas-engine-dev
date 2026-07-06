#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/source/source_probe.h>
#include <atlas/source/spawn.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <cstdint>

namespace atlas {

class Source final {
public:
    class Builder;

public:
    Source() = default;

    Source(const Source&) = delete;

    Source(Source&&) noexcept = default;

    ~Source() = default;

    Source&
    operator=(const Source&)
        = delete;

    Source&
    operator=(Source&&) noexcept = default;

    ATLAS_HOST
    Source(UniverseHostPtr universe,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<Spawn> spawn_operators,
           FluidHostPtr fluid,
           bool flip                = false,
           float spacing            = 0.1f,
           float tolerance          = 0.0f,
           float temperature        = 273.15f,
           ObserverHostPtr observer = nullptr) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    update(float dt);

    ATLAS_HOST void
    emit();

    ATLAS_HOST void
    rebuild_cache() noexcept;

    ATLAS_HOST void
    shuffle_species(std::size_t count);

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() noexcept;

    ATLAS_HOST void
    rebuild_spawn_cache() noexcept;

    ATLAS_HOST void
    clear_spawn_cache() noexcept;

    ATLAS_HOST void
    permute_species(std::size_t count);

    ATLAS_HOST void
    emit_particles(std::size_t dst_offset, std::size_t emit_count);

private:
    UniverseHostPtr _universe;

    DeviceBuffer<SpawnType> _spawn_types;

    DeviceBuffer<Spawn> _spawn_operators;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    bool _flip = false;

    float _spacing = 0.1f;

    float _tolerance = 0.0f;

    float _temperature { 273.15f };

    HostBuffer<int> _local_unit_counts;

    DeviceBuffer<Float3> _flat_local_positions;

    DeviceBuffer<int> _flat_unit_indices;

    std::size_t _local_particle_count = 0;

    DeviceBuffer<std::size_t> _species_cache;

    DeviceBuffer<std::size_t> _shuffled_species;

    DeviceBuffer<std::uint64_t> _shuffle_keys;

    SourceProbe _probe {};

    std::uint64_t _shuffle_seed = 0;

    bool _is_invalidated_cache = true;

    std::size_t _step_index = 0;
};

class Source::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Source
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Source>
    make_host_shared();

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(FluidHostPtr fluid) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    ATLAS_HOST Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    ATLAS_HOST Builder&
    with_spawn_operator(const Spawn& spawn_operator) noexcept;

    ATLAS_HOST Builder&
    with_spawn_operators(const HostBuffer<Spawn>& spawn_operators);

    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST Builder&
    with_spacing(float spacing) noexcept;

    ATLAS_HOST Builder&
    with_temperature(float temperature) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    ObserverHostPtr _observer {};

    HostBuffer<SpawnType> _spawn_types;

    HostBuffer<Spawn> _spawn_operators;

    bool _flip = false;

    float _spacing = 0.1f;

    float _tolerance = 0.0f;

    float _temperature { 273.15f };
};

using SourceHostPtr = atlas::host_shared_ptr<Source>;

using SourceDevicePtr = atlas::device_shared_ptr<Source>;

}
