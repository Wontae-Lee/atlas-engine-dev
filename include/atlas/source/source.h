#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/memory.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace atlas::fluid {

template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    class Builder;

public:
    Source() = default;

    ~Source() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(DeviceBuffer<Unit<T>> units,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<SpawnOperator<T>> spawn_operators,
           atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
           bool flip     = false,
           T spacing     = T(0.1),
           T tolerance   = T(0),
           T temperature = T(273.15)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    shuffle_species(std::size_t count);

private:
    DeviceBuffer<Unit<T>> _units;
    DeviceBuffer<SpawnType> _spawn_types;
    DeviceBuffer<SpawnOperator<T>> _spawn_operators;

    FluidHostPtr<T> _fluid;

    bool _flip = false;

    T _spacing = T(0.1);

    T _tolerance = T(0);

    T _temperature { T(273.15) };

    HostBuffer<DeviceBuffer<Vector3<T>>> _local_positions;

    std::size_t _local_particle_count = 0;

    DeviceBuffer<std::size_t> _species_cache;

    DeviceBuffer<std::size_t> _shuffled_species;

    DeviceBuffer<std::uint64_t> _shuffle_keys;

    std::uint64_t _shuffle_seed = 0;

    bool _is_invalidated_cache = true;
};

template <typename T>
class Source<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Source<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Source<T>>
    make_host_shared();

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<Unit<T>> _units;

    FluidHostPtr<T> _fluid;

    HostBuffer<SpawnType> _spawn_types;

    HostBuffer<SpawnOperator<T>> _spawn_operators;

    bool _flip = false;

    T _spacing = T(0.1);

    T _tolerance = T(0);

    T _temperature { T(273.15) };
};

}

namespace atlas {

template <typename T>
using Source = atlas::fluid::Source<T>;

template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

}

#include <atlas/source/source.hpp>
