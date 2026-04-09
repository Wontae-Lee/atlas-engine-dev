#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/sync/sync_operator.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace atlas::system {

template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    class Builder;

public:
    Source()  = default;
    ~Source() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(Unit<T> unit,
           FluidHostPtr<T> fluid,
           SpawnType spawn_type = SpawnType::Surface,
           bool flip            = false,
           T tolerance          = T(0),
           T temperature        = T(273.15)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(FluidDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(Unit<T> unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_type(SpawnType spawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operator(SpawnOperator<T> spawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_generator(GeneratorHostPtr<T> generator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Unit<T>&
    unit() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SpawnType
    spawn_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SpawnOperator<T>&
    spawn_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    spacing() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const GeneratorHostPtr<T>&
    generator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Vector3<T>>&
    local_positions() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

private:
    Unit<T> _unit;
    FluidHostPtr<T> _fluid;
    GeneratorHostPtr<T> _generator;
    SpawnOperator<T> _spawn_operator { SpawnType::Surface };
    bool _flip   = false;
    T _tolerance = T(0);
    T _spacing   = T(0.1);
    T _temperature { T(273.15) };
    DeviceBuffer<Vector3<T>> _local_positions;
    DeviceBuffer<size_t> _species_cache;
    DeviceBuffer<size_t> _shuffled_species;
    DeviceBuffer<std::uint64_t> _shuffle_keys;
    std::uint64_t _shuffle_seed = 0;
    bool _is_invalidated_cache  = true;
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
    with_unit(const Unit<T>& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(Unit<T>&& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_type(SpawnType spawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generator(GeneratorHostPtr<T> generator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<Unit<T>> _unit;
    FluidHostPtr<T> _fluid;
    GeneratorHostPtr<T> _generator;
    SpawnType _spawn_type = SpawnType::Surface;
    bool _flip   = false;
    T _tolerance = T(0);
    T _spacing   = T(0.1);
    T _temperature { T(273.15) };
};

}

namespace atlas {

template <typename T>
using Source = atlas::system::Source<T>;

template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

}

#include <atlas/source/source.hpp>
