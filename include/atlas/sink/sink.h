#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/memory.h>
#include <atlas/remove/remove.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    class Builder;

public:
    Sink()  = default;
    ~Sink() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Sink(DeviceBuffer<Unit<T>> units,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<DespawnOperator<T>> despawn_operators,
         bool flip   = false,
         T tolerance = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink(FluidDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operators(DeviceBuffer<DespawnOperator<T>> despawn_operators) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_types(DeviceBuffer<DespawnType> despawn_types) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<DespawnOperator<T>>&
    despawn_operators() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<DespawnOperator<T>>&
    despawn_operators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<DespawnType>&
    despawn_types() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<DespawnType>&
    despawn_types() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    DeviceBuffer<Unit<T>> _units;
    DeviceBuffer<DespawnType> _despawn_types;
    DeviceBuffer<DespawnOperator<T>> _despawn_operators;
    bool _flip   = false;
    T _tolerance = T(0);
};

template <typename T>
class Sink<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Sink<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sink<T>>
    make_host_shared();

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_types(const HostBuffer<DespawnType>& despawn_types);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operator(const DespawnOperator<T>& despawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_operators(const HostBuffer<DespawnOperator<T>>& despawn_operators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<Unit<T>> _units;
    HostBuffer<DespawnType> _despawn_types;
    HostBuffer<DespawnOperator<T>> _despawn_operators;
    bool _flip                = false;
    T _tolerance              = T(0);
};

}

namespace atlas {

template <typename T>
using Sink = atlas::system::Sink<T>;

template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

}

#include <atlas/sink/sink.hpp>
