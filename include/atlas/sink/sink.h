#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sink/despawn_operator.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::fluid {

template <typename T>
class Sink final {
    static_assert(std::is_floating_point_v<T>, "Sink requires a floating-point T");

public:
    class Builder;

public:
    Sink() = default;

    ~Sink() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Sink(DeviceBuffer<Unit<T>> units,
         DeviceBuffer<DespawnType> despawn_types,
         DeviceBuffer<DespawnOperator<T>> despawn_operators,
         atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
         bool flip   = false,
         T tolerance = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink();

private:
    DeviceBuffer<Unit<T>> _units;

    DeviceBuffer<DespawnType> _despawn_types;

    DeviceBuffer<DespawnOperator<T>> _despawn_operators;

    atlas::host_shared_ptr<atlas::Fluid<T>> _fluid;

    bool _flip = false;

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
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

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

    FluidHostPtr<T> _fluid;

    HostBuffer<DespawnType> _despawn_types;

    HostBuffer<DespawnOperator<T>> _despawn_operators;

    bool _flip = false;

    T _tolerance = T(0);
};

}

namespace atlas {

template <typename T>
using Sink = atlas::fluid::Sink<T>;

template <typename T>
using SinkHostPtr = atlas::host_shared_ptr<Sink<T>>;

template <typename T>
using SinkDevicePtr = atlas::device_shared_ptr<Sink<T>>;

}

#include <atlas/sink/sink.hpp>
