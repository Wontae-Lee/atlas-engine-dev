#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
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
    Sink(Unit<T> unit,
         DespawnType despawn_type = DespawnType::Surface,
         bool flip                = false,
         T tolerance              = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    sink(FluidDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(Unit<T> unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_type(DespawnType despawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_despawn_operator(DespawnOperator<T> despawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Unit<T>&
    unit() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DespawnType
    despawn_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DespawnOperator<T>&
    despawn_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

private:
    Unit<T> _unit;
    DespawnOperator<T> _despawn_operator { DespawnType::Surface };
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
    with_unit(const Unit<T>& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(Unit<T>&& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_despawn_type(DespawnType despawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<Unit<T>> _unit;
    DespawnType _despawn_type = DespawnType::Surface;
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
