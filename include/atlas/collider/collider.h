#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    class Builder;

public:
    Collider() = default;

    ~Collider() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(DeviceBuffer<Unit<T>> units,
             DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions,
             atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(T dt) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    DeviceBuffer<Unit<T>> _units;

    FluidHostPtr<T> _fluid;

    DeviceBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

template <typename T>
class Collider<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<Unit<T>> _units;

    FluidHostPtr<T> _fluid;

    HostBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

}

namespace atlas {

template <typename T>
using Collider = atlas::system::Collider<T>;

template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

}

#include <atlas/collider/collider.hpp>
