#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider.h>
#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Advector final {
    static_assert(std::is_floating_point_v<T>, "Advector requires a floating-point T");

public:
    class Builder;

public:
    Advector()  = default;
    ~Advector() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_colliders() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<ColliderHostPtr<T>>&
    colliders() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    operator()(const ParticleDeviceProbe<T>& probe, T dt) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    time_integration(const ParticleDeviceProbe<T>& probe, T dt) const;

private:
    HostBuffer<ColliderHostPtr<T>> _colliders;
};

template <typename T>
class Advector<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const ColliderHostPtr<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collider(const Collider<T>& collider);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_colliders(const HostBuffer<ColliderHostPtr<T>>& colliders);

    ATLAS_HOST ATLAS_FORCE_INLINE Advector<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Advector<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<ColliderHostPtr<T>> _colliders;
};

}

namespace atlas {

template <typename T>
using Advector = atlas::system::Advector<T>;

template <typename T>
using AdvectorHostPtr = atlas::host_shared_ptr<Advector<T>>;

template <typename T>
using AdvectorDevicePtr = atlas::device_shared_ptr<Advector<T>>;

}

#include <atlas/advector/advector.hpp>
