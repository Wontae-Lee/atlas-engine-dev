#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_probe.h>
#include <atlas/collider/detail/collider_bound_cache.h>
#include <atlas/collider/detail/collider_collision_kernel.h>
#include <atlas/collider/detail/collider_probe_builder.h>
#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <type_traits>

namespace atlas {

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
             DeviceBuffer<SurfaceInteractionKernel<T>> surface_interactions,
             DeviceBuffer<std::uint8_t> flips,
             PostColliderType post_collider_type,
             atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(T dt) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() const noexcept;

private:
    DeviceBuffer<Unit<T>> _units;

    mutable detail::ColliderBoundCache<T> _bound_cache;

    FluidHostPtr<T> _fluid;

    DeviceBuffer<SurfaceInteractionKernel<T>> _surface_interactions;

    DeviceBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };

    mutable ColliderProbe<T> _probe {};
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
    with_surface_interactions(const HostBuffer<IsothermalSurfaceInteraction<T>>& surface_interactions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<MaxwellianSurfaceInteraction<T>>& surface_interactions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interaction_kernel(const SurfaceInteractionKernel<T>& surface_interaction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interaction_kernels(const HostBuffer<SurfaceInteractionKernel<T>>& surface_interactions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flips(const HostBuffer<std::uint8_t>& flips);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_post_collider_type(PostColliderType type) noexcept;

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

    HostBuffer<SurfaceInteractionKernel<T>> _surface_interactions;

    HostBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };
};

}

namespace atlas {
template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

}

#include <atlas/collider/collider.hpp>