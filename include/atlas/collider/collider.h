#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_probe.h>
#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/collider/kernel/collider_collision_kernel.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstdint>

namespace atlas {

class Collider final {
public:
    class Builder;

public:
    Collider() = default;

    Collider(const Collider&) = delete;

    Collider(Collider&&) noexcept = default;

    ~Collider() = default;

    Collider&
    operator=(const Collider&)
        = delete;

    Collider&
    operator=(Collider&&) noexcept = default;

    ATLAS_HOST
    Collider(UniverseHostPtr universe,
             DeviceBuffer<SurfaceInteractionKernel> surface_interactions,
             DeviceBuffer<std::uint8_t> flips,
             PostColliderType post_collider_type,
             atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST void
    update(float dt);

    ATLAS_HOST void
    collide(float dt) const;

    ATLAS_NODISCARD ATLAS_HOST bool
    empty() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    make_probe() const noexcept;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    DeviceBuffer<SurfaceInteractionKernel> _surface_interactions;

    DeviceBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };

    mutable ColliderProbe _probe {};
};

class Collider::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid> fluid) noexcept;

    ATLAS_HOST Builder&
    with_surface_interactions(const HostBuffer<IsothermalSurfaceInteraction>& surface_interactions);

    ATLAS_HOST Builder&
    with_surface_interactions(const HostBuffer<MaxwellianSurfaceInteraction>& surface_interactions);

    ATLAS_HOST Builder&
    with_surface_interaction_kernel(const SurfaceInteractionKernel& surface_interaction);

    ATLAS_HOST Builder&
    with_surface_interaction_kernels(const HostBuffer<SurfaceInteractionKernel>& surface_interactions);

    ATLAS_HOST Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST Builder&
    with_flips(const HostBuffer<std::uint8_t>& flips);

    ATLAS_HOST Builder&
    with_post_collider_type(PostColliderType type) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Collider
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Collider>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe;

    FluidHostPtr _fluid;

    HostBuffer<SurfaceInteractionKernel> _surface_interactions;

    HostBuffer<std::uint8_t> _flips;

    PostColliderType _post_collider_type { PostColliderType::fast };
};

using ColliderHostPtr = atlas::host_shared_ptr<Collider>;

using ColliderDevicePtr = atlas::device_shared_ptr<Collider>;

}
