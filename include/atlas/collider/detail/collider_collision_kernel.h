#pragma once

#include <atlas/collider/collider_probe.h>
#include <atlas/collider/detail/collider_hit.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/core/macros.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace atlas::detail {

template <typename T>
class ColliderCollisionKernel final {
    static_assert(std::is_floating_point_v<T>, "ColliderCollisionKernel requires a floating-point T");

private:
    struct ParticleSweep {
        Vector3<T> origin {};
        Vector3<T> velocity {};
        Vector3<T> direction {};
        T speed {};
        T length {};
    };

public:
    ATLAS_HOST ATLAS_FORCE_INLINE static void
    resolve_particles(const ColliderProbe<T>& probe, PostColliderType post_collider_type, T dt);

private:
    ATLAS_DEVICE static ParticleSweep
    make_particle_sweep(const ColliderProbe<T>& probe, int particle_index, T dt);

    ATLAS_DEVICE static ColliderHit<T>
    closest_hit(const ColliderProbe<T>& probe,
                const PostColliderKernel<T>& post_collider_kernel,
                const ParticleSweep& sweep,
                T dt);

    ATLAS_DEVICE static void
    resolve_hit(const ColliderProbe<T>& probe,
                const PostColliderKernel<T>& post_collider_kernel,
                int particle_index,
                const ParticleSweep& sweep,
                const ColliderHit<T>& hit,
                T dt);
};

}

#include <atlas/collider/detail/collider_collision_kernel.hpp>