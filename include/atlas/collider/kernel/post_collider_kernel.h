#pragma once

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>
#include <atlas/collider/kernel/fast_collider_kernel.h>
#include <atlas/collider/kernel/precise_collider_kernel.h>
#include <atlas/core/detail/device_variant.h>

namespace atlas {

enum struct PostColliderType : int {

    fast,

    dt_remain,

    precise
};

template <typename T>
struct PostColliderKernel final {

    PostColliderType type = PostColliderType::fast;

    union {
        FastColliderKernel<T> fast;
        DtRemainColliderKernel<T> dt_remain;
        PreciseColliderKernel<T> precise;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PostColliderKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit PostColliderKernel(PostColliderType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PostColliderKernel(const PostColliderKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE PostColliderKernel&
    operator=(const PostColliderKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~PostColliderKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit<T>& unit,
                 const Vector3<T>& origin,
                 const Vector3<T>& incident,
                 T incident_speed,
                 T dt,
                 Vector3<T>& sweep_direction,
                 T& sweep_speed,
                 T& sweep_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& position,
               Vector3<T>& velocity,
               const Vector3<T>& incident,
               const Vector3<T>& hit_position,
               const Vector3<T>& hit_normal,
               T hit_distance,
               T sweep_speed,
               T dt,
               const Unit<T>& unit,
               const SurfaceInteractionKernel<T>& interaction) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const PostColliderKernel& other) noexcept;
};

}

#include <atlas/collider/kernel/post_collider_kernel.hpp>