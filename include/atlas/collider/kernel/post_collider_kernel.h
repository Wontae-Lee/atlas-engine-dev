#pragma once

#include <../../core/device_variant.h>
#include <atlas/collider/kernel/dt_remain_collider_kernel.h>
#include <atlas/collider/kernel/fast_collider_kernel.h>
#include <atlas/collider/kernel/precise_collider_kernel.h>

namespace atlas {

enum struct PostColliderType : int {

    fast,

    dt_remain,

    precise
};

struct PostColliderKernel final {

    PostColliderType type = PostColliderType::fast;

    union {
        FastColliderKernel fast;
        DtRemainColliderKernel dt_remain;
        PreciseColliderKernel precise;
    };

    ATLAS_ALL_DEVICE
    PostColliderKernel() noexcept;

    ATLAS_ALL_DEVICE explicit PostColliderKernel(PostColliderType type) noexcept;

    ATLAS_ALL_DEVICE
    PostColliderKernel(const PostColliderKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE PostColliderKernel&
    operator=(const PostColliderKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~PostColliderKernel() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit& unit,
                 const Float3& origin,
                 const Float3& incident,
                 float incident_speed,
                 float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& position,
               Float3& velocity,
               const Float3& incident,
               const Float3& hit_position,
               const Float3& hit_normal,
               float hit_distance,
               float sweep_speed,
               float dt,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept;
};

using PostColliderVariant = DeviceVariant<
    PostColliderKernel,
    PostColliderType,
    PostColliderType::fast,
    DeviceVariantCase<PostColliderType::fast, &PostColliderKernel::fast>,
    DeviceVariantCase<PostColliderType::dt_remain, &PostColliderKernel::dt_remain>,
    DeviceVariantCase<PostColliderType::precise, &PostColliderKernel::precise>>;

struct PostColliderSweepMotion {
    const Unit& unit;
    const Float3& origin;
    const Float3& incident;
    float incident_speed;
    float dt;
    Float3& sweep_direction;
    float& sweep_speed;
    float& sweep_length;

    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const K& kernel) const noexcept {
        kernel.sweep_motion(unit, origin, incident, incident_speed, dt, sweep_direction, sweep_speed, sweep_length);
    }
};

struct PostColliderApply {
    Float3& position;
    Float3& velocity;
    const Float3& incident;
    const Float3& hit_position;
    const Float3& hit_normal;
    float hit_distance;
    float sweep_speed;
    float dt;
    const Unit& unit;
    const SurfaceInteractionKernel& interaction;

    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const K& kernel) const noexcept {
        kernel(position, velocity, incident, hit_position, hit_normal, hit_distance, sweep_speed, dt, unit, interaction);
    }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
PostColliderKernel::PostColliderKernel() noexcept {
    PostColliderVariant::construct(*this, PostColliderType::fast);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
PostColliderKernel::PostColliderKernel(const PostColliderType type_) noexcept {
    PostColliderVariant::construct(*this, type_);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
PostColliderKernel::sweep_motion(const Unit& unit,
                                 const Float3& origin,
                                 const Float3& incident,
                                 const float incident_speed,
                                 const float dt,
                                 Float3& sweep_direction,
                                 float& sweep_speed,
                                 float& sweep_length) const noexcept {
    PostColliderVariant::apply(
        *this,
        PostColliderSweepMotion {
            unit,
            origin,
            incident,
            incident_speed,
            dt,
            sweep_direction,
            sweep_speed,
            sweep_length });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
PostColliderKernel::operator()(Float3& position,
                               Float3& velocity,
                               const Float3& incident,
                               const Float3& hit_position,
                               const Float3& hit_normal,
                               const float hit_distance,
                               const float sweep_speed,
                               const float dt,
                               const Unit& unit,
                               const SurfaceInteractionKernel& interaction) const noexcept {
    PostColliderVariant::apply(
        *this,
        PostColliderApply {
            position,
            velocity,
            incident,
            hit_position,
            hit_normal,
            hit_distance,
            sweep_speed,
            dt,
            unit,
            interaction });
}

}
