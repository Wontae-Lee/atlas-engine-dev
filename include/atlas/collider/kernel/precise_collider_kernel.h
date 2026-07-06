#pragma once

#include <atlas/collider/kernel/dt_remain_collider_kernel.h>

namespace atlas {

class PreciseColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    PreciseColliderKernel() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit& unit,
                 const Float3& origin,
                 const Float3& incident,
                 const float,
                 const float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept {
        const Float3 relative_velocity = incident - FastColliderKernel::surface_velocity(unit, origin);
        sweep_direction                = relative_velocity * dt;
        sweep_speed                    = relative_velocity.length();
        sweep_length                   = sweep_speed * dt;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& position,
               Float3& velocity,
               const Float3& incident,
               const Float3& hit_position,
               const Float3& hit_normal,
               const float hit_distance,
               const float sweep_speed,
               const float dt,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept {
        DtRemainColliderKernel {}(
            position,
            velocity,
            incident,
            hit_position,
            hit_normal,
            hit_distance,
            sweep_speed,
            dt,
            unit,
            interaction);
    }
};

}
