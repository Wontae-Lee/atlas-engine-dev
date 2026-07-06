#pragma once

#include <atlas/collider/kernel/fast_collider_kernel.h>

namespace atlas {

class DtRemainColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DtRemainColliderKernel() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sweep_motion(const Unit&,
                 const Float3&,
                 const Float3& incident,
                 const float incident_speed,
                 const float dt,
                 Float3& sweep_direction,
                 float& sweep_speed,
                 float& sweep_length) const noexcept {
        sweep_direction = incident * dt;
        sweep_speed     = incident_speed;
        sweep_length    = incident_speed * dt;
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
        const Float3 wall_velocity     = FastColliderKernel::surface_velocity(unit, hit_position);
        const Float3 relative_incident = incident - wall_velocity;
        const Float3 reflected         = interaction(relative_incident, hit_normal) + wall_velocity;
        const Float3 offset_position   = hit_position + hit_normal * atlas::tol;

        position = offset_position;
        velocity = reflected;

        if (!(sweep_speed > atlas::eps)) {
            return;
        }

        const float hit_time  = hit_distance / sweep_speed;
        const float remaining = (dt > hit_time) ? (dt - hit_time) : 0.0f;

        position = offset_position + reflected * remaining;
    }
};

}
