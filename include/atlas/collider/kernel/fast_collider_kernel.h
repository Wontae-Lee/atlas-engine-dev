#pragma once

#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/math/math.h>
#include <atlas/unit/unit.h>

namespace atlas {

class FastColliderKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    FastColliderKernel() noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    surface_velocity(const Unit& unit,
                     const Float3& surface_point) noexcept {
        Float3 velocity(0.0f, 0.0f, 0.0f);

        if (unit.velocity().has_value()) {
            velocity += *unit.velocity();
        }

        if (unit.angular_velocity().has_value()) {
            const Float3 radius = surface_point - unit.sync().translation;
            velocity += atlas::cross(*unit.angular_velocity(), radius);
        }

        return velocity;
    }

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
               const float,
               const float,
               const float,
               const Unit& unit,
               const SurfaceInteractionKernel& interaction) const noexcept {
        const Float3 wall_velocity     = surface_velocity(unit, hit_position);
        const Float3 relative_incident = incident - wall_velocity;

        position = hit_position + hit_normal * atlas::tol;
        velocity = interaction(relative_incident, hit_normal) + wall_velocity;
    }
};

}
