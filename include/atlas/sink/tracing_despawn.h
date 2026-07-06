#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

namespace atlas {

struct TracingDespawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& position,
            const Float3& velocity,
            const float time = 0.0f) noexcept {
        const float speed = velocity.length();
        if (!(time > 0.0f) || !(speed > 0.0f)) {
            return false;
        }
        const auto hit = query.trace(atlas::Ray(position, velocity));
        return hit.is_intersecting && hit.distance >= 0.0f && hit.distance <= speed * time;
    }
};

}
