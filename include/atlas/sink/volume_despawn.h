#pragma once

#include <atlas/geometry/geometry.h>

namespace atlas {

struct VolumeDespawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& particle,
            const float tolerance = 0.0f) noexcept {
        return query.is_inside(particle, tolerance);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3&,
            const Float3& particle,
            const float tolerance) noexcept {
        return despawn(query, particle, tolerance);
    }
};

}
