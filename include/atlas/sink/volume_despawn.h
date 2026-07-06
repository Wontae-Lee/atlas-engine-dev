#pragma once

#include <atlas/geometry/geometry.h>

/**
 * @file volume_despawn.h
 * @brief Despawn rule: a particle is removed once it lies inside a
 *        unit's volume — the standard outflow model for a solid sink
 *        region (e.g. an absorbing wall or exit port with real
 *        thickness) rather than an infinitesimally thin surface.
 */

namespace atlas {

/**
 * @brief `despawn()` true iff the particle is inside the unit's volume
 *        within `tolerance` (`Geometry::is_inside`).
 */
struct VolumeDespawn final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Vector3& particle,
            const float tolerance = 0.0f) noexcept {
        return query.is_inside(particle, tolerance);
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Vector3&,
            const Vector3& particle,
            const float tolerance) noexcept {
        return despawn(query, particle, tolerance);
    }
};

}
