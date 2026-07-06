#pragma once

#include <atlas/geometry/geometry.h>

/**
 * @file surface_despawn.h
 * @brief Despawn rule: a particle is removed once it lies on (within
 *        tolerance of) a unit's boundary surface — the simplest outflow
 *        model, for boundaries meant to absorb particles that touch
 *        them exactly (e.g. a thin membrane/detector plane).
 */

namespace atlas {

/**
 * @brief `despawn()` true iff the particle is on the unit's surface
 *        within `tolerance` (`Geometry::is_on_surface`).
 */
struct SurfaceDespawn final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Vector3& particle,
            const float tolerance = 0.0f) noexcept {
        return query.is_on_surface(particle, tolerance);
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
