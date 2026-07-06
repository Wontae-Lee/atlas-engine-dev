#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

/**
 * @file collider_hit.h
 * @brief Result record of a single particle-vs-surface ray query, used by
 *        `atlas::detail::ColliderCollisionKernel` to carry the closest
 *        surface intersection found along a particle's motion segment for
 *        this timestep.
 */

namespace atlas::detail {

/**
 * @brief Closest-hit record for one particle's sweep against every unit's
 *        geometry in a single timestep.
 *
 * A default-constructed `ColliderHit` (`unit_index == -1`) represents "no
 * intersection found" — `found()` is the sentinel check the collision
 * kernel uses to fall back to unobstructed straight-line motion. `distance`
 * and `speed` are in the *local* (unit-space, moving-frame-aware) ray
 * parameterization used to find the hit; `position`/`normal` are already
 * converted back to world space by the time they are stored here (see
 * `ColliderCollisionKernel::closest_hit`).
 */
struct ColliderHit final {
    /** Local-frame ray parameter (distance along the sweep) at the hit. */
    float distance {};
    /** World-space speed of the particle relative to the surface at the
     *  moment of impact; used to convert `distance` into a time-of-impact. */
    float speed {};
    /** World-space point of intersection. */
    Vector3 position {};
    /** World-space surface normal at the intersection (pre-flip; see
     *  `ColliderProbe::flips`). */
    Vector3 normal {};
    /** Index into `ColliderProbe::units` of the hit surface, or `-1` if no
     *  surface was hit within the sweep. */
    int unit_index { -1 };

    /** @brief Whether a surface was hit (`unit_index >= 0`). */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    found() const noexcept {
        return unit_index >= 0;
    }
};

}
