#pragma once

#include <atlas/math/math.h>

#include <cmath>
#include <limits>

/**
 * @file ray.h
 * @brief The parametric ray `p(t) = origin + t * direction` and its
 *        surface-intersection result type, the shared primitive every
 *        geometric query in Atlas is built on (`Geometry`
 *        shape intersections, `AABB::trace`, `BVH` traversal, collider/
 *        sink/source geometry queries).
 */

namespace atlas {

/**
 * @brief Result of intersecting a `Ray` against a surface: whether it
 *        hit, at what parametric distance, and the hit point/normal.
 *        The default (`is_intersecting == false`) sentinel has
 *        `distance` at `float` max so a closest-hit comparison against
 *        an uninitialized result never spuriously wins.
 */
struct SurfaceRayIntersection {

    bool is_intersecting = false;

    float distance = std::numeric_limits<float>::max();

    Float3 point = Float3(0.0f, 0.0f, 0.0f);

    Float3 normal = Float3(0.0f, 0.0f, 1.0f);
};

/**
 * @brief A parametric ray `p(t) = origin + t * direction`, `direction`
 *        always stored normalized (a zero-length input direction
 *        normalizes to the zero vector via `normalized_or`, rather than
 *        producing `NaN`, so degenerate rays fail intersection tests
 *        cleanly instead of propagating `NaN`).
 */
class Ray final {
public:
    Float3 origin;

    Float3 direction;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept
        : origin(0.0f, 0.0f, 0.0f)
        , direction(1.0f, 0.0f, 0.0f) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Float3& origin_, const Float3& direction_) noexcept
        : origin(origin_)
        , direction(atlas::normalized_or(
              direction_,
              Float3(0.0f, 0.0f, 0.0f))) { }

    Ray(const Ray& other) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    point_at(const float t) const noexcept {
        return origin + t * direction;
    }
};

/**
 * @brief Forward ray-plane intersection distance. Solves
 *        `plane_normal . (p(t) - plane_point) = 0` for `t`, writing it to
 *        `distance`. Returns `false` for a grazing ray (denominator
 *        within `eps` of zero, no unique intersection) or a hit behind
 *        the origin (`distance < 0`); the caller supplies the plane by
 *        any point on it and its normal. Callers needing the
 *        parallel-and-coincident case handle it separately.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
ray_plane_distance(const Float3& plane_point,
                   const Float3& plane_normal,
                   const Ray& ray,
                   float& distance) noexcept {
    const float denominator = plane_normal.dot(ray.direction);

    if (std::abs(denominator) <= eps) {
        return false;
    }

    distance = (plane_point - ray.origin).dot(plane_normal) / denominator;

    return distance >= 0.0f;
}

using HitSurface = SurfaceRayIntersection;

using RayF = Ray;

}
