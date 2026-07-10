#pragma once

#include <atlas/math/math.h>

#include <cmath>
#include <limits>

namespace atlas {

/**
 * @brief Result of a ray/surface intersection query.
 *
 * A plain aggregate returned by value from ray-tracing helpers. When
 * @ref is_intersecting is `false` the remaining fields are left at their
 * sentinel defaults and must not be read as a real hit; a caller checks the
 * flag first. Distances are measured in the ray's parameter `t`, so with a
 * unit-length ray direction they are metric distances in world units.
 */
struct HitSurface {

    /// True when a valid forward intersection was found; if false the other
    /// members are meaningless sentinels.
    bool is_intersecting = false;

    /// Ray parameter `t` at the hit (world distance for a unit direction);
    /// defaults to +max so "no hit" always loses a nearest-hit comparison.
    float distance = std::numeric_limits<float>::max();

    /// World-space intersection point; only meaningful when intersecting.
    Float3 point = Float3(0.0f, 0.0f, 0.0f);

    /// Surface normal at the hit; only meaningful when intersecting.
    Float3 normal = Float3(0.0f, 0.0f, 1.0f);
};

/**
 * @brief A half-line in world space: an origin plus a normalized direction.
 *
 * The direction invariant is enforced at construction — the two-argument
 * constructor normalizes its input — so downstream code may treat the ray
 * parameter `t` as a world-space distance. Usable from host and device.
 */
class Ray final {
public:
    /// World-space starting point of the ray.
    Float3 origin;

    /// Travel direction; kept unit-length by the constructors (see class note).
    Float3 direction;

    /**
     * @brief Construct the canonical ray at the origin pointing along +x.
     *
     * Provides a valid, unit-direction default so a default-constructed Ray is
     * immediately usable in intersection queries.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept
        : origin(0.0f, 0.0f, 0.0f)
        , direction(1.0f, 0.0f, 0.0f) { }

    /**
     * @brief Construct a ray from an origin and a (not necessarily unit) direction.
     *
     * The direction is normalized via @ref atlas::normalized_or. If the given
     * direction has (near) zero length it cannot be normalized and the fallback
     * — the zero vector — is stored instead, yielding a degenerate ray that all
     * intersection helpers treat as "never hits."
     *
     * @param origin_    World-space origin.
     * @param direction_ Desired travel direction; magnitude is ignored.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Float3& origin_, const Float3& direction_) noexcept
        : origin(origin_)
        , direction(atlas::normalized_or(
              direction_,
              Float3(0.0f, 0.0f, 0.0f))) { }

    /**
     * @brief Copy constructor (defaulted).
     * @param other Ray to copy.
     */
    Ray(const Ray& other) noexcept = default;

    /**
     * @brief Evaluate the point at parameter @p t along the ray.
     * @param t Signed distance from the origin (world units for a unit direction).
     * @return `origin + t * direction`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    point_at(const float t) const noexcept {
        return origin + t * direction;
    }
};

/**
 * @brief Signed forward distance from a ray origin to an infinite plane.
 *
 * Solves the ray/plane equation and reports the hit only when it lies ahead of
 * the origin. A ray parallel to the plane (denominator within @ref eps of zero)
 * is treated as a miss. On a hit @p distance is filled with the ray parameter
 * `t` at the intersection; on a miss @p distance is left unchanged.
 *
 * @param plane_point  Any point lying on the plane.
 * @param plane_normal Plane normal (need not be unit length).
 * @param ray          Query ray (direction assumed unit length).
 * @param distance     Out-parameter set to the hit distance only on success.
 * @return True if the ray hits the plane at a non-negative distance, else false.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
ray_plane_distance(const Float3& plane_point,
                   const Float3& plane_normal,
                   const Ray& ray,
                   float& distance) noexcept {
    const float denominator = plane_normal.dot(ray.direction);

    // Direction nearly parallel to the plane: no well-defined single hit.
    if (std::abs(denominator) <= eps) {
        return false;
    }

    distance = (plane_point - ray.origin).dot(plane_normal) / denominator;

    // Reject intersections behind the origin so this reads as a forward query.
    return distance >= 0.0f;
}

/// Convenience alias naming the single-precision ray explicitly.
using RayF = Ray;

}