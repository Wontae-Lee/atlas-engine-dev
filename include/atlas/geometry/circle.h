#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

/**
 * @brief Flat circular disk primitive in 3D.
 *
 * A `Circle` is the filled disk of the given @ref radius lying in the plane
 * through @ref center with the given @ref normal. It is a zero-thickness surface,
 * so its "solid" queries treat the normal as picking a front/back side rather
 * than enclosing a volume. It is one leaf of the `Geometry` tagged union,
 * trivially copyable, and every query is `ATLAS_ALL_DEVICE`.
 *
 * The stored @ref normal need not be unit length; queries normalize it on the
 * fly and fall back to `+z` when it is degenerate.
 */
class Circle final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Circle.
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f); ///< Disk center in world space.
    Float3 normal = Float3(0.0f, 0.0f, 1.0f); ///< Plane normal; normalized on use, `+z` fallback.
    float radius  = 1.0f;                     ///< Disk radius; must be positive to be valid.

    /**
     * @brief Constructs a unit disk in the z=0 plane facing `+z`.
     */
    Circle() noexcept = default;

    /**
     * @brief Constructs a disk from center, normal, and radius.
     *
     * @param center_ Disk center.
     * @param normal_ Plane normal (any nonzero length; normalized on use).
     * @param radius_ Disk radius; should be positive.
     * @note Callable on host and device; parameters are stored verbatim.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Circle(const Float3& center_, const Float3& normal_, const float radius_) noexcept
        : center(center_)
        , normal(normal_)
        , radius(radius_) { }

    Circle(const Circle& other) noexcept = default; ///< Trivial copy.
    Circle(Circle&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Circle&
    operator=(const Circle& other) noexcept = default;
    /// Trivial move assignment.
    Circle&
    operator=(Circle&& other) noexcept = default;

    ~Circle() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Circle.
     * @return A default-initialized `Circle::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the disk nearest to @p p.
     *
     * @p p is projected onto the disk plane; if the projection lies within the
     * radius it is returned directly (a point on the disk face), otherwise the
     * projection is pushed to the rim.
     *
     * @param p Query point in world space.
     * @return The nearest point on the disk. Degenerate inputs are handled
     *         defensively: a zero normal or non-positive radius returns @p p
     *         unchanged, and a projection landing on the center returns an
     *         arbitrary rim point (`center + radius * x`).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const float n2 = normal.length_squared();

        if (n2 <= 0.0f || radius <= 0.0f) {
            return p;
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 offset        = p - center;
        const float plane_distance = offset.dot(n);

        // Component of the offset lying in the disk plane.
        const Float3 planar     = offset - n * plane_distance;
        const float planar_len2 = planar.length_squared();
        const float rr          = radius * radius;

        if (planar_len2 <= rr) {
            // Projection is on the face: drop the out-of-plane component only.
            return p - n * plane_distance;
        }

        if (planar_len2 <= std::numeric_limits<float>::epsilon()) {
            // Degenerate: projection coincides with the center, so any rim point
            // is equally close; pick one deterministically.
            return center + Float3(radius, 0.0f, 0.0f);
        }

        const float planar_len = atlas::sqrt_nonnegative(planar_len2);
        return center + planar * (radius / planar_len);
    }

    /**
     * @brief Returns the disk's unit face normal, independent of the query point.
     * @return The normalized @ref normal, or `+z` when it is degenerate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));
    }

    /**
     * @brief Signed distance from @p p to the disk.
     *
     * The magnitude is the Euclidean distance to the nearest point on the disk
     * (face or rim); the sign is positive on the side the normal points toward and
     * negative on the far side.
     *
     * @param p Query point in world space.
     * @return The signed distance.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 cp       = closest_point(p);
        const Float3 nn       = closest_normal(p);
        const float magnitude = (p - cp).length();

        const float side = (p - center).dot(nn);

        return (side >= 0.0f) ? magnitude : -magnitude;
    }

    /**
     * @brief Tests whether @p p lies on the disk's interior side.
     *
     * Because the disk has no volume, "inside" is a one-sided classification: the
     * back side of the plane (where the signed plane distance is negative) counts
     * as interior, and @p tolerance controls a shell around the disk surface. When
     * @p p is on or in front of the plane, it is accepted only if it is within
     * @p tolerance of the disk. When @p p is behind the plane it is accepted for
     * any non-negative tolerance, and for negative tolerance only once it is at
     * least `|tolerance|` away from the disk (an eroded interior).
     *
     * @param p Query point in world space.
     * @param tolerance Signed shell half-thickness; interpreted as described above.
     * @return `true` when @p p is classified as inside. Always `false` for a
     *         non-positive radius or a degenerate normal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f)) {
            return false;
        }

        float plane_distance;
        float distance2;

        if (!plane_and_radial(p, plane_distance, distance2)) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;

        if (!(plane_distance < 0.0f)) {
            // On or in front of the plane: inside only within the tolerance shell.
            return tolerance >= 0.0f && distance2 <= tolerance2;
        }

        // Behind the plane: interior, unless a negative tolerance erodes the shell.
        return tolerance >= 0.0f || distance2 >= tolerance2;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the disk surface.
     *
     * Uses the squared distance to the nearest disk point (combining out-of-plane
     * offset and any radial overshoot past the rim).
     *
     * @param p Query point in world space.
     * @param tolerance Non-negative shell half-thickness; a negative value returns
     *        `false`.
     * @return `true` when the squared distance to the disk is at most
     *         `tolerance^2`. Always `false` for a non-positive radius.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f) || tolerance < 0.0f) {
            return false;
        }

        float plane_distance;
        float distance2;

        if (!plane_and_radial(p, plane_distance, distance2)) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;

        return distance2 <= tolerance2;
    }

    /**
     * @brief Geometric center of the disk.
     * @return @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    /**
     * @brief Tight axis-aligned bounding box of the (possibly tilted) disk.
     *
     * The per-axis half-extent of a unit-normal disk is `radius * sqrt(1 - n_i^2)`,
     * i.e. the projected radius onto each axis.
     *
     * @return The bounding box; a degenerate disk (zero normal or non-positive
     *         radius) collapses to the single point @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float n2 = normal.length_squared();

        if (n2 <= 0.0f || radius <= 0.0f) {
            return AABB(center, center);
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 extent(
            radius * atlas::sqrt_nonnegative(1.0f - n.x * n.x),
            radius * atlas::sqrt_nonnegative(1.0f - n.y * n.y),
            radius * atlas::sqrt_nonnegative(1.0f - n.z * n.z));

        return AABB(center - extent, center + extent);
    }

    /**
     * @brief Reports whether the disk parameters are well-formed.
     * @return `true` when center, normal, and radius are finite, the normal is
     *         nonzero, and the radius is positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(center)
            && atlas::isfinite(normal)
            && normal.length_squared() > 0.0f
            && atlas::isfinite(radius)
            && radius > 0.0f;
    }

    /**
     * @brief Intersects a ray with the disk.
     *
     * When the ray is (nearly) parallel to the disk plane, the disk is hit only if
     * the ray origin already lies in the plane and within the radius, reported as a
     * grazing hit at distance zero. Otherwise the ray-plane crossing is computed
     * and accepted when the hit point falls inside the rim.
     *
     * @param ray Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` on a miss or when the
     *         disk is invalid. The reported normal is the disk's own face normal,
     *         not flipped toward the ray.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        if (!is_valid()) {
            return result;
        }

        const Float3 nn   = closest_normal(ray.origin);
        const float denom = nn.dot(ray.direction);

        if (std::abs(denom) <= eps) {
            // Ray parallel to the plane: a hit is only possible if the origin
            // already lies in the plane and inside the disk.
            const float plane_distance = (ray.origin - center).dot(nn);
            if (std::abs(plane_distance) > eps) {
                return result;
            }

            if ((ray.origin - center).length_squared() > radius * radius) {
                return result;
            }

            result.is_intersecting = true;
            result.distance        = 0.0f;
            result.point           = ray.origin;
            result.normal          = nn;

            return result;
        }

        float t;

        if (!atlas::ray_plane_distance(center, nn, ray, t)) {
            return result;
        }

        const Float3 hit_point = ray.point_at(t);

        if ((hit_point - center).length_squared() > radius * radius) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = hit_point;
        result.normal          = nn;

        return result;
    }

private:
    /**
     * @brief Decomposes @p p into its signed plane distance and squared distance
     *        to the disk.
     *
     * @param p Query point in world space.
     * @param[out] plane_distance Signed distance from @p p to the disk plane along
     *        the unit normal.
     * @param[out] distance2 Squared distance from @p p to the nearest disk point:
     *        `plane_distance^2` plus, when @p p projects beyond the rim, the
     *        squared radial overshoot.
     * @return `false` (leaving the outputs untouched) when the normal is
     *         degenerate; `true` otherwise.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    plane_and_radial(const Float3& p,
                     float& plane_distance,
                     float& distance2) const noexcept {
        const float n2 = normal.length_squared();

        if (!(n2 > 0.0f)) {
            return false;
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 offset     = p - center;
        const Float3 planar     = offset - n * offset.dot(n);
        const float planar_len2 = planar.length_squared();
        const float rr          = radius * radius;

        plane_distance = offset.dot(n);
        distance2      = plane_distance * plane_distance;

        if (planar_len2 > rr) {
            // Outside the rim: add the squared overshoot past the radius.
            const float radial_distance = atlas::sqrt_nonnegative(planar_len2) - radius;
            distance2 += radial_distance * radial_distance;
        }

        return true;
    }
};

/**
 * @brief Host-side builder for `Circle` with validation.
 *
 * Accumulates parameters through `with_*` setters and, on `build()`, throws if the
 * resulting disk is invalid. Runs on the host only.
 */
class Circle::Builder final {
public:
    Builder() = default; ///< Starts from the default unit disk in the z=0 plane.

    /**
     * @brief Validates the accumulated parameters and constructs a `Circle`.
     * @return The constructed disk.
     * @throws std::runtime_error when the parameters are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST Circle
    build() const;

    /**
     * @brief Builds the disk and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Circle>` owning the constructed disk.
     * @throws std::runtime_error when the parameters are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Circle>
    make_host_shared() const;

    /**
     * @brief Sets the disk center.
     * @param center_ Disk center.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_center(const Float3& center_) noexcept;

    /**
     * @brief Sets the plane normal.
     * @param normal_ Nonzero normal; normalized on use.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

    /**
     * @brief Sets the disk radius.
     * @param radius_ Positive radius.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_radius(float radius_) noexcept;

private:
    /**
     * @brief Throws if the accumulated parameters do not form a valid disk.
     * @throws std::runtime_error on invalid parameters.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f); ///< Pending center.
    Float3 _normal = Float3(0.0f, 0.0f, 1.0f); ///< Pending normal.
    float _radius  = 1.0f;                     ///< Pending radius.
};

/// Owning host handle to a `Circle`.
using CircleHostPtr = atlas::host_shared_ptr<Circle>;

/// Owning device handle to a `Circle`.
using CircleDevicePtr = atlas::device_shared_ptr<Circle>;

}
