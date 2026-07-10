#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cstddef>
#include <limits>

namespace atlas {

/**
 * @brief Solid sphere primitive.
 *
 * The sphere is the ball of the given @ref radius about @ref center. One leaf of
 * the `Geometry` tagged union, trivially copyable, every query is
 * `ATLAS_ALL_DEVICE`.
 */
class Sphere final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Sphere.
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f); ///< Sphere center in world space.

    float radius = 1.0f; ///< Sphere radius; must be positive to be valid.

    /**
     * @brief Constructs a unit sphere at the origin.
     */
    Sphere() noexcept = default;

    /**
     * @brief Constructs a sphere from center and radius.
     * @param center_ Sphere center.
     * @param radius_ Sphere radius; should be positive.
     * @note Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sphere(const Float3& center_, const float radius_) noexcept
        : center(center_)
        , radius(radius_) { }

    Sphere(const Sphere& other) noexcept = default; ///< Trivial copy.
    Sphere(Sphere&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Sphere&
    operator=(const Sphere& other) noexcept = default;
    /// Trivial move assignment.
    Sphere&
    operator=(Sphere&& other) noexcept = default;

    ~Sphere() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Sphere.
     * @return A default-initialized `Sphere::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the sphere surface nearest to @p p.
     * @param p Query point in world space.
     * @return `center + radius * dir`, where `dir` is the unit direction from the
     *         center to @p p, or the fallback `+x` when @p p coincides with the
     *         center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const Float3 v = p - center;
        const float e  = std::numeric_limits<float>::epsilon();

        const Float3 direction = atlas::normalized_or(
            v,
            Float3(1.0f, 0.0f, 0.0f),
            e);
        return center + direction * radius;
    }

    /**
     * @brief Returns the outward unit normal at the surface point nearest to @p p.
     * @param p Query point in world space.
     * @return The unit direction from the center to @p p, or `+x` when @p p is at
     *         the center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const Float3 v = p - center;
        const float e  = std::numeric_limits<float>::epsilon();

        return atlas::normalized_or(
            v,
            Float3(1.0f, 0.0f, 0.0f),
            e);
    }

    /**
     * @brief Signed distance from @p p to the sphere surface.
     * @param p Query point in world space.
     * @return `|p - center| - radius`: negative inside, zero on the surface,
     *         positive outside.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        return (p - center).length() - radius;
    }

    /**
     * @brief Tests whether @p p lies inside the sphere, optionally dilated.
     * @param p Query point in world space.
     * @param tolerance Signed radius adjustment; positive grows the ball, negative
     *        shrinks it. An adjusted radius below zero always returns `false`.
     * @return `true` when @p p is within the tolerance-adjusted radius.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        const float expanded_radius = radius + tolerance;

        if (expanded_radius < 0.0f) {
            return false;
        }

        return (p - center).length_squared() <= expanded_radius * expanded_radius;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the sphere surface.
     * @param p Query point in world space.
     * @param tolerance Non-negative shell half-thickness; a negative value returns
     *        `false`.
     * @return `true` when @p p lies in the spherical shell between
     *         `radius - tolerance` (clamped at zero) and `radius + tolerance`.
     *         Always `false` for a non-positive radius.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f) || tolerance < 0.0f) {
            return false;
        }

        const float outer_radius = radius + tolerance;
        const float inner_radius = (radius > tolerance) ? radius - tolerance : 0.0f;
        const float d2           = (p - center).length_squared();

        return d2 >= inner_radius * inner_radius
            && d2 <= outer_radius * outer_radius;
    }

    /**
     * @brief Geometric center of the sphere.
     * @return @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    /**
     * @brief Axis-aligned bounding box of the sphere.
     * @return The cube of half-extent @ref radius about the center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const Float3 dr(radius, radius, radius);

        return AABB(center - dr, center + dr);
    }

    /**
     * @brief Reports whether the sphere parameters are well-formed.
     * @return `true` when @ref radius is positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return radius > 0.0f;
    }

    /**
     * @brief Intersects a ray with the sphere.
     *
     * Solves the standard ray-sphere quadratic and returns the nearest
     * non-negative root, so a ray originating inside the sphere reports the forward
     * exit hit.
     *
     * @param ray Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` on a miss or when both
     *         roots are behind the origin. The normal points outward from the
     *         center through the hit point.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        const Float3 oc = ray.origin - center;

        const float a  = ray.direction.length_squared();
        const float b  = 2.0f * oc.dot(ray.direction);
        const float cc = oc.length_squared() - radius * radius;

        float t0;
        float t1;

        if (!atlas::solve_quadratic(a, b, cc, t0, t1)) {
            return result;
        }

        if (t0 < 0.0f && t1 < 0.0f) {
            return result;
        }

        // Pick the nearest non-negative root (t0 <= t1 from solve_quadratic).
        float t = std::numeric_limits<float>::infinity();

        if (t0 >= 0.0f) {
            t = t0;
        }

        if (t1 >= 0.0f && t1 < t) {
            t = t1;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = ray.point_at(t);

        result.normal = atlas::normalized_or(
            result.point - center,
            Float3(1.0f, 0.0f, 0.0f));

        return result;
    }
};

/**
 * @brief Host-side builder for `Sphere` with validation.
 *
 * Accumulates center and radius through `with_*` setters and, on `build()`, throws
 * if the radius is non-positive. Runs on the host only.
 */
class Sphere::Builder final {
public:
    Builder() = default; ///< Starts from the unit sphere at the origin.

    /**
     * @brief Validates the accumulated parameters and constructs a `Sphere`.
     * @return The constructed sphere.
     * @throws std::runtime_error when the radius is non-positive.
     */
    ATLAS_NODISCARD ATLAS_HOST Sphere
    build() const;

    /**
     * @brief Builds the sphere and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Sphere>` owning the constructed sphere.
     * @throws std::runtime_error when the radius is non-positive.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Sphere>
    make_host_shared() const;

    /**
     * @brief Sets the sphere center.
     * @param c Sphere center.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_center(const Float3& c) noexcept;

    /**
     * @brief Sets the sphere radius.
     * @param r Positive radius.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_radius(float r) noexcept;

private:
    /**
     * @brief Throws if the accumulated radius is not positive.
     * @throws std::runtime_error on invalid parameters.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f); ///< Pending center.

    float _radius = 1.0f; ///< Pending radius.
};

/// Owning host handle to a `Sphere`.
using SphereHostPtr = atlas::host_shared_ptr<Sphere>;

/// Owning device handle to a `Sphere`.
using SphereDevicePtr = atlas::device_shared_ptr<Sphere>;

}
