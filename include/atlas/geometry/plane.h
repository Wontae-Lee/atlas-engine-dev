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
 * @brief Infinite plane primitive in Hessian normal form.
 *
 * The plane is the locus `normal . p + offset == 0`. For the distance and
 * classification queries to be metric the @ref normal is assumed unit length;
 * the builder does not renormalize it, so callers own that invariant. The
 * half-space `normal . p + offset <= 0` (the side the normal points away from) is
 * treated as the plane's "inside".
 *
 * One leaf of the `Geometry` tagged union, trivially copyable, every query is
 * `ATLAS_ALL_DEVICE`.
 */
class Plane final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Plane.
    class Builder;

public:
    Float3 normal = Float3(0.0f, 0.0f, 1.0f); ///< Plane normal; assumed unit length.

    float offset = 0.0f; ///< Signed plane offset `d` in `normal . p + d = 0`.

    /**
     * @brief Constructs the default `z = 0` plane facing `+z`.
     */
    Plane() noexcept = default;

    /**
     * @brief Constructs a plane from a normal and a scalar offset.
     *
     * @param normal_ Plane normal (assumed unit length for metric queries).
     * @param offset_ Signed offset `d` such that the plane is `normal . p + d = 0`.
     * @note Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Plane(const Float3& normal_, float offset_) noexcept
        : normal(normal_)
        , offset(offset_) { }

    /**
     * @brief Constructs a plane through a point with a given normal.
     *
     * The offset is derived so the plane passes through @p point:
     * `offset = -(normal . point)`.
     *
     * @param point A point on the plane.
     * @param normal_ Plane normal (assumed unit length for metric queries).
     * @note Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Plane(const Float3& point, const Float3& normal_) noexcept
        : normal(normal_)
        , offset(-(normal_.dot(point))) { }

    Plane(const Plane& other) noexcept = default; ///< Trivial copy.
    Plane(Plane&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Plane&
    operator=(const Plane& other) noexcept = default;
    /// Trivial move assignment.
    Plane&
    operator=(Plane&& other) noexcept = default;

    ~Plane() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Plane.
     * @return A default-initialized `Plane::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Projects @p p orthogonally onto the plane.
     * @param p Query point in world space.
     * @return The foot of the perpendicular from @p p to the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const float sdev = normal.dot(p) + offset;

        return p - sdev * normal;
    }

    /**
     * @brief Returns the plane normal, independent of the query point.
     * @return @ref normal verbatim (not renormalized).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return normal;
    }

    /**
     * @brief Signed distance from @p p to the plane.
     * @param p Query point in world space.
     * @return `normal . p + offset`: positive on the normal-facing side, negative
     *         on the far side. Metric only when @ref normal is unit length.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        return normal.dot(p) + offset;
    }

    /**
     * @brief Tests whether @p p is in the plane's interior half-space.
     * @param p Query point in world space.
     * @param tolerance Signed shift of the classification threshold; positive
     *        admits points slightly on the front side, negative excludes points
     *        just behind the plane.
     * @return `true` when `normal . p + offset <= tolerance`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return normal.dot(p) + offset <= tolerance;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the plane.
     * @param p Query point in world space.
     * @param tolerance Non-negative slab half-thickness; a negative value returns
     *        `false`.
     * @return `true` when `|normal . p + offset| <= tolerance`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const float distance = normal.dot(p) + offset;
        return distance >= -tolerance && distance <= tolerance;
    }

    /**
     * @brief Nominal centroid of the (infinite) plane.
     * @return The origin. A plane has no finite centroid; the origin is returned
     *         as a stable placeholder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return Float3(0.0f, 0.0f, 0.0f);
    }

    /**
     * @brief Axis-aligned bounding box of the plane.
     * @return An `AABB` spanning the entire representable float range, since the
     *         plane is unbounded.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float lo = std::numeric_limits<float>::lowest();
        const float hi = std::numeric_limits<float>::max();

        return AABB(
            Float3(lo, lo, lo),
            Float3(hi, hi, hi));
    }

    /**
     * @brief Reports whether the plane parameters are well-formed.
     * @return `true` when the normal is finite and nonzero and the offset is
     *         finite. Does not require the normal to be unit length.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        const float n2 = normal.length_squared();
        return atlas::isfinite(normal)
            && (n2 > 0.0f)
            && atlas::isfinite(offset);
    }

    /**
     * @brief Intersects a ray with the plane.
     *
     * When the ray is parallel to the plane (`|normal . dir| <= eps`) it hits only
     * if the origin already lies on the plane, reported at distance zero;
     * otherwise the single forward crossing is returned.
     *
     * @param ray Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` for a parallel miss or a
     *         crossing behind the origin (`t < 0`). The reported normal is the
     *         plane's own normal, not flipped toward the ray.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        const float denom = normal.dot(ray.direction);

        const float numer = -(normal.dot(ray.origin) + offset);

        if (std::abs(denom) <= eps) {

            // Parallel ray: only a hit if the origin sits exactly on the plane.
            if (numer != 0.0f) {
                return result;
            }

            result.is_intersecting = true;
            result.distance        = 0.0f;
            result.point           = ray.origin;
            result.normal          = normal;

            return result;
        }

        const float t = numer / denom;

        if (t < 0.0f) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = ray.point_at(t);
        result.normal          = normal;

        return result;
    }
};

/**
 * @brief Host-side builder for `Plane` with validation.
 *
 * Offers several ways to specify the plane (normal + offset, or point + normal)
 * and, on `build()`, throws if the result is invalid. Runs on the host only.
 */
class Plane::Builder final {
public:
    Builder() = default; ///< Starts from the default `z = 0` plane.

    /**
     * @brief Validates the accumulated parameters and constructs a `Plane`.
     * @return The constructed plane.
     * @throws std::runtime_error when the parameters are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST Plane
    build() const;

    /**
     * @brief Builds the plane and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Plane>` owning the constructed plane.
     * @throws std::runtime_error when the parameters are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Plane>
    make_host_shared() const;

    /**
     * @brief Sets the plane normal.
     * @param normal_ Plane normal (should be unit length for metric queries).
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

    /**
     * @brief Sets the scalar offset.
     * @param offset_ Signed offset `d`.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_offset(float offset_) noexcept;

    /**
     * @brief Sets normal and offset together.
     * @param normal_ Plane normal.
     * @param offset_ Signed offset `d`.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_normal_offset(const Float3& normal_, float offset_) noexcept;

    /**
     * @brief Sets the plane from a point on it and a normal.
     *
     * Derives `offset = -(normal . point)`.
     *
     * @param point A point lying on the plane.
     * @param normal_ Plane normal.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_point_normal(const Float3& point, const Float3& normal_) noexcept;

private:
    /**
     * @brief Throws if the accumulated parameters do not form a valid plane.
     * @throws std::runtime_error on invalid parameters.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _normal = Float3(0.0f, 0.0f, 1.0f); ///< Pending normal.

    float _offset = 0.0f; ///< Pending offset.
};

/// Owning host handle to a `Plane`.
using PlaneHostPtr = atlas::host_shared_ptr<Plane>;

/// Owning device handle to a `Plane`.
using PlaneDevicePtr = atlas::device_shared_ptr<Plane>;

}
