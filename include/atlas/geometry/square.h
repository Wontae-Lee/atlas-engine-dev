#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

/**
 * @brief Flat square patch primitive in 3D.
 *
 * The square is the axis-of-normal-oriented planar patch of the given
 * @ref side_length, centered at @ref center and lying in the plane with the given
 * @ref normal. Its in-plane axes are derived on the fly from the normal via
 * `atlas::orthonormal_basis`, so the square's rotation about its own normal is not
 * independently controllable — it is whatever the basis routine produces. Being a
 * zero-thickness patch, its "inside" test coincides with its surface test.
 *
 * One leaf of the `Geometry` tagged union, trivially copyable, every query is
 * `ATLAS_ALL_DEVICE`.
 */
class Square final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Square.
    class Builder;

public:
    Float3 center     = Float3(0.0f, 0.0f, 0.0f); ///< Patch center in world space.
    Float3 normal     = Float3(0.0f, 0.0f, 1.0f); ///< Plane normal; normalized on use.
    float side_length = 1.0f;                     ///< Edge length; must be positive to be valid.

    /**
     * @brief Constructs a unit square in the z=0 plane facing `+z`.
     */
    Square() noexcept = default;

    /**
     * @brief Constructs a square from center, normal, and side length.
     * @param center_ Patch center.
     * @param normal_ Plane normal (nonzero; normalized on use).
     * @param side_length_ Edge length; should be positive.
     * @note Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Square(const Float3& center_,
           const Float3& normal_,
           const float side_length_) noexcept
        : center(center_)
        , normal(normal_)
        , side_length(side_length_) { }

    Square(const Square& other) noexcept = default; ///< Trivial copy.
    Square(Square&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Square&
    operator=(const Square& other) noexcept = default;
    /// Trivial move assignment.
    Square&
    operator=(Square&& other) noexcept = default;

    ~Square() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Square.
     * @return A default-initialized `Square::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the square nearest to @p p.
     *
     * @p p is projected onto the patch plane and the in-plane coordinates are
     * clamped to `[-side/2, side/2]` along each of the patch axes.
     *
     * @param p Query point in world space.
     * @return The nearest point on the square, or @p p unchanged when the square is
     *         degenerate (non-positive side length or unusable normal).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        if (!(side_length > 0.0f)) {
            return p;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return p;
        }

        const float half_side        = side_length * 0.5f;
        const Float3 center_to_point = p - center;

        const float signed_plane_offset = center_to_point.dot(unit_normal);
        const Float3 projected_point    = p - unit_normal * signed_plane_offset;

        const Float3 planar_offset = projected_point - center;

        // Clamp the in-plane coordinates to the square's extent.
        const float u = std::clamp(planar_offset.dot(tangent), -half_side, half_side);
        const float v = std::clamp(planar_offset.dot(bitangent), -half_side, half_side);

        return center + tangent * u + bitangent * v;
    }

    /**
     * @brief Returns the patch's unit face normal, independent of the query point.
     * @return The normalized @ref normal, or `+z` when it is degenerate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));
    }

    /**
     * @brief Signed distance from @p p to the square.
     * @param p Query point in world space.
     * @return Distance to the nearest patch point, signed positive on the
     *         normal-facing side and negative on the far side.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 projected_closest_point = closest_point(p);

        const Float3 unit_normal = closest_normal(p);

        const float distance_magnitude = (p - projected_closest_point).length();

        const float sign_test = (p - center).dot(unit_normal);

        return (sign_test >= 0.0f) ? distance_magnitude : -distance_magnitude;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the square patch.
     *
     * Because the patch has no volume, this is a slab test: @p p must be within
     * @p tolerance of the plane and within `side/2 + tolerance` of the center along
     * each in-plane axis.
     *
     * @param p Query point in world space.
     * @param tolerance Shell half-thickness applied to both the plane offset and
     *        the in-plane extent.
     * @return `true` when @p p is within the thickened patch. Always `false` for a
     *         non-positive side length or an unusable normal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(side_length > 0.0f)) {
            return false;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return false;
        }

        const float half_side        = side_length * 0.5f;
        const Float3 center_to_point = p - center;

        const float signed_plane_offset = center_to_point.dot(unit_normal);

        const float u = center_to_point.dot(tangent);
        const float v = center_to_point.dot(bitangent);

        return std::abs(signed_plane_offset) <= tolerance
            && std::abs(u) <= half_side + tolerance
            && std::abs(v) <= half_side + tolerance;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the square surface.
     * @param p Query point in world space.
     * @param tolerance Shell half-thickness.
     * @return Identical to `is_inside` — for a zero-thickness patch the interior
     *         and the surface coincide.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return is_inside(p, tolerance);
    }

    /**
     * @brief Geometric center of the square.
     * @return @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    /**
     * @brief Axis-aligned bounding box of the (possibly tilted) square.
     *
     * The half-extent is `(|tangent| + |bitangent|) * side/2` componentwise, the
     * world-axis projection of the two in-plane half-edges.
     *
     * @return The bounding box; a degenerate square collapses to the point
     *         @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        if (!(side_length > 0.0f)) {
            return AABB(center, center);
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return AABB(center, center);
        }

        const float half_side = side_length * 0.5f;

        const Float3 extent = (atlas::abs(tangent) + atlas::abs(bitangent)) * half_side;

        return AABB(center - extent, center + extent);
    }

    /**
     * @brief Reports whether the square parameters are well-formed.
     * @return `true` when center, normal, and side length are finite, the normal is
     *         nonzero, and the side length is positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(center)
            && atlas::isfinite(normal)
            && normal.length_squared() > 0.0f
            && atlas::isfinite(side_length)
            && side_length > 0.0f;
    }

    /**
     * @brief Intersects a ray with the square.
     *
     * Computes the ray-plane crossing then accepts it only when the hit point lies
     * within the square's extent along both in-plane axes (with an epsilon slack).
     *
     * @param ray Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` when the square is
     *         invalid, the ray is parallel/behind, or the hit lands outside the
     *         patch. The reported normal is the patch's own face normal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface hit {};

        if (!is_valid()) {
            return hit;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return hit;
        }

        float distance;

        if (!atlas::ray_plane_distance(center, unit_normal, ray, distance)) {
            return hit;
        }

        const float epsilon = std::numeric_limits<float>::epsilon();

        const Float3 hit_point = ray.point_at(distance);

        const Float3 center_to_hit = hit_point - center;
        const float u              = center_to_hit.dot(tangent);
        const float v              = center_to_hit.dot(bitangent);
        const float half_side      = side_length * 0.5f;

        // Reject hits that fall outside the square's finite extent.
        if (std::abs(u) > half_side + epsilon || std::abs(v) > half_side + epsilon) {
            return hit;
        }

        hit.is_intersecting = true;
        hit.distance        = distance;
        hit.point           = hit_point;
        hit.normal          = unit_normal;

        return hit;
    }

private:
    /**
     * @brief Builds the patch's orthonormal frame from a normal.
     *
     * Thin wrapper over `atlas::orthonormal_basis`; exists so the query methods can
     * share one construction point for the in-plane axes.
     *
     * @param input_normal The (possibly non-unit) plane normal.
     * @param[out] unit_normal Normalized plane normal.
     * @param[out] tangent First in-plane unit axis.
     * @param[out] bitangent Second in-plane unit axis.
     * @return `false` when the normal is too small to build a stable frame; `true`
     *         otherwise.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    build_basis(const Float3& input_normal,
                Float3& unit_normal,
                Float3& tangent,
                Float3& bitangent) const noexcept {
        return atlas::orthonormal_basis(input_normal, unit_normal, tangent, bitangent);
    }
};

/**
 * @brief Host-side builder for `Square` with validation.
 *
 * Accumulates parameters through `with_*` setters and, on `build()`, throws a
 * message that names the first failing invariant. Runs on the host only.
 */
class Square::Builder final {
public:
    Builder() = default; ///< Starts from the unit square in the z=0 plane.

    /**
     * @brief Validates the accumulated parameters and constructs a `Square`.
     * @return The constructed square.
     * @throws std::runtime_error with a specific message when a parameter is
     *         non-finite, the normal is zero, or the side length is non-positive.
     */
    ATLAS_NODISCARD ATLAS_HOST Square
    build() const;

    /**
     * @brief Builds the square and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Square>` owning the constructed square.
     * @throws std::runtime_error when the parameters are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Square>
    make_host_shared() const;

    /**
     * @brief Sets the patch center.
     * @param center_ Patch center.
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
     * @brief Sets the edge length.
     * @param side_length_ Positive edge length.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_side_length(float side_length_) noexcept;

private:
    /**
     * @brief Throws a specific error for the first violated invariant.
     * @throws std::runtime_error when a parameter is non-finite, the normal is
     *         zero, or the side length is non-positive.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center     = Float3(0.0f, 0.0f, 0.0f); ///< Pending center.
    Float3 _normal     = Float3(0.0f, 0.0f, 1.0f); ///< Pending normal.
    float _side_length = 1.0f;                     ///< Pending side length.
};

/// Owning host handle to a `Square`.
using SquareHostPtr = atlas::host_shared_ptr<Square>;

/// Owning device handle to a `Square`.
using SquareDevicePtr = atlas::device_shared_ptr<Square>;

}
