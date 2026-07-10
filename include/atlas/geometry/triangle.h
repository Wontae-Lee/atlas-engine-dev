#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <cstddef>
#include <optional>

namespace atlas {

/**
 * @brief Single triangle primitive with a stored face normal.
 *
 * The triangle is the patch spanned by vertices @ref a, @ref b, @ref c, oriented
 * by @ref normal (which the querying methods treat as the face normal rather than
 * recomputing it per call). Like the other planar primitives it is a
 * zero-thickness surface, so its "inside" test is a one-sided half-space
 * classification rather than a volume test.
 *
 * One leaf of the `Geometry` tagged union, trivially copyable, every query is
 * `ATLAS_ALL_DEVICE`.
 */
class Triangle final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Triangle.
    class Builder;

public:
    Float3 a = Float3(0.0f, 0.0f, 0.0f); ///< First vertex.

    Float3 b = Float3(0.0f, 0.0f, 0.0f); ///< Second vertex.

    Float3 c = Float3(0.0f, 0.0f, 0.0f); ///< Third vertex.

    /**
     * Companion face-normal slot, filled by the triangle-mesh/BVH traversal path
     * (which assigns it the container's stored area normal alongside @ref normal).
     * The triangle's own member functions read @ref normal, not this field; it
     * defaults to `+z` and is left untouched by the three-vertex constructor.
     */
    Float3 n = Float3(0.0f, 0.0f, 1.0f);

    Float3 normal = Float3(0.0f, 0.0f, 1.0f); ///< Unit face normal used by all queries.

    /**
     * @brief Constructs a degenerate triangle at the origin facing `+z`.
     */
    Triangle() noexcept = default;

    /**
     * @brief Constructs a triangle from three vertices and derives the normal.
     *
     * @param a_ First vertex.
     * @param b_ Second vertex.
     * @param c_ Third vertex.
     * @note The normal is set to the normalized `cross(b - a, c - a)`, falling back
     *       to the zero vector for a degenerate (collinear) triangle. The @ref n
     *       companion slot is not written here. Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Triangle(const Float3& a_, const Float3& b_, const Float3& c_) noexcept
        : a(a_)
        , b(b_)
        , c(c_) {
        normal = atlas::normalized_or(
            atlas::cross(b - a, c - a),
            Float3(0.0f, 0.0f, 0.0f));
    }

    Triangle(const Triangle& other) noexcept = default; ///< Trivial copy.
    Triangle(Triangle&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Triangle&
    operator=(const Triangle& other) noexcept = default;
    /// Trivial move assignment.
    Triangle&
    operator=(Triangle&& other) noexcept = default;

    ~Triangle() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Triangle.
     * @return A default-initialized `Triangle::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the triangle nearest to @p p.
     *
     * Uses the standard Voronoi-region decomposition (Ericson): the result is a
     * vertex, an edge point, or an interior face point depending on which feature's
     * region contains the projection of @p p.
     *
     * @param p Query point in world space.
     * @return The closest point on the triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const Float3 v0 = a;
        const Float3 v1 = b;
        const Float3 v2 = c;

        const Float3 ab = v1 - v0;
        const Float3 ac = v2 - v0;
        const Float3 ap = p - v0;

        const float d1 = ab.dot(ap);
        const float d2 = ac.dot(ap);

        // Vertex region of v0.
        if (d1 <= 0.0f && d2 <= 0.0f) {
            return v0;
        }

        const Float3 bp = p - v1;
        const float d3  = ab.dot(bp);
        const float d4  = ac.dot(bp);

        // Vertex region of v1.
        if (d3 >= 0.0f && d4 <= d3) {
            return v1;
        }

        const float vc = d1 * d4 - d3 * d2;

        // Edge region of ab.
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float vv = d1 / (d1 - d3);
            return v0 + ab * vv;
        }

        const Float3 cpv = p - v2;
        const float d5   = ab.dot(cpv);
        const float d6   = ac.dot(cpv);

        // Vertex region of v2.
        if (d6 >= 0.0f && d5 <= d6) {
            return v2;
        }

        const float vb = d5 * d2 - d1 * d6;

        // Edge region of ac.
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float ww = d2 / (d2 - d6);
            return v0 + ac * ww;
        }

        const float va = d3 * d6 - d5 * d4;

        // Edge region of bc.
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + (v2 - v1) * ww;
        }

        // Interior face region: barycentric combination of the three vertices.
        const float denom = 1.0f / (va + vb + vc);
        const float vv    = vb * denom;
        const float ww    = vc * denom;

        return v0 + ab * vv + ac * ww;
    }

    /**
     * @brief Returns the stored face normal, independent of the query point.
     * @return @ref normal verbatim.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return normal;
    }

    /**
     * @brief Signed distance from @p p to the triangle.
     *
     * The magnitude is the Euclidean distance to the nearest triangle point; the
     * sign follows which side of the triangle plane @p p is on (positive on the
     * normal-facing side).
     *
     * @param p Query point in world space.
     * @return The signed distance.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 nn      = closest_normal(p);
        const float sd_plane = (p - a).dot(nn);

        const Float3 cp = closest_point(p);
        const float d   = (p - cp).length();

        return (sd_plane >= 0.0f) ? d : -d;
    }

    /**
     * @brief Tests whether @p p lies on the triangle's interior side.
     *
     * As with the other planar primitives, "inside" is a one-sided classification:
     * the back half-space (negative side of the plane) is interior, and
     * @p tolerance controls a shell about the triangle surface. A point in front of
     * the plane is accepted only within `tolerance` of the triangle; a point behind
     * is accepted for any non-negative tolerance and, for negative tolerance, only
     * once it is at least `|tolerance|` from the triangle.
     *
     * @param p Query point in world space.
     * @param tolerance Signed shell half-thickness, interpreted as above.
     * @return `true` when @p p is classified inside. Always `false` for a degenerate
     *         (zero-normal) triangle.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        const Float3 nn = normal;

        const float nn_len2 = nn.length_squared();

        if (nn_len2 <= 0.0f) {
            return false;
        }

        const float side = (p - a).dot(nn);

        // Quick accept/reject on the plane side before the exact distance test.
        if (side <= 0.0f) {
            if (tolerance >= 0.0f) {
                return true;
            }
        } else if (tolerance < 0.0f) {
            return false;
        }

        const Float3 cp = closest_point(p);
        const float d2  = (p - cp).length_squared();

        return side <= 0.0f ? d2 >= tolerance * tolerance
                            : d2 <= tolerance * tolerance;
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the triangle surface.
     * @param p Query point in world space.
     * @param tolerance Non-negative shell half-thickness; a negative value returns
     *        `false`.
     * @return `true` when the squared distance to the triangle is at most
     *         `tolerance^2`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const Float3 cp = closest_point(p);
        return (p - cp).length_squared() <= tolerance * tolerance;
    }

    /**
     * @brief Centroid of the triangle.
     * @return The average of the three vertices.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return (a + b + c) * (1.0f / 3.0f);
    }

    /**
     * @brief Axis-aligned bounding box of the triangle.
     * @return The `AABB` enclosing the three vertices.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const Float3 mn = atlas::cmin(a, atlas::cmin(b, c));
        const Float3 mx = atlas::cmax(a, atlas::cmax(b, c));

        return AABB(mn, mx);
    }

    /**
     * @brief Reports whether the triangle is non-degenerate.
     * @return `true` when the vertices are not collinear (`cross(b-a, c-a)` is
     *         nonzero). Recomputed from the vertices rather than relying on the
     *         stored @ref normal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        const Float3 nn = atlas::cross(b - a, c - a);
        return nn.length_squared() > 0.0f;
    }

    /**
     * @brief Intersects a ray with the triangle (Möller–Trumbore).
     *
     * @param r Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` for a parallel ray
     *         (`|det| <= eps`), a barycentric miss, or a hit nearer than `eps`. The
     *         reported normal is the stored @ref normal (renormalized, `+x`
     *         fallback), not flipped toward the ray.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface result {};

        const Float3 v0 = a;
        const Float3 v1 = b;
        const Float3 v2 = c;

        const Float3 e1 = v1 - v0;
        const Float3 e2 = v2 - v0;

        const Float3 pvec = atlas::cross(r.direction, e2);
        const float det   = e1.dot(pvec);

        // Ray parallel to the triangle plane.
        if (std::abs(det) <= eps) {
            return result;
        }

        const float inv_det = 1.0f / det;

        const Float3 tvec = r.origin - v0;
        const float u     = tvec.dot(pvec) * inv_det;

        if (u < 0.0f || u > 1.0f) {
            return result;
        }

        const Float3 qvec = atlas::cross(tvec, e1);
        const float v     = r.direction.dot(qvec) * inv_det;

        if (v < 0.0f || (u + v) > 1.0f) {
            return result;
        }

        const float t = e2.dot(qvec) * inv_det;

        // Reject hits behind (or coincident within eps of) the origin.
        if (t < eps) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = r.point_at(t);

        const Float3 normal_vec = normal;

        result.normal = atlas::normalized_or(
            normal_vec,
            Float3(1.0f, 0.0f, 0.0f));

        return result;
    }
};

/**
 * @brief Host-side builder for `Triangle` with validation.
 *
 * Vertices are set through `with_*` setters; the normal is optional and, when
 * left unset, is derived from the vertices at `build()`. Runs on the host only.
 */
class Triangle::Builder final {
public:
    Builder() = default; ///< Starts with all vertices at the origin and no explicit normal.

    /**
     * @brief Validates the accumulated vertices and constructs a `Triangle`.
     *
     * When no explicit normal was supplied it is derived from the vertices;
     * otherwise the supplied normal is used verbatim.
     *
     * @return The constructed triangle.
     * @throws std::runtime_error when the vertices are collinear (degenerate).
     */
    ATLAS_NODISCARD ATLAS_HOST Triangle
    build() const;

    /**
     * @brief Builds the triangle and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Triangle>` owning the constructed triangle.
     * @throws std::runtime_error when the vertices are collinear (degenerate).
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Triangle>
    make_host_shared() const;

    /**
     * @brief Sets the first vertex.
     * @param a_ First vertex.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_a(const Float3& a_) noexcept;

    /**
     * @brief Sets the second vertex.
     * @param b_ Second vertex.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_b(const Float3& b_) noexcept;

    /**
     * @brief Sets the third vertex.
     * @param c_ Third vertex.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_c(const Float3& c_) noexcept;

    /**
     * @brief Sets all three vertices at once.
     * @param a_ First vertex.
     * @param b_ Second vertex.
     * @param c_ Third vertex.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_vertices(const Float3& a_, const Float3& b_, const Float3& c_) noexcept;

    /**
     * @brief Overrides the automatically derived normal.
     * @param normal_ Explicit face normal to store on the built triangle.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

private:
    /**
     * @brief Throws if the accumulated vertices form a degenerate triangle.
     * @throws std::runtime_error on collinear vertices.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _a = Float3(0.0f, 0.0f, 0.0f); ///< Pending first vertex.

    Float3 _b = Float3(0.0f, 0.0f, 0.0f); ///< Pending second vertex.

    Float3 _c = Float3(0.0f, 0.0f, 0.0f); ///< Pending third vertex.

    std::optional<Float3> _normal; ///< Explicit normal override; empty means derive from vertices.
};

/// Owning host handle to a `Triangle`.
using TriangleHostPtr = atlas::host_shared_ptr<Triangle>;

/// Owning device handle to a `Triangle`.
using TriangleDevicePtr = atlas::device_shared_ptr<Triangle>;

}
