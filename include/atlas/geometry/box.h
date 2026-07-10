#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cstddef>

namespace atlas {

/**
 * @brief Axis-aligned box primitive defined by two opposite corners.
 *
 * A `Box` is the solid region `[lower_corner, upper_corner]` with faces parallel
 * to the world axes; it carries no rotation. It is one leaf of the `Geometry`
 * tagged union, is trivially copyable, and every query below is marked
 * `ATLAS_ALL_DEVICE` so it can run inside a device lambda captured by value.
 *
 * The default box is the unit cube centered on the origin, spanning
 * `[-1, -1, -1] .. [1, 1, 1]`.
 *
 * @note All queries assume `is_valid()` (finite corners with
 *       `upper_corner > lower_corner` componentwise); a collapsed or inverted
 *       box yields defined but not necessarily meaningful results.
 */
class Box final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Box.
    class Builder;

public:
    Float3 lower_corner = Float3(-1.0f, -1.0f, -1.0f); ///< Minimum corner (smallest x, y, z).
    Float3 upper_corner = Float3(1.0f, 1.0f, 1.0f);    ///< Maximum corner (largest x, y, z).

    /**
     * @brief Constructs the default unit cube `[-1,-1,-1]..[1,1,1]`.
     */
    Box() noexcept = default;

    /**
     * @brief Constructs a box from its two opposite corners.
     *
     * @param lower_corner_ Minimum corner; each component should not exceed the
     *        matching component of @p upper_corner_ for a valid box.
     * @param upper_corner_ Maximum corner.
     * @note Callable on host and device; the corners are stored verbatim without
     *       reordering, so an inverted pair produces an invalid box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Box(const Float3& lower_corner_, const Float3& upper_corner_) noexcept
        : lower_corner(lower_corner_)
        , upper_corner(upper_corner_) { }

    Box(const Box& other) noexcept = default; ///< Trivial copy.
    Box(Box&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Box&
    operator=(const Box& other) noexcept = default;
    /// Trivial move assignment.
    Box&
    operator=(Box&& other) noexcept = default;

    ~Box() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Box.
     * @return A default-initialized `Box::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the box surface nearest to @p p.
     *
     * For a point outside the box the result is @p p clamped to the box extents.
     * For a point inside the box it is pushed out to the closest of the six faces
     * (the face minimizing the perpendicular distance), so the returned point
     * always lies on the boundary rather than in the interior.
     *
     * @param p Query point in world space.
     * @return The closest surface point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        Float3 cp = atlas::clamp(p, lower_corner, upper_corner);

        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        if (inside) {
            bool hit_lower;
            const std::size_t axis = nearest_face(p, hit_lower);

            // Interior points clamp to themselves, so snap the nearest-face axis
            // out to that face plane to land on the surface.
            cp[axis] = hit_lower ? lower_corner[axis] : upper_corner[axis];
        }

        return cp;
    }

    /**
     * @brief Returns the outward unit surface normal nearest to @p p.
     *
     * When @p p is inside the box the normal is that of the closest face
     * (`-1`/`+1` on the nearest axis). When @p p is outside, the normal is taken
     * from the dominant axis of the offset between @p p and its clamped point,
     * which for points off an edge or corner selects the axis of largest
     * separation.
     *
     * @param p Query point in world space.
     * @return An axis-aligned unit normal, or the zero vector only if @p p sits
     *         exactly on a corner where the offset vanishes.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        Float3 n(0.0f);

        if (inside) {
            bool hit_lower;
            const std::size_t axis = nearest_face(p, hit_lower);

            n[axis] = hit_lower ? -1.0f : 1.0f;
            return n;
        }

        const Float3 cp = atlas::clamp(p, lower_corner, upper_corner);
        const Float3 d  = p - cp;

        // Outside: the largest-magnitude offset component names the face the
        // point is escaping through.
        const std::size_t axis = atlas::abs(d).major_axis();

        n[axis] = (d[axis] >= 0.0f) ? 1.0f : -1.0f;
        return n;
    }

    /**
     * @brief Signed distance from @p p to the box surface.
     *
     * @param p Query point in world space.
     * @return Negative inside the box (magnitude equal to the distance to the
     *         nearest face), zero on the surface, positive outside (the Euclidean
     *         distance to the surface).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const bool inside = atlas::all(p >= lower_corner)
            && atlas::all(p <= upper_corner);

        if (inside) {
            const Float3 l_to_p = p - lower_corner;
            const Float3 p_to_u = upper_corner - p;

            // Deepest penetration is the smallest gap to any of the six faces;
            // negate because interior distances are signed negative.
            const float m1 = l_to_p.min();
            const float m2 = p_to_u.min();

            return -((m1 < m2) ? m1 : m2);
        }

        const Float3 cp = atlas::clamp(p, lower_corner, upper_corner);
        return (cp - p).length();
    }

    /**
     * @brief Tests whether @p p lies within the box, optionally dilated.
     *
     * @param p Query point in world space.
     * @param tolerance Signed dilation of the box before the test. Positive values
     *        grow the box outward; negative values shrink it inward.
     * @return `true` when @p p is inside the tolerance-adjusted box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return atlas::all(p >= lower_corner - tolerance)
            && atlas::all(p <= upper_corner + tolerance);
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the box surface.
     *
     * Interior points are accepted when their distance to the nearest face is at
     * most @p tolerance; exterior points are accepted when their squared distance
     * to the box is at most `tolerance^2`.
     *
     * @param p Query point in world space.
     * @param tolerance Non-negative surface half-thickness. A negative value
     *        always returns `false`.
     * @return `true` when @p p is on the (thickened) surface shell.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const Float3& lo       = lower_corner;
        const Float3& hi       = upper_corner;
        const float tolerance2 = tolerance * tolerance;

        const bool inside = atlas::all(p >= lo)
            && atlas::all(p <= hi);

        if (inside) {
            // Interior: nearest face is the minimum of the per-axis gaps.
            const float dx = (p.x - lo.x < hi.x - p.x) ? p.x - lo.x : hi.x - p.x;
            const float dy = (p.y - lo.y < hi.y - p.y) ? p.y - lo.y : hi.y - p.y;
            const float dz = (p.z - lo.z < hi.z - p.z) ? p.z - lo.z : hi.z - p.z;
            const float d  = (dx < dy) ? ((dx < dz) ? dx : dz)
                                       : ((dy < dz) ? dy : dz);

            return d <= tolerance;
        }

        // Exterior: accumulate the squared per-axis overshoot beyond each slab.
        const float dx = p.x < lo.x ? lo.x - p.x
            : p.x > hi.x            ? p.x - hi.x
                                    : 0.0f;
        const float dy = p.y < lo.y ? lo.y - p.y
            : p.y > hi.y            ? p.y - hi.y
                                    : 0.0f;
        const float dz = p.z < lo.z ? lo.z - p.z
            : p.z > hi.z            ? p.z - hi.z
                                    : 0.0f;

        return dx * dx + dy * dy + dz * dz <= tolerance2;
    }

    /**
     * @brief Geometric center of the box.
     * @return The midpoint of the two corners.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return (lower_corner + upper_corner) * 0.5f;
    }

    /**
     * @brief Axis-aligned bounding box of the primitive.
     * @return An `AABB` coincident with the box itself.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        return AABB(lower_corner, upper_corner);
    }

    /**
     * @brief Reports whether the corners describe a well-formed box.
     *
     * A box collapsed on any axis encloses no volume, so `is_inside` can never be true
     * and `closest_normal` has no well-defined face to pick there. Such a box is rejected
     * rather than silently behaving as a degenerate plane or point.
     *
     * @return `true` when both corners are finite and `upper_corner > lower_corner` on
     *         every axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(lower_corner)
            && atlas::isfinite(upper_corner)
            && atlas::all(upper_corner > lower_corner);
    }

    /**
     * @brief Intersects a ray with the box using the AABB slab test.
     *
     * Delegates the interval test to `AABB::trace` and picks the entry hit when
     * the ray starts outside (`enter >= eps`), otherwise the exit hit for a ray
     * that originates inside the box. The surface normal is recovered from
     * `closest_normal` at the hit point.
     *
     * @param r Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` when the ray misses or
     *         the whole intersection interval lies behind the origin (`exit < eps`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface result {};

        const HitAABB hit = bound().trace(r);

        if (!hit.is_intersecting || hit.exit < eps) {
            return result;
        }

        // A positive entry means the origin is outside; otherwise the ray starts
        // inside and the first forward crossing is the exit face.
        const bool use_enter = (hit.enter >= eps);
        const float t        = use_enter ? hit.enter : hit.exit;

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = r.point_at(t);
        result.normal          = closest_normal(result.point);

        return result;
    }

private:
    /**
     * @brief Finds the box face nearest to an interior point.
     *
     * @param p Query point, assumed to lie inside the box.
     * @param[out] hit_lower Set to `true` when the nearest face is a lower-corner
     *        face, `false` when it is an upper-corner face.
     * @return The axis index (0=x, 1=y, 2=z) of the nearest face; the minor axis
     *         of the winning corner's distance vector, i.e. the axis with the
     *         smallest gap.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    nearest_face(const Float3& p, bool& hit_lower) const noexcept {
        const Float3 l_to_p = p - lower_corner;
        const Float3 p_to_u = upper_corner - p;

        hit_lower = (l_to_p.min() < p_to_u.min());

        return hit_lower ? l_to_p.minor_axis() : p_to_u.minor_axis();
    }
};

/**
 * @brief Host-side builder for `Box` with validation.
 *
 * Accumulates corner parameters through `with_*` setters and, on `build()`,
 * throws if the resulting box is invalid. Runs on the host only.
 */
class Box::Builder final {
public:
    Builder() = default; ///< Starts from the default unit-cube corners.

    /**
     * @brief Validates the accumulated parameters and constructs a `Box`.
     * @return The constructed box.
     * @throws std::runtime_error when the corners do not form a valid box.
     */
    ATLAS_NODISCARD ATLAS_HOST Box
    build() const;

    /**
     * @brief Builds the box and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Box>` owning the constructed box.
     * @throws std::runtime_error when the corners do not form a valid box.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Box>
    make_host_shared() const;

    /**
     * @brief Sets the minimum corner.
     * @param lower_corner_ Minimum corner.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_lower_corner(const Float3& lower_corner_) noexcept;

    /**
     * @brief Sets the maximum corner.
     * @param upper_corner_ Maximum corner.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_upper_corner(const Float3& upper_corner_) noexcept;

private:
    /**
     * @brief Throws if the accumulated corners do not form a valid box.
     * @throws std::runtime_error on invalid parameters.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _lower_corner = Float3(-1.0f, -1.0f, -1.0f); ///< Pending minimum corner.
    Float3 _upper_corner = Float3(1.0f, 1.0f, 1.0f);    ///< Pending maximum corner.
};

/// Owning host handle to a `Box`.
using BoxHostPtr = atlas::host_shared_ptr<Box>;

/// Owning device handle to a `Box`.
using BoxDevicePtr = atlas::device_shared_ptr<Box>;

}
