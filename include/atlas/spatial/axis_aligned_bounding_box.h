#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

/**
 * @brief Result of a ray/AABB slab-test intersection.
 *
 * Returned by @ref AABB::trace. The interval `[enter, exit]` is expressed in
 * the ray's parameter `t`; when the ray originates inside the box @ref enter is
 * clamped to zero. Read the interval only when @ref is_intersecting is true.
 */
struct HitAABB {

    /// True when the ray overlaps the box over some `t` interval.
    bool is_intersecting = false;

    /// Ray parameter `t` where the ray enters the box (0 if it starts inside).
    float enter = 0.0f;

    /// Ray parameter `t` where the ray exits the box; defaults to +max ("open").
    float exit = std::numeric_limits<float>::max();
};

/**
 * @brief An axis-aligned bounding box defined by its two extreme corners.
 *
 * The box is the set of points between @ref lower_corner and @ref upper_corner
 * component-wise. The default constructor builds the canonical *empty* box
 * (lower = +inf, upper = -inf) so that a sequence of @ref merge calls grows a
 * correct bound from nothing. Every member is host- and device-callable and
 * force-inlined, so an AABB can be built and queried inside a device lambda.
 *
 * @note An "empty" or "invalid" box has some lower component exceeding its
 *       upper; @ref is_valid / @ref is_empty distinguish these states, and most
 *       size queries return negative or meaningless values on such a box by
 *       design rather than guarding.
 */
class AABB final {
public:
    /// Minimum corner (per-axis smallest coordinate) of the box.
    Float3 lower_corner;

    /// Maximum corner (per-axis largest coordinate) of the box.
    Float3 upper_corner;

    /**
     * @brief Construct the canonical empty box (lower = +inf, upper = -inf).
     *
     * The inverted extremes make the first @ref merge collapse the box exactly
     * onto the merged point or box, so accumulation starts from a clean slate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB() noexcept {
        reset();
    }

    /**
     * @brief Construct the tight box spanning two points in any order.
     *
     * The corners are sorted component-wise, so the arguments may be passed in
     * either order and each axis's smaller/larger value lands in the correct
     * corner.
     *
     * @param point1 One corner of the desired box.
     * @param point2 The opposite corner.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const Float3& point1, const Float3& point2) noexcept
        : lower_corner(cmin(point1, point2))
        , upper_corner(cmax(point1, point2)) { }

    /**
     * @brief Copy constructor (defaulted, trivial).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const AABB&) noexcept = default;

    /**
     * @brief Copy assignment (defaulted, trivial).
     * @return Reference to this box.
     */
    AABB&
    operator=(const AABB&) noexcept = default;

    /**
     * @brief Total surface area of the box (the SAH cost metric).
     *
     * Computes `2 * (wh + wd + hd)` from the three side lengths. On an empty or
     * degenerate box the side lengths may be negative or zero and the result is
     * correspondingly meaningless; callers guard emptiness themselves.
     *
     * @return The surface area in world units squared.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    area() const noexcept {
        const float w = width();
        const float h = height();
        const float d = depth();

        return 2.0f * (w * h + w * d + h * d);
    }

    /**
     * @brief Extent along x (`upper.x - lower.x`); negative on an empty box.
     * @return The width in world units.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    width() const noexcept {
        return upper_corner.x - lower_corner.x;
    }

    /**
     * @brief Extent along y (`upper.y - lower.y`); negative on an empty box.
     * @return The height in world units.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    height() const noexcept {
        return upper_corner.y - lower_corner.y;
    }

    /**
     * @brief Extent along z (`upper.z - lower.z`); negative on an empty box.
     * @return The depth in world units.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    depth() const noexcept {
        return upper_corner.z - lower_corner.z;
    }

    /**
     * @brief Extent along an arbitrary axis.
     * @param axis Axis index (0 = x, 1 = y, 2 = z).
     * @return `upper[axis] - lower[axis]`; negative on an empty box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length(const std::size_t axis) const noexcept {
        return upper_corner[axis] - lower_corner[axis];
    }

    /**
     * @brief Test whether this box and @p other share any volume (boundary inclusive).
     *
     * Two boxes overlap unless they are separated on at least one axis; the
     * check tests all three axes at once and negates. Touching faces count as
     * overlapping because the comparison uses strict inequalities for separation.
     *
     * @param other The box to test against.
     * @return True if the boxes intersect or touch.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AABB& other) const noexcept {
        const bool separated = any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));

        return !separated;
    }

    /**
     * @brief Test whether a point lies inside the box (boundary inclusive).
     * @param point World-space point to test.
     * @return True if @p point is within `[lower, upper]` on every axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Float3& point) const noexcept {
        return all((point >= lower_corner) & (point <= upper_corner));
    }

    /**
     * @brief Boolean ray-hit test: does the ray touch the box at all?
     * @param ray Query ray.
     * @return True if the ray intersects the box; a thin wrapper over @ref trace.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray& ray) const noexcept {
        return trace(ray).is_intersecting;
    }

    /**
     * @brief Ray/box intersection via the slab method, returning the hit interval.
     *
     * Intersects the ray against the three axis-aligned slabs and keeps the
     * running `[t_enter, t_exit]` overlap. An axis whose direction component is
     * (near) zero is parallel to that slab: it is a miss unless the origin
     * already lies between the slab planes, in which case that axis imposes no
     * constraint. If the ray starts inside the box the reported entry distance
     * is clamped to zero so the interval begins at the origin. `t_enter` is
     * initialized to 0, so intersections strictly behind the origin are excluded.
     *
     * @param ray Query ray (direction assumed unit length).
     * @return A @ref HitAABB; on a miss `is_intersecting` is false and the
     *         interval fields hold their partially-updated sentinels.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitAABB
    trace(const Ray& ray) const noexcept {
        HitAABB isect {};

        float t_enter = 0.0f;
        float t_exit  = inf;

        for (int i = 0; i < 3; ++i) {
            const float o  = ray.origin[i];
            const float d  = ray.direction[i];
            const float mn = lower_corner[i];
            const float mx = upper_corner[i];

            // Ray parallel to this axis' slab: a hit is only possible if the
            // origin already lies between the two planes; otherwise reject.
            if (std::abs(d) <= eps) {
                if (o < mn || o > mx) return isect;
                continue;
            }

            float t0 = (mn - o) / d;
            float t1 = (mx - o) / d;

            // A negative direction component swaps which plane is the near one.
            if (t0 > t1) {
                const float tmp = t0;
                t0              = t1;
                t1              = tmp;
            }

            // Intersect this slab's interval into the running [enter, exit].
            t_enter = std::max(t_enter, t0);
            t_exit  = std::min(t_exit, t1);

            // Slabs no longer overlap: the ray misses the box.
            if (t_enter > t_exit) return isect;
        }

        // Origin inside the box: report entry at the origin, not behind it.
        if (contains(ray.origin)) t_enter = 0.0f;

        isect.is_intersecting = true;
        isect.enter           = t_enter;
        isect.exit            = t_exit;

        return isect;
    }

    /**
     * @brief Geometric center (midpoint of the two corners).
     * @return The box center in world space.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    center() const noexcept {
        return (lower_corner + upper_corner) * 0.5f;
    }

    /**
     * @brief Per-axis size vector (`upper - lower`).
     * @return The extents; any component is negative on an empty box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    extents() const noexcept {
        return upper_corner - lower_corner;
    }

    /**
     * @brief Length of the box's main diagonal.
     * @return Euclidean distance between the two corners.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length() const noexcept {
        return (upper_corner - lower_corner).length();
    }

    /**
     * @brief Squared length of the main diagonal (avoids the sqrt).
     * @return Squared distance between the two corners.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length_squared() const noexcept {
        return (upper_corner - lower_corner).length_squared();
    }

    /**
     * @brief Whether the box is a well-formed, finite region.
     *
     * Requires both corners to be finite (no inf/NaN — so the default empty box
     * is *not* valid) and lower <= upper on every axis. Used by
     * @ref transform_aabb to skip transforming an uninitialized bound.
     *
     * @return True if the box represents a real finite region.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(lower_corner)
            && atlas::isfinite(upper_corner)
            && atlas::all(lower_corner <= upper_corner);
    }

    /**
     * @brief Reset to the canonical empty box so accumulation can restart.
     *
     * Sets lower to +inf and upper to -inf; the first subsequent @ref merge then
     * snaps both corners onto the merged geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept {
        lower_corner = Float3(inf, inf, inf);
        upper_corner = Float3(-inf, -inf, -inf);
    }

    /**
     * @brief Grow the box to include a point.
     * @param point Point to absorb into the bound.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Float3& point) noexcept {
        lower_corner = cmin(lower_corner, point);
        upper_corner = cmax(upper_corner, point);
    }

    /**
     * @brief Grow the box to include another box (their union bound).
     * @param other Box to absorb.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AABB& other) noexcept {
        lower_corner = cmin(lower_corner, other.lower_corner);
        upper_corner = cmax(upper_corner, other.upper_corner);
    }

    /**
     * @brief Inflate (or, with a negative delta, shrink) the box on every side.
     * @param delta Amount added to each face outward; applied to all three axes.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(const float delta) noexcept {
        const Float3 d(delta, delta, delta);

        lower_corner -= d;
        upper_corner += d;
    }

    /**
     * @brief Return one of the box's eight corners, selected by a 3-bit index.
     *
     * Bit 0 chooses upper vs lower x, bit 1 the y axis, bit 2 the z axis, so
     * indices 0..7 enumerate every corner. Used by @ref transform_aabb to
     * rebuild a bound after an arbitrary point transform.
     *
     * @param idx Corner selector in the range [0, 7].
     * @return The selected corner in world space.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    corner(const std::size_t idx) const noexcept {
        const float x = (idx & 1) ? upper_corner.x : lower_corner.x;
        const float y = (idx & 2) ? upper_corner.y : lower_corner.y;
        const float z = (idx & 4) ? upper_corner.z : lower_corner.z;

        return Float3(x, y, z);
    }

    /**
     * @brief Clamp a point into the box (nearest point on or inside it).
     * @param point Point to project.
     * @return @p point with each component clamped to `[lower, upper]`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    clamp(const Float3& point) const noexcept {
        return atlas::clamp(point, lower_corner, upper_corner);
    }

    /**
     * @brief Whether the box has zero or negative extent on any axis.
     *
     * Uses `<=`, so a box that is merely flat on one axis (a degenerate slab)
     * counts as empty. This is a stricter emptiness test than the mere
     * lower > upper condition and is distinct from @ref is_valid, which also
     * rejects non-finite corners.
     *
     * @return True if the box encloses no positive volume.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept {
        return any(upper_corner <= lower_corner);
    }
};

/**
 * @brief Build a degenerate (point) box collapsed onto a single point.
 * @param p The point the box should enclose.
 * @return An AABB whose two corners both equal @p p.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
make_aabb(const Float3& p) noexcept {
    AABB b;
    b.lower_corner = p;
    b.upper_corner = p;
    return b;
}

/**
 * @brief Return the union bound of two boxes without mutating either.
 *
 * The free-function counterpart to @ref AABB::merge, convenient in reductions
 * where a fresh result box is wanted rather than in-place growth.
 *
 * @param a First box.
 * @param b Second box.
 * @return The smallest AABB containing both inputs.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
merge_aabb(const AABB& a,
           const AABB& b) noexcept {
    AABB out;
    out.lower_corner = cmin(a.lower_corner, b.lower_corner);
    out.upper_corner = cmax(a.upper_corner, b.upper_corner);
    return out;
}

/**
 * @brief Squared distance from a point to the nearest point of a box.
 *
 * Zero when the point lies inside the box (the clamp is a no-op there). Squared
 * to avoid a sqrt in nearest-primitive traversals.
 *
 * @param bounds The box.
 * @param point  Query point.
 * @return Squared distance; zero if @p point is inside @p bounds.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
aabb_distance_squared(const AABB& bounds,
                      const Float3& point) noexcept {
    const Float3 closest = bounds.clamp(point);
    return (closest - point).length_squared();
}

/**
 * @brief Rebuild an AABB after applying an arbitrary point transform.
 *
 * Transforms all eight corners of @p bound and re-merges them into a fresh
 * axis-aligned box. Because a rotated box is generally not axis-aligned, the
 * result is the tight AABB of the transformed corners (a conservative bound,
 * not the exact transformed shape). An invalid input bound yields the empty box.
 *
 * @tparam TransformPoint Callable `Float3(Float3)` mapping a world point.
 * @param bound     Source box; if not @ref AABB::is_valid the empty box is returned.
 * @param transform Point-to-point transform applied to each corner.
 * @return The axis-aligned bound of the eight transformed corners.
 */
template <typename TransformPoint>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
transform_aabb(const AABB& bound,
               TransformPoint transform) noexcept {
    AABB transformed {};
    if (!bound.is_valid()) {
        return transformed;
    }

    for (int corner = 0; corner < 8; ++corner) {
        transformed.merge(transform(bound.corner(static_cast<std::size_t>(corner))));
    }

    return transformed;
}

}