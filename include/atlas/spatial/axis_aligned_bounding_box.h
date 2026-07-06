#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

/**
 * @file axis_aligned_bounding_box.h
 * @brief The axis-aligned bounding box (AABB) used throughout Atlas as
 *        the broad-phase reject test before an exact geometric query
 *        (`UnitField`, `SinkUnitBounds`, `BVH` nodes, ...).
 *
 * @details
 * ### Derivation — the slab method (`intersects`/`trace`)
 * An AABB is the intersection of three axis-aligned "slabs" (regions
 * between two parallel planes), one per axis:
 * `lower_corner[i] <= x[i] <= upper_corner[i]`. For a ray
 * `p(t) = origin + t*direction`, the ray enters/exits slab `i` at
 * `t = (lower_corner[i] - origin[i]) / direction[i]` and
 * `t = (upper_corner[i] - origin[i]) / direction[i]` respectively
 * (solving `p(t)[i] = lower_corner[i]` / `= upper_corner[i]` for `t`);
 * sorting the two so `t0 <= t1` gives that slab's entry/exit interval
 * `[t0, t1]`. The ray intersects the box iff the *intersection* of all
 * three slabs' intervals is non-empty: tracking a running
 * `[tmin, tmax]` initialized to `[0, inf)` (clamped to `t >= 0`, i.e.
 * the ray's forward half only) and narrowing it by each axis'
 * `[t0, t1]` (`tmin = max(tmin, t0)`, `tmax = min(tmax, t1)`), the box
 * is hit iff `tmin <= tmax` after all three axes — this is the standard
 * "slab method" for ray-box intersection (Kay & Kajiya, 1986), the same
 * one `HitAABB`/`trace()` return the resulting `[enter, exit]` interval
 * from. A near-zero `direction[i]` (the ray runs parallel to that
 * axis's slab) skips the division and instead rejects outright if the
 * ray's origin falls outside that slab on that axis (a parallel ray
 * either lies entirely inside the slab, in which case it imposes no
 * constraint, or entirely outside, in which case there is no
 * intersection regardless of the other axes).
 *
 * `trace()` additionally special-cases a ray origin already inside the
 * box (`contains(ray.origin)`): the computed `t_enter` from the slab
 * sweep can come out negative or otherwise not reflect "the ray is
 * already inside," so it is clamped to `0` in that case, matching the
 * intuitive semantics that a ray starting inside the box enters
 * immediately.
 */

namespace atlas {

/**
 * @brief Result of `AABB::trace`: whether a ray hits the box, and the
 *        parametric `[enter, exit]` interval along the ray where it is
 *        inside the box (only meaningful if `is_intersecting`).
 */
struct HitAABB {

    bool is_intersecting = false;

    float enter = 0.0f;

    float exit = std::numeric_limits<float>::max();
};

/**
 * @brief Axis-aligned bounding box. See this file's top-of-file
 *        documentation for the slab-method ray intersection derivation.
 */
class AABB final {
public:
    Float3 lower_corner;

    Float3 upper_corner;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB() noexcept {
        reset();
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const Float3& point1, const Float3& point2) noexcept
        : lower_corner(cmin(point1, point2))
        , upper_corner(cmax(point1, point2)) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AABB(const AABB&) noexcept = default;

    AABB&
    operator=(const AABB&) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    area() const noexcept {
        const float w = width();
        const float h = height();
        const float d = depth();

        return 2.0f * (w * h + w * d + h * d);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    width() const noexcept {
        return upper_corner.x - lower_corner.x;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    height() const noexcept {
        return upper_corner.y - lower_corner.y;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    depth() const noexcept {
        return upper_corner.z - lower_corner.z;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length(const std::size_t axis) const noexcept {
        return upper_corner[axis] - lower_corner[axis];
    }

    /** @brief Whether this box and `other` overlap: the separating-axis
     *  test for AABBs — two boxes are disjoint iff they are separated
     *  along at least one axis, so this checks the negation across all
     *  three. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AABB& other) const noexcept {
        const bool separated = any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));

        return !separated;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Float3& point) const noexcept {
        return all((point >= lower_corner) & (point <= upper_corner));
    }

    /** @brief Boolean-only slab-method hit test (see this file's
     *  top-of-file Derivation); cheaper than `trace()` when the
     *  `[enter, exit]` interval itself isn't needed. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray& ray) const noexcept {
        return trace(ray).is_intersecting;
    }

    /** @brief Full slab-method intersection returning the `[enter,
     *  exit]` parametric interval, with the ray-origin-inside-the-box
     *  special case; see this file's top-of-file Derivation. */
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

            if (std::abs(d) <= eps) {
                if (o < mn || o > mx) return isect;
                continue;
            }

            float t0 = (mn - o) / d;
            float t1 = (mx - o) / d;

            if (t0 > t1) {
                const float tmp = t0;
                t0              = t1;
                t1              = tmp;
            }

            t_enter = std::max(t_enter, t0);
            t_exit  = std::min(t_exit, t1);

            if (t_enter > t_exit) return isect;
        }

        if (contains(ray.origin)) t_enter = 0.0f;

        isect.is_intersecting = true;
        isect.enter           = t_enter;
        isect.exit            = t_exit;

        return isect;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    center() const noexcept {
        return (lower_corner + upper_corner) * 0.5f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    extents() const noexcept {
        return upper_corner - lower_corner;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length() const noexcept {
        return (upper_corner - lower_corner).length();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    diagonal_length_squared() const noexcept {
        return (upper_corner - lower_corner).length_squared();
    }

    /** @brief Whether this box has finite, correctly-ordered corners —
     *  `false` for the sentinel state `reset()` produces, or after
     *  transforming/merging invalid input. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(lower_corner)
            && atlas::isfinite(upper_corner)
            && atlas::all(lower_corner <= upper_corner);
    }

    /** @brief Sets this box to the "empty/unset" sentinel: inverted
     *  infinite bounds (`lower_corner > upper_corner` on every axis),
     *  so the first `merge()` call establishes real bounds and
     *  `is_valid()` correctly reports `false` until then. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept {
        lower_corner = Float3(inf, inf, inf);
        upper_corner = Float3(-inf, -inf, -inf);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Float3& point) noexcept {
        lower_corner = cmin(lower_corner, point);
        upper_corner = cmax(upper_corner, point);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AABB& other) noexcept {
        lower_corner = cmin(lower_corner, other.lower_corner);
        upper_corner = cmax(upper_corner, other.upper_corner);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(const float delta) noexcept {
        const Float3 d(delta, delta, delta);

        lower_corner -= d;
        upper_corner += d;
    }

    /** @brief One of the box's 8 corners, selected by the low 3 bits of
     *  `idx` (bit 0 -> x, bit 1 -> y, bit 2 -> z; `0` = lower_corner,
     *  `1` set = that axis takes upper_corner). Used to enumerate all 8
     *  corners for e.g. `transform_aabb`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    corner(const std::size_t idx) const noexcept {
        const float x = (idx & 1) ? upper_corner.x : lower_corner.x;
        const float y = (idx & 2) ? upper_corner.y : lower_corner.y;
        const float z = (idx & 4) ? upper_corner.z : lower_corner.z;

        return Float3(x, y, z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    clamp(const Float3& point) const noexcept {
        return atlas::clamp(point, lower_corner, upper_corner);
    }

    /** @brief Whether the box has zero or negative extent on any axis
     *  — distinct from `is_valid()`: a box can be valid (finite,
     *  correctly ordered) yet empty (a degenerate flat/point box, e.g.
     *  a bound around a single point or a planar shape). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept {
        return any(upper_corner <= lower_corner);
    }
};

/** @brief A degenerate (zero-extent) AABB at a single point — the
 *  identity element for building up a bound via repeated `merge()`
 *  calls starting from one point instead of `reset()`. */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
make_aabb(const Float3& p) noexcept {
    AABB b;
    b.lower_corner = p;
    b.upper_corner = p;
    return b;
}

/** @brief Free-function form of `AABB::merge(const AABB&)`, for
 *  functional-style reduction (e.g. `atlas::transform_reduce`). */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
merge_aabb(const AABB& a,
           const AABB& b) noexcept {
    AABB out;
    out.lower_corner = cmin(a.lower_corner, b.lower_corner);
    out.upper_corner = cmax(a.upper_corner, b.upper_corner);
    return out;
}

/** @brief Squared distance from `point` to the closest point on/in
 *  `bounds` (`0` if `point` is inside); avoids a `sqrt` when only
 *  ordering/thresholding by distance is needed. */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
aabb_distance_squared(const AABB& bounds,
                      const Float3& point) noexcept {
    const Float3 closest = bounds.clamp(point);
    return (closest - point).length_squared();
}

/**
 * @brief Transforms `bound` by mapping each of its 8 corners through
 *        `transform` and merging the results — the standard way to
 *        re-bound an AABB after an arbitrary (e.g. rotating) transform,
 *        since a transformed box is generally no longer axis-aligned
 *        and the tightest enclosing axis-aligned box is exactly the
 *        merge of the transformed corners. Returns an invalid
 *        (`reset()`-state) box if `bound` itself is invalid.
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
