#pragma once
#include <algorithm>
#include <cmath>

namespace atlas::spatial {

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox() noexcept {
    // Default-construct an "empty" AABB by resetting to inverted extremes.
    // After reset():
    //   lower_corner = (+max, +max, +max)
    //   upper_corner = (-max, -max, -max)
    // so that the first merge(point) correctly initializes a valid box.
    reset();
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept {
    // Construct from two points, ensuring proper ordering on each axis.
    // cmin/cmax are component-wise min/max.
    lower_corner = math::cmin(point1, point2);
    upper_corner = math::cmax(point1, point2);
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
    // Copy constructor: preserves corners exactly.
}

template <typename T>
T
AxisAlignedBoundingBox<T>::area() const noexcept {
    // Surface area of an axis-aligned box:
    //   A = 2 * (wh + wd + hd)
    // where w/h/d are extents along x/y/z.
    //
    // Note:
    // - For an "empty" AABB (reset() state), widths can be negative.
    // - Callers generally merge() into the box before using area().
    T width  = this->width();
    T height = this->height();
    T depth  = this->depth();
    return T(2) * (width * height + width * depth + height * depth);
}

template <typename T>
T
AxisAlignedBoundingBox<T>::width() const noexcept {
    // Extent along x-axis.
    return upper_corner.x - lower_corner.x;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::height() const noexcept {
    // Extent along y-axis.
    return upper_corner.y - lower_corner.y;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::depth() const noexcept {
    // Extent along z-axis.
    return upper_corner.z - lower_corner.z;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::length(std::size_t axis) const noexcept {
    // Generic per-axis extent accessor:
    // axis = 0 -> x, 1 -> y, 2 -> z.
    return upper_corner[axis] - lower_corner[axis];
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::overlaps(const AxisAlignedBoundingBox& other) const noexcept {
    // AABB overlap test (separating axis on x/y/z):
    // If there exists any axis where this is completely to the "left" of other,
    // or completely to the "right" of other, they do NOT overlap.
    //
    // Here, we detect separation using vector comparisons:
    //   upper < other.lower  OR  lower > other.upper
    // then "any()" across components.
    const bool overlap = math::any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));
    return !overlap;
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::contains(const Vector3<T>& point) const noexcept {
    // Point-in-AABB test (inclusive on boundaries).
    // True iff for every axis: lower <= point <= upper.
    return math::all((point >= lower_corner) & (point <= upper_corner));
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::intersects(const Ray<T>& ray) const noexcept {
    // Ray vs AABB intersection using the "slab" method.
    //
    // For each axis i:
    //   Solve intersection with the two planes x=mn and x=mx (or y/z).
    //   This yields an interval [t0, t1] along the ray parameter t.
    // The intersection with the box exists if the intersection of the three
    // intervals across axes is non-empty:
    //   tmin = max over axes of entry t
    //   tmax = min over axes of exit  t
    // and tmin <= tmax.
    //
    // Special case: if direction component is ~0, the ray is parallel to that slab.
    // Then the origin must lie within [mn, mx] for that axis, otherwise miss.
    T tmin = T(0);
    T tmax = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // Parallel ray to slab planes -> no division; either always inside or always outside.
        if (std::abs(d) <= eps) {
            if (o < mn || o > mx) return false; // outside slab => no intersection
            continue;
        }

        // Compute parametric distances to the two slab planes.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;

        // Ensure t0 is entry and t1 is exit for this axis.
        if (t0 > t1) std::swap(t0, t1);

        // Tighten global interval.
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);

        // Early-out: empty interval => miss.
        if (tmin > tmax) return false;
    }

    return true;
}

template <typename T>
AxisAlignedBoundingBoxRayIntersection<T>
AxisAlignedBoundingBox<T>::trace(const Ray<T>& ray) const noexcept {
    // Like intersects(), but returns the entry/exit distances (t_enter/t_exit).
    //
    // Convention:
    // - If no hit, returns default-initialized `isect` (is_intersecting = false).
    // - If the ray starts inside the box, we force enter = 0.
    AxisAlignedBoundingBoxRayIntersection<T> isect {};

    T t_enter = T(0);
    T t_exit  = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // Parallel slab handling (same logic as intersects()).
        if (std::abs(d) <= eps) {
            if (o < mn || o > mx) return isect; // miss
            continue;
        }

        // Slab interval on this axis.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;
        if (t0 > t1) std::swap(t0, t1);

        // Combine with the running interval.
        t_enter = std::max(t_enter, t0);
        t_exit  = std::min(t_exit, t1);

        // Interval became empty -> no intersection.
        if (t_enter > t_exit) return isect;
    }

    // If the origin is inside the box, treat the entry distance as 0
    // (useful for traversal logic that expects non-negative entry).
    if (contains(ray.origin)) t_enter = T(0);

    // Successful hit: fill result.
    isect.is_intersecting = true;
    isect.enter           = t_enter;
    isect.exit            = t_exit;
    return isect;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::center() const noexcept {
    // Geometric center = (min + max) / 2.
    return (lower_corner + upper_corner) * T(0.5);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::extents() const noexcept {
    // Extents vector (width, height, depth) = max - min.
    return upper_corner - lower_corner;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length() const noexcept {
    // Euclidean length of the diagonal vector.
    return (upper_corner - lower_corner).length();
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length_squared() const noexcept {
    // Squared diagonal length (avoids sqrt, useful for comparisons).
    return (upper_corner - lower_corner).length_squared();
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_valid() const noexcept {
    // A valid sampling/traversal AABB must be finite and non-inverted on every axis.
    return std::isfinite(lower_corner.x) && std::isfinite(lower_corner.y) && std::isfinite(lower_corner.z)
        && std::isfinite(upper_corner.x) && std::isfinite(upper_corner.y) && std::isfinite(upper_corner.z)
        && lower_corner.x <= upper_corner.x && lower_corner.y <= upper_corner.y && lower_corner.z <= upper_corner.z;
}

template <typename T>
void
AxisAlignedBoundingBox<T>::reset() noexcept {
    // Initialize to an "empty" inverted box:
    // - lower set to +max, upper set to -max
    // This way, merge(point) will correctly shrink/grow into a valid box.
    lower_corner = Vector3<T> { static_cast<T>(inf), static_cast<T>(inf), static_cast<T>(inf) };
    upper_corner = Vector3<T> { -static_cast<T>(inf), -static_cast<T>(inf), -static_cast<T>(inf) };
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const Vector3<T>& point) noexcept {
    // Expand the box to include a point by taking component-wise min/max.
    // Works even if the box is currently "empty" from reset().
    lower_corner = math::cmin(lower_corner, point);
    upper_corner = math::cmax(upper_corner, point);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const AxisAlignedBoundingBox& other) noexcept {
    // Expand the box to include another box (union).
    lower_corner = math::cmin(lower_corner, other.lower_corner);
    upper_corner = math::cmax(upper_corner, other.upper_corner);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::expand(T delta) noexcept {
    // Uniformly expand the box outward by `delta` along each axis:
    //   min -= (delta,delta,delta)
    //   max += (delta,delta,delta)
    //
    // If delta is negative, this shrinks the box (callers should ensure validity).
    const Vector3<T> d { delta, delta, delta };
    lower_corner -= d;
    upper_corner += d;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::corner(const std::size_t idx) const noexcept {
    // Return one of the 8 corners by bit-encoding:
    //   bit0 -> choose x (0=min, 1=max)
    //   bit1 -> choose y (0=min, 1=max)
    //   bit2 -> choose z (0=min, 1=max)
    //
    // idx in [0,7]:
    //   0: (min,min,min)
    //   1: (max,min,min)
    //   2: (min,max,min)
    //   3: (max,max,min)
    //   4: (min,min,max)
    //   5: (max,min,max)
    //   6: (min,max,max)
    //   7: (max,max,max)
    const T x = (idx & 1) ? upper_corner.x : lower_corner.x;
    const T y = (idx & 2) ? upper_corner.y : lower_corner.y;
    const T z = (idx & 4) ? upper_corner.z : lower_corner.z;
    return Vector3<T>(x, y, z);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::clamp(const Vector3<T>& point) const noexcept {
    // Clamp a point into the box (component-wise).
    // Useful for closest-point queries on an AABB.
    return math::clamp(point, lower_corner, upper_corner);
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_empty() const noexcept {
    // Empty test: if any axis has upper <= lower, the box has non-positive volume
    // (or is inverted). This matches the reset() state and also catches invalid boxes.
    return math::any(upper_corner <= lower_corner);
}

} // namespace atlas::spatial
