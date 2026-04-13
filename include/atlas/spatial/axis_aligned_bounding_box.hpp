#pragma once
#include <algorithm>
#include <cmath>

namespace atlas::spatial {

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox() noexcept {
    // Initialize the bounding box into an "empty / inverted" state.
    //
    // After reset():
    // - lower_corner starts at (+inf, +inf, +inf)
    // - upper_corner starts at (-inf, -inf, -inf)
    //
    // This convention is useful because the first merge() call will correctly
    // shrink/grow the box to the inserted point or box.
    reset();
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept {
    // Build an axis-aligned box from two arbitrary corner candidates.
    //
    // The inputs are not assumed to already be ordered. Component-wise min/max
    // are used so the final box always satisfies:
    // - lower_corner <= upper_corner
    // on every axis.
    lower_corner = math::cmin(point1, point2);
    upper_corner = math::cmax(point1, point2);
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
    // Copy constructor:
    // duplicate the lower/upper corners exactly from the source box.
}

template <typename T>
T
AxisAlignedBoundingBox<T>::area() const noexcept {
    // Return the surface area of the 3D axis-aligned box:
    //
    //   A = 2 * (w*h + w*d + h*d)
    //
    // where:
    // - w = width  (x extent)
    // - h = height (y extent)
    // - d = depth  (z extent)
    T width  = this->width();
    T height = this->height();
    T depth  = this->depth();
    return T(2) * (width * height + width * depth + height * depth);
}

template <typename T>
T
AxisAlignedBoundingBox<T>::width() const noexcept {
    // Extent along the x-axis.
    return upper_corner.x - lower_corner.x;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::height() const noexcept {
    // Extent along the y-axis.
    return upper_corner.y - lower_corner.y;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::depth() const noexcept {
    // Extent along the z-axis.
    return upper_corner.z - lower_corner.z;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::length(std::size_t axis) const noexcept {
    // Generic axis extent:
    //
    // axis = 0 -> x length
    // axis = 1 -> y length
    // axis = 2 -> z length
    //
    // Caller is expected to provide a valid axis index.
    return upper_corner[axis] - lower_corner[axis];
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::overlaps(const AxisAlignedBoundingBox& other) const noexcept {
    // Two AABBs do NOT overlap if there exists at least one axis where:
    // - this box lies completely before the other, or
    // - this box lies completely after the other.
    //
    // Component-wise test:
    //   upper_corner < other.lower_corner   -> separated on the low side
    //   lower_corner > other.upper_corner   -> separated on the high side
    //
    // If any axis is separating, there is no overlap.
    const bool overlap = math::any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));
    return !overlap;
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::contains(const Vector3<T>& point) const noexcept {
    // Point containment test with inclusive bounds:
    //
    // A point is considered inside if:
    //   lower_corner <= point <= upper_corner
    //
    // on every axis.
    return math::all((point >= lower_corner) & (point <= upper_corner));
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::intersects(const Ray<T>& ray) const noexcept {
    // Fast ray/AABB intersection test using the slab method.
    //
    // For each axis:
    // - compute the parametric interval [t0, t1] where the ray lies inside
    //   that axis-aligned slab
    // - intersect all slab intervals together
    //
    // If the final interval is non-empty, the ray intersects the box.
    T tmin = T(0);
    T tmax = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // If the ray is effectively parallel to this slab, then the origin
        // must already lie within the slab interval; otherwise there is no hit.
        if (std::abs(d) <= eps) {
            if (o < mn || o > mx) return false;
            continue;
        }

        // Parametric entry/exit values for this slab.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;

        // Ensure t0 <= t1 regardless of ray direction sign.
        if (t0 > t1) std::swap(t0, t1);

        // Intersect current slab interval with the running interval.
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);

        // Empty interval means no intersection.
        if (tmin > tmax) return false;
    }

    return true;
}

template <typename T>
AxisAlignedBoundingBoxRayIntersection<T>
AxisAlignedBoundingBox<T>::trace(const Ray<T>& ray) const noexcept {
    // Full ray/AABB intersection query using the slab method.
    //
    // In addition to the boolean result, this returns:
    // - enter : parametric entry distance
    // - exit  : parametric exit distance
    AxisAlignedBoundingBoxRayIntersection<T> isect {};

    T t_enter = T(0);
    T t_exit  = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // Parallel-to-slab case:
        // if origin is outside slab, there is no intersection at all.
        if (std::abs(d) <= eps) {
            if (o < mn || o > mx) return isect;
            continue;
        }

        // Parametric slab crossing interval for axis i.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;
        if (t0 > t1) std::swap(t0, t1);

        t_enter = std::max(t_enter, t0);
        t_exit  = std::min(t_exit, t1);

        // If entry is after exit, the total interval is empty.
        if (t_enter > t_exit) return isect;
    }

    // If the ray starts inside the box, the natural "entry" point is the origin.
    // In that case expose enter = 0.
    if (contains(ray.origin)) t_enter = T(0);

    isect.is_intersecting = true;
    isect.enter           = t_enter;
    isect.exit            = t_exit;
    return isect;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::center() const noexcept {
    // Geometric center of the box:
    //
    //   center = (lower_corner + upper_corner) / 2
    return (lower_corner + upper_corner) * T(0.5);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::extents() const noexcept {
    // Per-axis size vector:
    //
    //   (width, height, depth)
    return upper_corner - lower_corner;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length() const noexcept {
    // Euclidean length of the box diagonal:
    //
    //   || upper_corner - lower_corner ||
    return (upper_corner - lower_corner).length();
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length_squared() const noexcept {
    // Squared Euclidean length of the box diagonal.
    //
    // Useful when only relative comparisons are needed and square root can be avoided.
    return (upper_corner - lower_corner).length_squared();
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_valid() const noexcept {
    // Validity conditions:
    // 1. all coordinates must be finite
    // 2. lower_corner must not exceed upper_corner on any axis
    //
    // A valid AABB may still be degenerate (zero extent on one or more axes).
    return std::isfinite(lower_corner.x) && std::isfinite(lower_corner.y) && std::isfinite(lower_corner.z)
        && std::isfinite(upper_corner.x) && std::isfinite(upper_corner.y) && std::isfinite(upper_corner.z)
        && lower_corner.x <= upper_corner.x && lower_corner.y <= upper_corner.y && lower_corner.z <= upper_corner.z;
}

template <typename T>
void
AxisAlignedBoundingBox<T>::reset() noexcept {
    // Reset to the canonical empty/inverted state.
    //
    // This is intentionally NOT a valid non-empty box.
    // It is designed so that subsequent merge(point) / merge(box) calls
    // naturally initialize and expand the AABB.
    lower_corner = Vector3<T> { static_cast<T>(inf), static_cast<T>(inf), static_cast<T>(inf) };
    upper_corner = Vector3<T> { -static_cast<T>(inf), -static_cast<T>(inf), -static_cast<T>(inf) };
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const Vector3<T>& point) noexcept {
    // Expand this box so it also contains the given point.
    //
    // Component-wise:
    // - lower_corner takes the minimum
    // - upper_corner takes the maximum
    lower_corner = math::cmin(lower_corner, point);
    upper_corner = math::cmax(upper_corner, point);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const AxisAlignedBoundingBox& other) noexcept {
    // Expand this box so it also contains the other AABB.
    //
    // This computes the union of two axis-aligned bounding boxes.
    lower_corner = math::cmin(lower_corner, other.lower_corner);
    upper_corner = math::cmax(upper_corner, other.upper_corner);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::expand(T delta) noexcept {
    // Uniformly grow the box by `delta` along all axes in both directions.
    //
    // Result:
    // - lower_corner moves by (-delta, -delta, -delta)
    // - upper_corner moves by (+delta, +delta, +delta)
    //
    // Negative delta shrinks the box.
    const Vector3<T> d { delta, delta, delta };
    lower_corner -= d;
    upper_corner += d;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::corner(const std::size_t idx) const noexcept {
    // Return one of the 8 AABB corners selected by the three low bits of `idx`.
    //
    // Bit convention:
    // - bit 0 -> choose x: lower(0) / upper(1)
    // - bit 1 -> choose y: lower(0) / upper(1)
    // - bit 2 -> choose z: lower(0) / upper(1)
    //
    // Examples:
    // - idx = 0 -> (lx, ly, lz)
    // - idx = 7 -> (ux, uy, uz)
    const T x = (idx & 1) ? upper_corner.x : lower_corner.x;
    const T y = (idx & 2) ? upper_corner.y : lower_corner.y;
    const T z = (idx & 4) ? upper_corner.z : lower_corner.z;
    return Vector3<T>(x, y, z);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::clamp(const Vector3<T>& point) const noexcept {
    // Clamp a point into the closed AABB interval.
    //
    // Result:
    // - unchanged if already inside
    // - projected to the nearest box boundary component-wise if outside
    return math::clamp(point, lower_corner, upper_corner);
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_empty() const noexcept {
    // Treat the box as empty if any extent is non-positive:
    //
    //   upper_corner <= lower_corner
    //
    // on at least one axis.
    //
    // Note:
    // - this means zero-thickness boxes are considered empty by this function,
    //   even though they may still pass is_valid().
    return math::any(upper_corner <= lower_corner);
}

} // namespace atlas::spatial