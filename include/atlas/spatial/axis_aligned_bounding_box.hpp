#pragma once

#include <algorithm>
#include <cmath>

namespace atlas {

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox() noexcept {
    // Start from an empty inverted box.
    reset();
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const Vector3<T>& point1,
                                                  const Vector3<T>& point2) noexcept {
    // Build a valid box regardless of input point order.
    lower_corner = cmin(point1, point2);
    upper_corner = cmax(point1, point2);
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(
    const AxisAlignedBoundingBox& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
}

template <typename T>
T
AxisAlignedBoundingBox<T>::area() const noexcept {
    // Surface area of a rectangular box: 2(w h + w d + h d).
    T width  = this->width();
    T height = this->height();
    T depth  = this->depth();

    return T(2) * (width * height + width * depth + height * depth);
}

template <typename T>
T
AxisAlignedBoundingBox<T>::width() const noexcept {
    return upper_corner.x - lower_corner.x;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::height() const noexcept {
    return upper_corner.y - lower_corner.y;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::depth() const noexcept {
    return upper_corner.z - lower_corner.z;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::length(std::size_t axis) const noexcept {
    return upper_corner[axis] - lower_corner[axis];
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::overlaps(const AxisAlignedBoundingBox& other) const noexcept {
    // Boxes do not overlap if one is completely separated on any axis.
    const bool separated =
        any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));

    return !separated;
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::contains(const Vector3<T>& point) const noexcept {
    // A point is inside when it lies within all axis intervals.
    return all((point >= lower_corner) & (point <= upper_corner));
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::intersects(const Ray<T>& ray) const noexcept {
    // Slab interval of valid ray parameters.
    T tmin = T(0);
    T tmax = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // Parallel ray must already lie inside this axis interval.
        if (atlas::abs(d) <= eps) {
            if (o < mn || o > mx) return false;
            continue;
        }

        // Compute near and far intersection times for this axis slab.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;

        if (t0 > t1) std::swap(t0, t1);

        // Intersect this slab interval with the global ray interval.
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);

        if (tmin > tmax) return false;
    }

    return true;
}

template <typename T>
AxisAlignedBoundingBoxRayIntersection<T>
AxisAlignedBoundingBox<T>::trace(const Ray<T>& ray) const noexcept {
    AxisAlignedBoundingBoxRayIntersection<T> isect {};

    // Slab interval of entry and exit ray parameters.
    T t_enter = T(0);
    T t_exit  = inf;

    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];

        // Parallel ray misses if its origin is outside this axis interval.
        if (atlas::abs(d) <= eps) {
            if (o < mn || o > mx) return isect;
            continue;
        }

        // Compute near and far intersection times for this axis slab.
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;

        if (t0 > t1) std::swap(t0, t1);

        // Keep the latest entry and earliest exit across all slabs.
        t_enter = std::max(t_enter, t0);
        t_exit  = std::min(t_exit, t1);

        if (t_enter > t_exit) return isect;
    }

    // A ray starting inside the box enters at the current origin.
    if (contains(ray.origin)) t_enter = T(0);

    isect.is_intersecting = true;
    isect.enter           = t_enter;
    isect.exit            = t_exit;

    return isect;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::center() const noexcept {
    // Midpoint between opposite corners.
    return (lower_corner + upper_corner) * T(0.5);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::extents() const noexcept {
    // Full side lengths along x, y, and z.
    return upper_corner - lower_corner;
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length() const noexcept {
    return (upper_corner - lower_corner).length();
}

template <typename T>
T
AxisAlignedBoundingBox<T>::diagonal_length_squared() const noexcept {
    return (upper_corner - lower_corner).length_squared();
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_valid() const noexcept {
    // Valid boxes have finite coordinates and ordered corners.
    return atlas::isfinite(lower_corner)
        && atlas::isfinite(upper_corner)
        && atlas::all(lower_corner <= upper_corner);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::reset() noexcept {
    // Inverted infinite bounds represent an empty box before merging.
    lower_corner = Vector3<T> {
        static_cast<T>(inf),
        static_cast<T>(inf),
        static_cast<T>(inf)
    };
    upper_corner = Vector3<T> {
        -static_cast<T>(inf),
        -static_cast<T>(inf),
        -static_cast<T>(inf)
    };
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const Vector3<T>& point) noexcept {
    // Expand bounds to include the point.
    lower_corner = cmin(lower_corner, point);
    upper_corner = cmax(upper_corner, point);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const AxisAlignedBoundingBox& other) noexcept {
    // Expand bounds to include another box.
    lower_corner = cmin(lower_corner, other.lower_corner);
    upper_corner = cmax(upper_corner, other.upper_corner);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::expand(T delta) noexcept {
    // Grow the box uniformly in every direction.
    const Vector3<T> d { delta, delta, delta };

    lower_corner -= d;
    upper_corner += d;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::corner(const std::size_t idx) const noexcept {
    // Use index bits to select lower or upper coordinate on each axis.
    const T x = (idx & 1) ? upper_corner.x : lower_corner.x;
    const T y = (idx & 2) ? upper_corner.y : lower_corner.y;
    const T z = (idx & 4) ? upper_corner.z : lower_corner.z;

    return Vector3<T>(x, y, z);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::clamp(const Vector3<T>& point) const noexcept {
    // Project the point onto the box by clamping each coordinate.
    return clamp(point, lower_corner, upper_corner);
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_empty() const noexcept {
    // A box is empty when at least one axis has non-positive length.
    return any(upper_corner <= lower_corner);
}

} // namespace atlas
