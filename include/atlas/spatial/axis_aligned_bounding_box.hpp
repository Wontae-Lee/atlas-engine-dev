#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace atlas::spatial {
template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox() noexcept {
    reset();
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept {
    lower_corner = math::cmin(point1, point2);
    upper_corner = math::cmax(point1, point2);
}

template <typename T>
AxisAlignedBoundingBox<T>::AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
}

template <typename T>
T
AxisAlignedBoundingBox<T>::area() const noexcept {
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
    bool overlap = math::any((upper_corner < other.lower_corner) | (lower_corner > other.upper_corner));
    return !overlap;
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::contains(const Vector3<T>& point) const noexcept {
    return math::all((point >= lower_corner) & (point <= upper_corner));
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::intersects(const Ray<T>& ray) const noexcept {
    T tmin = T(0);
    T tmax = std::numeric_limits<T>::infinity();
    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];
        if (std::abs(d) <= std::numeric_limits<T>::epsilon()) {
            if (o < mn || o > mx) return false;
            continue;
        }
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;
        if (t0 > t1) std::swap(t0, t1);
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
    T t_enter = T(0);
    T t_exit  = std::numeric_limits<T>::infinity();
    for (int i = 0; i < 3; ++i) {
        const T o  = ray.origin[i];
        const T d  = ray.direction[i];
        const T mn = lower_corner[i];
        const T mx = upper_corner[i];
        if (std::abs(d) <= std::numeric_limits<T>::epsilon()) {
            if (o < mn || o > mx) return isect;
            continue;
        }
        T t0 = (mn - o) / d;
        T t1 = (mx - o) / d;
        if (t0 > t1) std::swap(t0, t1);
        t_enter = std::max(t_enter, t0);
        t_exit  = std::min(t_exit, t1);
        if (t_enter > t_exit) return isect;
    }
    if (contains(ray.origin)) t_enter = T(0);
    isect.is_intersecting = true;
    isect.enter           = t_enter;
    isect.exit            = t_exit;
    return isect;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::center() const noexcept {
    return (lower_corner + upper_corner) * T(0.5);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::extents() const noexcept {
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
void
AxisAlignedBoundingBox<T>::reset() noexcept {
    const T M    = std::numeric_limits<T>::max();
    lower_corner = Vector3<T> { M, M, M };
    upper_corner = Vector3<T> { -M, -M, -M };
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const Vector3<T>& point) noexcept {
    lower_corner = math::cmin(lower_corner, point);
    upper_corner = math::cmax(upper_corner, point);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::merge(const AxisAlignedBoundingBox& other) noexcept {
    lower_corner = math::cmin(lower_corner, other.lower_corner);
    upper_corner = math::cmax(upper_corner, other.upper_corner);
}

template <typename T>
void
AxisAlignedBoundingBox<T>::expand(T delta) noexcept {
    const Vector3<T> d { delta, delta, delta };
    lower_corner -= d;
    upper_corner += d;
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::corner(std::size_t idx) const noexcept {
    const T x = (idx & 1) ? upper_corner.x : lower_corner.x;
    const T y = (idx & 2) ? upper_corner.y : lower_corner.y;
    const T z = (idx & 4) ? upper_corner.z : lower_corner.z;
    return Vector3<T>(x, y, z);
}

template <typename T>
Vector3<T>
AxisAlignedBoundingBox<T>::clamp(const Vector3<T>& point) const noexcept {
    return math::clamp(point, lower_corner, upper_corner);
}

template <typename T>
bool
AxisAlignedBoundingBox<T>::is_empty() const noexcept {
    return math::any(upper_corner <= lower_corner);
}
}