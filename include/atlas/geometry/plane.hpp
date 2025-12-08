#pragma once
#include <atlas/memory/raw_pointer_cast.h>
#include <cmath>
#include <limits>

namespace atlas::geometry {
template <typename T>
ATLAS_DEVICE HitSurface<T>

PlaneTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> result;
    if (!normal || !offset) return result;
    const T denom = math::dot(*normal, ray.direction);
    const T numer = -(math::dot(*normal, ray.origin) + *offset);
    if (denom == T(0)) {
        if (numer != T(0)) return result;
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = *normal;
        return result;
    }
    const T t = numer / denom;
    if (t < T(0)) return result;
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = *normal;
    return result;
}

template <typename T>
Plane<T>::Plane() noexcept
    : normal(T(0), T(0), T(1))
    , offset(T(0)) {
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& normal_, T offset_) noexcept
    : normal(normal_)
    , offset(offset_) {
    ATLAS_ASSERT(normal_.length() == T(1));
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept
    : normal(normal_)
    , offset(T(0)) {
    ATLAS_ASSERT(normal_.length() == T(1));
    offset = -(normal.dot(point));
}

template <typename T>
T
Plane<T>::signed_distance(const Vector3<T>& point) const {
    return normal.dot(point) + offset;
}

template <typename T>
Vector3<T>
Plane<T>::closest_point(const Vector3<T>& point) const {
    const T sd = signed_distance(point);
    return point - sd * normal;
}

template <typename T>
Vector3<T>
Plane<T>::closest_normal(const Vector3<T>&) const {
    return normal;
}

template <typename T>
T
Plane<T>::closest_distance(const Vector3<T>& point) const {
    const T sd = signed_distance(point);
    return (sd >= T(0)) ? sd : -sd;
}

template <typename T>
AABB<T>
Plane<T>::bound() const {
    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();
    return AABB<T>(Vector3<T>(lo, lo, lo), Vector3<T>(hi, hi, hi));
}

template <typename T>
bool
Plane<T>::intersects(const Ray<T>& ray) const {
    const T denom = normal.dot(ray.direction);
    const T numer = -(normal.dot(ray.origin) + offset);
    if (denom == T(0)) {
        return numer == T(0);
    }
    const T t = numer / denom;
    return t >= T(0);
}

template <typename T>
PlaneTraceOperator<T>
Plane<T>::make_trace_operator() const {
    PlaneTraceOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op;
}

template <typename T>
bool
Plane<T>::is_inside(const Vector3<T>& point) const {
    return signed_distance(point) <= T(0);
}

template <typename T>
void
Plane<T>::set_params_from_point_normal(const Vector3<T>& point, const Vector3<T>& normal_) noexcept {
    normal = normal_;
    ATLAS_ASSERT(normal_.length() == T(1));
    offset = -(normal.dot(point));
}

template <typename T>
void
Plane<T>::set_params_from_normal_offset(const Vector3<T>& normal_, T offset_) noexcept {
    normal = normal_;
    ATLAS_ASSERT(normal_.length() == T(1));
    offset = offset_;
}

template <typename T>
Vector3<T>
Plane<T>::extents() const noexcept {
    const T inf = std::numeric_limits<T>::infinity();
    return Vector3<T>(inf, inf, inf);
}

template <typename T>
bool
Plane<T>::is_valid() const noexcept {
    const T n2 = normal.length_squared();
    return (n2 > T(0)) && std::isfinite(static_cast<double>(offset));
}
}