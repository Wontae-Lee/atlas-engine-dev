#pragma once
#include <atlas/memory/raw_pointer_cast.h>
#include <cmath>
#include <limits>

namespace atlas::geometry {
template <typename T>
ATLAS_DEVICE HitSurface<T>

SphereTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> result;
    if (!center || !radius) return result;
    const Vector3<T> oc = ray.origin - *center;
    const T a           = ray.direction.length_squared();
    const T b           = T(2) * oc.dot(ray.direction);
    const T c           = oc.length_squared() - (*radius) * (*radius);
    const T disc        = b * b - T(4) * a * c;
    if (disc < T(0)) return result;
    const T sqrt_disc = static_cast<T>(std::sqrt(disc));
    const T inv2a     = T(0.5) / a;
    T t0              = (-b - sqrt_disc) * inv2a;
    T t1              = (-b + sqrt_disc) * inv2a;
    if (t0 < T(0) && t1 < T(0)) return result;
    T t = std::numeric_limits<T>::max();
    if (t0 >= T(0)) t = t0;
    if (t1 >= T(0) && t1 < t) t = t1;
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    Vector3<T> n           = result.point - *center;
    const T len2           = n.length_squared();
    if (len2 > T(0)) {
        n *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {
        n = Vector3<T>(T(1), T(0), T(0));
    }
    result.normal = n;
    return result;
}

template <typename T>
Sphere<T>::Sphere() noexcept
    : center(T(0), T(0), T(0))
      , radius(T(1)) {}

template <typename T>
Sphere<T>::Sphere(const Vector3<T>& center_, T radius_) noexcept
    : center(center_)
      , radius(radius_) {}

template <typename T>
T
Sphere<T>::signed_distance(const Vector3<T>& point) const {
    const Vector3<T> v = point - center;
    return v.length() - radius;
}

template <typename T>
Vector3<T>
Sphere<T>::closest_point(const Vector3<T>& point) const {
    const Vector3<T> v = point - center;
    const T len2       = v.length_squared();
    const T eps        = std::numeric_limits<T>::epsilon();
    if (len2 <= eps) {
        return Vector3<T>(center.x + radius, center.y, center.z);
    }
    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return center + v * (radius * inv_len);
}

template <typename T>
Vector3<T>
Sphere<T>::closest_normal(const Vector3<T>& point) const {
    const Vector3<T> v = point - center;
    const T len2       = v.length_squared();
    const T eps        = std::numeric_limits<T>::epsilon();
    if (len2 <= eps) {
        return Vector3<T>(T(1), T(0), T(0));
    }
    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return v * inv_len;
}

template <typename T>
T
Sphere<T>::closest_distance(const Vector3<T>& p) const {
    const T sd = signed_distance(p);
    return sd >= T(0) ? sd : -sd;
}

template <typename T>
AABB<T>
Sphere<T>::bound() const {
    const Vector3<T> dr(radius, radius, radius);
    return AABB<T>(center - dr, center + dr);
}

template <typename T>
bool
Sphere<T>::intersects(const Ray<T>& ray) const {
    const Vector3<T> oc = ray.origin - center;
    const T a           = ray.direction.length_squared();
    const T b           = T(2) * oc.dot(ray.direction);
    const T c           = oc.length_squared() - radius * radius;
    const T disc        = b * b - T(4) * a * c;
    if (disc < T(0)) return false;
    const T sqrt_disc = static_cast<T>(std::sqrt(disc));
    const T inv2a     = T(0.5) / a;
    const T t0        = (-b - sqrt_disc) * inv2a;
    const T t1        = (-b + sqrt_disc) * inv2a;
    return (t0 >= T(0)) || (t1 >= T(0));
}

template <typename T>
SphereTraceOperator<T>
Sphere<T>::make_trace_operator() const {
    SphereTraceOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op;
}

template <typename T>
bool
Sphere<T>::is_inside(const Vector3<T>& point) const {
    const Vector3<T> v = point - center;
    return v.length_squared() < radius * radius;
}

template <typename T>
void
Sphere<T>::set_params(const Vector3<T>& center_, T radius_) noexcept {
    center = center_;
    radius = radius_;
}

template <typename T>
Vector3<T>
Sphere<T>::extents() const noexcept {
    const T d = T(2) * radius;
    return Vector3<T>(d, d, d);
}

template <typename T>
bool
Sphere<T>::is_valid() const noexcept {
    return radius >= T(0);
}
}