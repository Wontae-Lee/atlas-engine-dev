#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Circle<T>::Circle() noexcept = default;

template <typename T>
Circle<T>::Circle(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T radius_) noexcept
    : center(center_)
    , normal(normal_)
    , radius(radius_) { }

template <typename T>
typename Circle<T>::Builder
Circle<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Circle<T>::make_geometry_operator() const {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.closest_normal(p);
}

template <typename T>
T
Circle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.signed_distance(p);
}

template <typename T>
bool
Circle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    atlas::geometry::CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.is_inside(p, tolerance);
}

template <typename T>
bool
Circle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    atlas::geometry::CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::centroid() const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Circle<T>::bound() const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.bound();
}

template <typename T>
bool
Circle<T>::is_valid() const noexcept {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.is_valid();
}

template <typename T>
GeometryType
Circle<T>::type() const noexcept {
    return GeometryType::Circle;
}

template <typename T>
Circle<T>
Circle<T>::Builder::build() const {
    validate();

    Circle<T> circle {};
    circle.center = _center;
    circle.normal = _normal;
    circle.radius = _radius;
    return circle;
}

template <typename T>
atlas::host_shared_ptr<Circle<T>>
Circle<T>::Builder::make_host_shared() const {
    auto circle = build();
    return atlas::make_host_shared<Circle<T>>(std::move(circle));
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    _center = center_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    _normal = normal_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_radius(const T radius_) noexcept {
    _radius = radius_;
    return *this;
}

template <typename T>
void
Circle<T>::Builder::validate() const {
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&_center);
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.radius = atlas::raw_pointer_cast(&_radius);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Circle::Builder validation failed: center/normal must be finite, normal must be non-zero, radius must be > 0.";
        throw std::runtime_error("Circle::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    if (!center || !normal || !radius) return p;

    atlas::math::Vector<T, 3> n = *normal;
    const T n2                  = n.length_squared();
    if (n2 <= T(0) || *radius <= T(0)) return p;
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    const atlas::math::Vector<T, 3> offset = p - *center;
    const T plane_distance                 = offset.dot(n);
    const atlas::math::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2                    = planar.length_squared();
    const T rr                             = (*radius) * (*radius);

    if (planar_len2 <= rr) {
        return p - n * plane_distance;
    }

    if (planar_len2 <= std::numeric_limits<T>::epsilon()) {
        return *center + atlas::math::Vector<T, 3>(*radius, T(0), T(0));
    }

    const T planar_len = static_cast<T>(std::sqrt(planar_len2));
    return *center + planar * ((*radius) / planar_len);
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    const T n2 = normal->length_squared();
    if (n2 <= T(0)) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    return (*normal) * (T(1) / static_cast<T>(std::sqrt(n2)));
}

template <typename T>
T
CircleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    if (!center || !normal || !radius) return std::numeric_limits<T>::infinity();

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T magnitude                  = (p - cp).length();
    const T side                       = (p - *center).dot(nn);
    return (side >= T(0)) ? magnitude : -magnitude;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return signed_distance(p) <= tolerance;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::centroid() const noexcept {
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CircleGeometryOperator<T>::bound() const noexcept {
    if (!center || !normal || !radius) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    atlas::math::Vector<T, 3> n = *normal;
    const T n2                  = n.length_squared();
    if (n2 <= T(0) || *radius <= T(0)) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    n *= T(1) / static_cast<T>(std::sqrt(n2));
    const atlas::math::Vector<T, 3> extent(
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.x * n.x))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.y * n.y))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.z * n.z))));

    return atlas::spatial::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
CircleGeometryOperator<T>::is_valid() const noexcept {
    if (!center || !normal || !radius) return false;

    return std::isfinite(static_cast<double>(center->x))
        && std::isfinite(static_cast<double>(center->y))
        && std::isfinite(static_cast<double>(center->z))
        && std::isfinite(static_cast<double>(normal->x))
        && std::isfinite(static_cast<double>(normal->y))
        && std::isfinite(static_cast<double>(normal->z))
        && normal->length_squared() > T(0)
        && std::isfinite(static_cast<double>(*radius))
        && *radius > T(0);
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};
    if (!is_valid()) return result;

    const atlas::math::Vector<T, 3> nn = closest_normal(ray.origin);
    const T denom                      = nn.dot(ray.direction);
    const T eps                        = std::numeric_limits<T>::epsilon();

    if (std::abs(denom) <= eps) {
        const T plane_distance = (ray.origin - *center).dot(nn);
        if (std::abs(plane_distance) > eps) return result;
        if ((ray.origin - *center).length_squared() > (*radius) * (*radius)) return result;

        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = nn;
        return result;
    }

    const T t = ((*center - ray.origin).dot(nn)) / denom;
    if (t < T(0)) return result;

    const atlas::math::Vector<T, 3> hit_point = ray.point_at(t);
    if ((hit_point - *center).length_squared() > (*radius) * (*radius)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = hit_point;
    result.normal          = nn;
    return result;
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

}
