#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
atlas::Vector<T, 3>
CircleGeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !normal || !radius) {
        return p;
    }

    const T n2 = normal->length_squared();

    if (n2 <= T(0) || *radius <= T(0)) {
        return p;
    }

    const atlas::Vector<T, 3> n = atlas::normalized_or(
        *normal,
        atlas::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::Vector<T, 3> offset = p - *center;
    const T plane_distance           = offset.dot(n);

    const atlas::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2              = planar.length_squared();
    const T rr                       = (*radius) * (*radius);

    if (planar_len2 <= rr) {
        return p - n * plane_distance;
    }

    if (planar_len2 <= std::numeric_limits<T>::epsilon()) {
        return *center + atlas::Vector<T, 3>(*radius, T(0), T(0));
    }

    const T planar_len = atlas::sqrt_nonnegative(planar_len2);
    return *center + planar * ((*radius) / planar_len);
}

template <typename T>
atlas::Vector<T, 3>
CircleGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>&) const noexcept {

    if (!normal) {
        return atlas::Vector<T, 3>(T(0), T(0), T(1));
    }

    return atlas::normalized_or(
        *normal,
        atlas::Vector<T, 3>(T(0), T(0), T(1)));
}

template <typename T>
T
CircleGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !normal || !radius) {
        return std::numeric_limits<T>::infinity();
    }

    const atlas::Vector<T, 3> cp = closest_point(p);
    const atlas::Vector<T, 3> nn = closest_normal(p);
    const T magnitude            = (p - cp).length();

    const T side = (p - *center).dot(nn);

    return (side >= T(0)) ? magnitude : -magnitude;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p,
                                     const T tolerance) const noexcept {

    if (!center || !normal || !radius || !(*radius > T(0))) {
        return false;
    }

    const T n2 = normal->length_squared();

    if (!(n2 > T(0))) {
        return false;
    }

    const atlas::Vector<T, 3> n = atlas::normalized_or(
        *normal,
        atlas::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::Vector<T, 3> offset = p - *center;
    const T plane_distance           = offset.dot(n);
    const atlas::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2              = planar.length_squared();
    const T rr                       = (*radius) * (*radius);
    const T tolerance2               = tolerance * tolerance;

    T distance2 = plane_distance * plane_distance;

    if (planar_len2 > rr) {
        const T radial_distance = atlas::sqrt_nonnegative(planar_len2) - *radius;
        distance2 += radial_distance * radial_distance;
    }

    if (!(plane_distance < T(0))) {
        return tolerance >= T(0) && distance2 <= tolerance2;
    }

    return tolerance >= T(0) || distance2 >= tolerance2;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p,
                                         const T tolerance) const noexcept {

    if (!center || !normal || !radius || !(*radius > T(0)) || tolerance < T(0)) {
        return false;
    }

    const T n2 = normal->length_squared();

    if (!(n2 > T(0))) {
        return false;
    }

    const atlas::Vector<T, 3> n = atlas::normalized_or(
        *normal,
        atlas::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::Vector<T, 3> offset = p - *center;
    const T plane_distance           = offset.dot(n);
    const atlas::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2              = planar.length_squared();
    const T rr                       = (*radius) * (*radius);
    const T tolerance2               = tolerance * tolerance;

    T distance2 = plane_distance * plane_distance;

    if (planar_len2 > rr) {
        const T radial_distance = atlas::sqrt_nonnegative(planar_len2) - *radius;
        distance2 += radial_distance * radial_distance;
    }

    return distance2 <= tolerance2;
}

template <typename T>
atlas::Vector<T, 3>
CircleGeometryOperator<T>::centroid() const noexcept {

    if (!center) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    return *center;
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
CircleGeometryOperator<T>::bound() const noexcept {

    if (!center || !normal || !radius) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    const T n2 = normal->length_squared();

    if (n2 <= T(0) || *radius <= T(0)) {
        return atlas::AxisAlignedBoundingBox<T>(*center, *center);
    }

    const atlas::Vector<T, 3> n = atlas::normalized_or(
        *normal,
        atlas::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::Vector<T, 3> extent(
        (*radius) * atlas::sqrt_nonnegative(T(1) - n.x * n.x),
        (*radius) * atlas::sqrt_nonnegative(T(1) - n.y * n.y),
        (*radius) * atlas::sqrt_nonnegative(T(1) - n.z * n.z));

    return atlas::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
CircleGeometryOperator<T>::is_valid() const noexcept {

    if (!center || !normal || !radius) {
        return false;
    }

    return atlas::isfinite(*center)
        && atlas::isfinite(*normal)
        && normal->length_squared() > T(0)
        && atlas::isfinite(*radius)
        && *radius > T(0);
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::trace(const atlas::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};

    if (!is_valid()) {
        return result;
    }

    const atlas::Vector<T, 3> nn = closest_normal(ray.origin);
    const T denom                = nn.dot(ray.direction);
    const T eps                  = std::numeric_limits<T>::epsilon();

    if (atlas::abs(denom) <= eps) {

        const T plane_distance = (ray.origin - *center).dot(nn);
        if (atlas::abs(plane_distance) > eps) {
            return result;
        }

        if ((ray.origin - *center).length_squared() > (*radius) * (*radius)) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = nn;

        return result;
    }

    const T t = ((*center - ray.origin).dot(nn)) / denom;

    if (t < T(0)) {
        return result;
    }

    const atlas::Vector<T, 3> hit_point = ray.point_at(t);

    if ((hit_point - *center).length_squared() > (*radius) * (*radius)) {
        return result;
    }

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = hit_point;
    result.normal          = nn;

    return result;
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {

    return trace(ray);
}

template <typename T>
Circle<T>::Circle() noexcept = default;

template <typename T>
Circle<T>::Circle(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T radius_) noexcept
    : center(center_)
    , normal(normal_)
    , radius(radius_) {
}

template <typename T>
Circle<T>::Circle(const Circle& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , radius(other.radius) {
}

template <typename T>
Circle<T>::Circle(Circle&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , radius(other.radius) {
}

template <typename T>
Circle<T>&
Circle<T>::operator=(const Circle& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center = other.center;
    normal = other.normal;
    radius = other.radius;

    return *this;
}

template <typename T>
Circle<T>&
Circle<T>::operator=(Circle&& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    normal = std::move(other.normal);
    radius = other.radius;

    return *this;
}

template <typename T>
CircleGeometryOperator<T>
Circle<T>::make_circle_operator() const noexcept {
    CircleGeometryOperator<T> op {};
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op;
}

template <typename T>
typename Circle<T>::Builder
Circle<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Circle<T>::make_device_geometry_view() const {
    return GeometryOperator<T>(make_circle_operator());
}

template <typename T>
atlas::Vector<T, 3>
Circle<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return make_circle_operator().closest_point(p);
}

template <typename T>
atlas::Vector<T, 3>
Circle<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return make_circle_operator().closest_normal(p);
}

template <typename T>
T
Circle<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return make_circle_operator().signed_distance(p);
}

template <typename T>
bool
Circle<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_circle_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Circle<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_circle_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
Circle<T>::centroid() const noexcept {
    return make_circle_operator().centroid();
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
Circle<T>::bound() const noexcept {
    return make_circle_operator().bound();
}

template <typename T>
bool
Circle<T>::is_valid() const noexcept {
    return make_circle_operator().is_valid();
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
        throw std::runtime_error("Circle::Builder: invalid parameters.");
    }
}

}