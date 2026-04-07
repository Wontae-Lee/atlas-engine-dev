
#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <utility>

namespace atlas::geometry {

template <typename T>
Sphere<T>::Sphere() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1)) {
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(const Vector3<T>& center_, T radius_) noexcept
    : center(center_)
    , radius(radius_) {
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(const Sphere& other) noexcept
    : center(other.center)
    , radius(other.radius) {
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(Sphere&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius) {
    bind_operator();
    other.bind_operator();
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(const Sphere& other) noexcept {
    if (this == &other) return *this;

    center = other.center;
    radius = other.radius;
    bind_operator();
    return *this;
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(Sphere&& other) noexcept {
    if (this == &other) return *this;

    center = std::move(other.center);
    radius = other.radius;
    bind_operator();
    other.bind_operator();
    return *this;
}

template <typename T>
void
Sphere<T>::bind_operator() noexcept {
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.radius = atlas::raw_pointer_cast(&radius);
    this->invalidate_validity_cache();
}

template <typename T>
typename Sphere<T>::Builder
Sphere<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Sphere<T>::make_geometry_operator() const {
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_normal(p);
}

template <typename T>
T
Sphere<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.signed_distance(p);
}

template <typename T>
bool
Sphere<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Sphere<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::centroid() const noexcept {
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Sphere<T>::bound() const noexcept {
    return _operator.bound();
}

template <typename T>
bool
Sphere<T>::is_valid() const noexcept {
    return this->cached_is_valid([this]() noexcept {
        return _operator.is_valid();
    });
}

template <typename T>
GeometryType
Sphere<T>::type() const noexcept {

    return GeometryType::Sphere;
}

template <typename T>
Sphere<T>
Sphere<T>::Builder::build() const {

    validate();

    Sphere<T> s {};
    s.center = _center;
    s.radius = _radius;
    return s;
}

template <typename T>
atlas::host_shared_ptr<Sphere<T>>
Sphere<T>::Builder::make_host_shared() const {

    auto s = build();
    return atlas::make_host_shared<Sphere<T>>(std::move(s));
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_center(const Vector3<T>& c) noexcept {

    _center = c;
    return *this;
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_radius(T r) noexcept {

    _radius = r;
    return *this;
}

template <typename T>
void
Sphere<T>::Builder::validate() const {

    atlas::geometry::SphereGeometryOperator<T> op;

    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Sphere::Builder validation failed: radius must be > 0; center must be finite.";
        throw std::runtime_error("Sphere::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!center || !radius) return p;

    const atlas::math::Vector<T, 3> v = p - *center;
    const T len2                      = v.length_squared();
    const T e                         = std::numeric_limits<T>::epsilon();

    if (len2 <= e) {

        return atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y, (*center).z);
    }

    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return (*center) + v * ((*radius) * inv_len);
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!center || !radius) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    const atlas::math::Vector<T, 3> v = p - *center;
    const T len2                      = v.length_squared();
    const T e                         = std::numeric_limits<T>::epsilon();

    if (len2 <= e) return atlas::math::Vector<T, 3>(T(1), T(0), T(0));

    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return v * inv_len;
}

template <typename T>
T
SphereGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!center || !radius) return std::numeric_limits<T>::infinity();
    return (p - *center).length() - *radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    if (!center || !radius) return false;

    const T expanded_radius = *radius + tolerance;

    if (expanded_radius < T(0)) return false;

    return (p - *center).length_squared() <= expanded_radius * expanded_radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::centroid() const noexcept {

    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SphereGeometryOperator<T>::bound() const noexcept {

    if (!center || !radius) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> dr(*radius, *radius, *radius);
    return atlas::spatial::AxisAlignedBoundingBox<T>((*center) - dr, (*center) + dr);
}

template <typename T>
bool
SphereGeometryOperator<T>::is_valid() const noexcept {

    if (!radius) return false;
    return (*radius) > T(0);
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};
    if (!center || !radius) return result;

    const atlas::math::Vector<T, 3>& c = *center;
    const T r                          = *radius;
    const atlas::math::Vector<T, 3> oc = ray.origin - c;
    const T a                          = ray.direction.length_squared();
    const T b                          = T(2) * oc.dot(ray.direction);
    const T cc                         = oc.length_squared() - r * r;
    const T disc                       = b * b - T(4) * a * cc;
    if (disc < T(0)) return result;

    const T sqrt_disc = static_cast<T>(std::sqrt(disc));
    const T inv2a     = T(0.5) / a;
    const T t0        = (-b - sqrt_disc) * inv2a;
    const T t1        = (-b + sqrt_disc) * inv2a;
    if (t0 < T(0) && t1 < T(0)) return result;

    T t = std::numeric_limits<T>::infinity();
    if (t0 >= T(0)) t = t0;
    if (t1 >= T(0) && t1 < t) t = t1;

    result.is_intersecting      = true;
    result.distance             = t;
    result.point                = ray.point_at(t);
    atlas::math::Vector<T, 3> n = result.point - c;
    const T len2                = n.length_squared();
    if (len2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(len2)));
    else
        n = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    result.normal = n;
    return result;
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

}
