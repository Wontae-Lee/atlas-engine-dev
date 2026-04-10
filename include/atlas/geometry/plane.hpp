#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Plane<T>::Plane() noexcept
    : normal(T(0), T(0), T(1))
    , offset(T(0)) {
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& normal_, T offset_) noexcept
    : normal(normal_)
    , offset(offset_) {
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept
    : normal(normal_)
    , offset(-(normal_.dot(point))) {
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Plane& other) noexcept
    : normal(other.normal)
    , offset(other.offset) {
    bind_operator();
}

template <typename T>
Plane<T>::Plane(Plane&& other) noexcept
    : normal(std::move(other.normal))
    , offset(other.offset) {
    bind_operator();
    other.bind_operator();
}

template <typename T>
Plane<T>&
Plane<T>::operator=(const Plane& other) noexcept {
    if (this == &other) return *this;

    normal = other.normal;
    offset = other.offset;
    bind_operator();
    return *this;
}

template <typename T>
Plane<T>&
Plane<T>::operator=(Plane&& other) noexcept {
    if (this == &other) return *this;

    normal = std::move(other.normal);
    offset = other.offset;
    bind_operator();
    other.bind_operator();
    return *this;
}

template <typename T>
void
Plane<T>::bind_operator() noexcept {
    _operator.normal = atlas::raw_pointer_cast(&normal);
    _operator.offset = atlas::raw_pointer_cast(&offset);
}

template <typename T>
typename Plane<T>::Builder
Plane<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Plane<T>::make_geometry_operator() const {
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_normal(p);
}

template <typename T>
T
Plane<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.signed_distance(p);
}

template <typename T>
bool
Plane<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Plane<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::centroid() const noexcept {
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Plane<T>::bound() const noexcept {
    return _operator.bound();
}

template <typename T>
bool
Plane<T>::is_valid() const noexcept {
    return _operator.is_valid();
}

template <typename T>
GeometryType
Plane<T>::type() const noexcept {

    return GeometryType::Plane;
}

template <typename T>
Plane<T>
Plane<T>::Builder::build() const {

    validate();

    Plane<T> p {};

    p.normal = _normal;
    p.offset = _offset;

    return p;
}

template <typename T>
atlas::host_shared_ptr<Plane<T>>
Plane<T>::Builder::make_host_shared() const {

    auto p = build();
    return atlas::make_host_shared<Plane<T>>(std::move(p));
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {

    _normal = normal_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_offset(T offset_) noexcept {

    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept {

    _normal = normal_;
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_point_normal(const Vector3<T>& point,
                                     const Vector3<T>& normal_) noexcept {

    _normal = normal_;
    _offset = -(normal_.dot(point));
    return *this;
}

template <typename T>
void
Plane<T>::Builder::validate() const {

    atlas::geometry::PlaneGeometryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.offset = atlas::raw_pointer_cast(&_offset);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Plane::Builder validation failed: normal must be finite and non-zero; offset must be finite.";
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!normal || !offset) return p;

    const T sdev = ((*normal).dot(p) + (*offset));
    return p - sdev * (*normal);
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {

    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *normal;
}

template <typename T>
T
PlaneGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!normal || !offset) return std::numeric_limits<T>::infinity();
    return (*normal).dot(p) + (*offset);
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    if (!normal || !offset) return false;

    return (*normal).dot(p) + (*offset) <= tolerance;
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::centroid() const noexcept {

    return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
PlaneGeometryOperator<T>::bound() const noexcept {

    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>(lo, lo, lo),
        atlas::math::Vector<T, 3>(hi, hi, hi));
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_valid() const noexcept {

    if (!normal || !offset) return false;

    const T n2 = (*normal).length_squared();
    return (n2 > T(0)) && std::isfinite(static_cast<double>(*offset));
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};
    if (!normal || !offset) return result;

    const atlas::math::Vector<T, 3>& n = *normal;
    const T d                          = *offset;
    const T denom                      = n.dot(ray.direction);
    const T numer                      = -(n.dot(ray.origin) + d);

    if (denom == T(0)) {
        if (numer != T(0)) return result;
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = n;
        return result;
    }

    const T t = numer / denom;
    if (t < T(0)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

}
