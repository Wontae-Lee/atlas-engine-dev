#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
bool
SquareGeometryOperator<T>::build_basis(const atlas::Vector<T, 3>& input_normal,
                                       atlas::Vector<T, 3>& unit_normal,
                                       atlas::Vector<T, 3>& tangent,
                                       atlas::Vector<T, 3>& bitangent) const noexcept {
    return atlas::orthonormal_basis(input_normal, unit_normal, tangent, bitangent);
}

template <typename T>
atlas::Vector<T, 3>
SquareGeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !normal || !side_length) {
        return p;
    }

    if (!(*side_length > T(0))) {
        return p;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return p;
    }

    const T half_side                = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;

    const T signed_plane_offset      = center_to_point.dot(unit_normal);
    const Vector3<T> projected_point = p - unit_normal * signed_plane_offset;

    const Vector3<T> planar_offset = projected_point - *center;

    const T u = std::clamp(planar_offset.dot(tangent), -half_side, half_side);
    const T v = std::clamp(planar_offset.dot(bitangent), -half_side, half_side);

    return *center + tangent * u + bitangent * v;
}

template <typename T>
atlas::Vector<T, 3>
SquareGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>&) const noexcept {

    if (!normal) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    return atlas::normalized_or(*normal, Vector3<T>(T(0), T(0), T(1)));
}

template <typename T>
T
SquareGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !normal || !side_length) {
        return std::numeric_limits<T>::infinity();
    }

    const Vector3<T> projected_closest_point = closest_point(p);

    const Vector3<T> unit_normal = closest_normal(p);

    const T distance_magnitude = (p - projected_closest_point).length();

    const T sign_test = (p - *center).dot(unit_normal);

    return (sign_test >= T(0)) ? distance_magnitude : -distance_magnitude;
}

template <typename T>
bool
SquareGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p,
                                     const T tolerance) const noexcept {

    if (!center || !normal || !side_length || !(*side_length > T(0))) {
        return false;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return false;
    }

    const T half_side                = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;

    const T signed_plane_offset = center_to_point.dot(unit_normal);

    const T u = center_to_point.dot(tangent);
    const T v = center_to_point.dot(bitangent);

    return atlas::abs(signed_plane_offset) <= tolerance
        && atlas::abs(u) <= half_side + tolerance
        && atlas::abs(v) <= half_side + tolerance;
}

template <typename T>
bool
SquareGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p,
                                         const T tolerance) const noexcept {

    return is_inside(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
SquareGeometryOperator<T>::centroid() const noexcept {

    if (!center) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    return *center;
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
SquareGeometryOperator<T>::bound() const noexcept {

    if (!center || !normal || !side_length) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    if (!(*side_length > T(0))) {
        return atlas::AxisAlignedBoundingBox<T>(*center, *center);
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return atlas::AxisAlignedBoundingBox<T>(*center, *center);
    }

    const T half_side = (*side_length) * T(0.5);

    const Vector3<T> extent = (atlas::abs(tangent) + atlas::abs(bitangent)) * half_side;

    return atlas::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
SquareGeometryOperator<T>::is_valid() const noexcept {

    if (!center || !normal || !side_length) {
        return false;
    }

    return atlas::isfinite(*center)
        && atlas::isfinite(*normal)
        && normal->length_squared() > T(0)
        && atlas::isfinite(*side_length)
        && *side_length > T(0);
}

template <typename T>
HitSurface<T>
SquareGeometryOperator<T>::trace(const atlas::Ray<T>& ray) const noexcept {
    HitSurface<T> hit {};

    if (!is_valid()) {
        return hit;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return hit;
    }

    const T denominator = unit_normal.dot(ray.direction);
    const T epsilon     = std::numeric_limits<T>::epsilon();

    if (atlas::abs(denominator) <= epsilon) {
        return hit;
    }

    const T distance = ((*center - ray.origin).dot(unit_normal)) / denominator;

    if (distance < T(0)) {
        return hit;
    }

    const Vector3<T> hit_point = ray.point_at(distance);

    const Vector3<T> center_to_hit = hit_point - *center;
    const T u                      = center_to_hit.dot(tangent);
    const T v                      = center_to_hit.dot(bitangent);
    const T half_side              = (*side_length) * T(0.5);

    if (atlas::abs(u) > half_side + epsilon || atlas::abs(v) > half_side + epsilon) {
        return hit;
    }

    hit.is_intersecting = true;
    hit.distance        = distance;
    hit.point           = hit_point;
    hit.normal          = unit_normal;

    return hit;
}

template <typename T>
HitSurface<T>
SquareGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {

    return trace(ray);
}

template <typename T>
Square<T>::Square() noexcept = default;

template <typename T>
Square<T>::Square(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T side_length_) noexcept
    : center(center_)
    , normal(normal_)
    , side_length(side_length_) {
}

template <typename T>
Square<T>::Square(const Square& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , side_length(other.side_length) {
}

template <typename T>
Square<T>::Square(Square&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , side_length(other.side_length) {
}

template <typename T>
Square<T>&
Square<T>::operator=(const Square& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center      = other.center;
    normal      = other.normal;
    side_length = other.side_length;

    return *this;
}

template <typename T>
Square<T>&
Square<T>::operator=(Square&& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center      = std::move(other.center);
    normal      = std::move(other.normal);
    side_length = other.side_length;

    return *this;
}

template <typename T>
SquareGeometryOperator<T>
Square<T>::make_square_operator() const noexcept {
    SquareGeometryOperator<T> op {};
    op.center      = atlas::raw_pointer_cast(&center);
    op.normal      = atlas::raw_pointer_cast(&normal);
    op.side_length = atlas::raw_pointer_cast(&side_length);
    return op;
}

template <typename T>
typename Square<T>::Builder
Square<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Square<T>::make_device_geometry_view() const {
    return GeometryOperator<T>(make_square_operator());
}

template <typename T>
atlas::Vector<T, 3>
Square<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return make_square_operator().closest_point(p);
}

template <typename T>
atlas::Vector<T, 3>
Square<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return make_square_operator().closest_normal(p);
}

template <typename T>
T
Square<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return make_square_operator().signed_distance(p);
}

template <typename T>
bool
Square<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_square_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Square<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_square_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
Square<T>::centroid() const noexcept {
    return make_square_operator().centroid();
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
Square<T>::bound() const noexcept {
    return make_square_operator().bound();
}

template <typename T>
bool
Square<T>::is_valid() const noexcept {
    return make_square_operator().is_valid();
}

template <typename T>
GeometryType
Square<T>::type() const noexcept {

    return GeometryType::Square;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_center(const Vector3<T>& center_) noexcept {

    _center = center_;
    return *this;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {

    _normal = normal_;
    return *this;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_side_length(const T side_length_) noexcept {

    _side_length = side_length_;
    return *this;
}

template <typename T>
void
Square<T>::Builder::validate() const {

    if (!atlas::isfinite(_center)
        || !atlas::isfinite(_normal)
        || !atlas::isfinite(_side_length)) {
        throw std::runtime_error("Square::Builder: parameters must be finite.");
    }

    if (!(_normal.length_squared() > T(0))) {
        throw std::runtime_error("Square::Builder: normal must be non-zero.");
    }

    if (!(_side_length > T(0))) {
        throw std::runtime_error("Square::Builder: side_length must be positive.");
    }
}

template <typename T>
Square<T>
Square<T>::Builder::build() const {

    validate();

    return Square<T>(_center, _normal, _side_length);
}

template <typename T>
atlas::host_shared_ptr<Square<T>>
Square<T>::Builder::make_host_shared() const {

    auto square = build();
    return atlas::make_host_shared<Square<T>>(std::move(square));
}

}