#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {
template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::helper_axis(const atlas::math::Vector<T, 3>& unit_like_normal) const noexcept {
    if (std::abs(unit_like_normal.z) < static_cast<T>(0.9)) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    return Vector3<T>(T(0), T(1), T(0));
}

template <typename T>
bool
SquareGeometryOperator<T>::build_basis(const atlas::math::Vector<T, 3>& input_normal,
                                       atlas::math::Vector<T, 3>& unit_normal,
                                       atlas::math::Vector<T, 3>& tangent,
                                       atlas::math::Vector<T, 3>& bitangent) const noexcept {
    unit_normal = input_normal;
    const T normal_length_squared = unit_normal.length_squared();
    if (!(normal_length_squared > T(0))) {
        return false;
    }

    unit_normal *= T(1) / static_cast<T>(std::sqrt(normal_length_squared));

    tangent = atlas::math::cross(helper_axis(unit_normal), unit_normal);
    const T tangent_length_squared = tangent.length_squared();
    if (!(tangent_length_squared > T(0))) {
        return false;
    }

    tangent *= T(1) / static_cast<T>(std::sqrt(tangent_length_squared));

    bitangent = atlas::math::cross(unit_normal, tangent);
    const T bitangent_length_squared = bitangent.length_squared();
    if (!(bitangent_length_squared > T(0))) {
        return false;
    }

    bitangent *= T(1) / static_cast<T>(std::sqrt(bitangent_length_squared));
    return true;
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
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

    const T half_side = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;
    const T signed_plane_offset = center_to_point.dot(unit_normal);
    const Vector3<T> projected_point = p - unit_normal * signed_plane_offset;
    const Vector3<T> planar_offset = projected_point - *center;

    const T u = std::clamp(planar_offset.dot(tangent), -half_side, half_side);
    const T v = std::clamp(planar_offset.dot(bitangent), -half_side, half_side);

    return *center + tangent * u + bitangent * v;
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    if (!normal) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    const T normal_length_squared = normal->length_squared();
    if (!(normal_length_squared > T(0))) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    return (*normal) * (T(1) / static_cast<T>(std::sqrt(normal_length_squared)));
}

template <typename T>
T
SquareGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
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
SquareGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p,
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

    const T half_side = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;
    const T signed_plane_offset = center_to_point.dot(unit_normal);
    const T u = center_to_point.dot(tangent);
    const T v = center_to_point.dot(bitangent);

    return std::abs(signed_plane_offset) <= tolerance
        && std::abs(u) <= half_side + tolerance
        && std::abs(v) <= half_side + tolerance;
}

template <typename T>
bool
SquareGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p,
                                         const T tolerance) const noexcept {
    return is_inside(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::centroid() const noexcept {
    if (!center) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SquareGeometryOperator<T>::bound() const noexcept {
    if (!center || !normal || !side_length) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    if (!(*side_length > T(0))) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    const T half_side = (*side_length) * T(0.5);
    const Vector3<T> extent(
        half_side * (std::abs(tangent.x) + std::abs(bitangent.x)),
        half_side * (std::abs(tangent.y) + std::abs(bitangent.y)),
        half_side * (std::abs(tangent.z) + std::abs(bitangent.z)));

    return atlas::spatial::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
SquareGeometryOperator<T>::is_valid() const noexcept {
    if (!center || !normal || !side_length) {
        return false;
    }

    return std::isfinite(static_cast<double>(center->x))
        && std::isfinite(static_cast<double>(center->y))
        && std::isfinite(static_cast<double>(center->z))
        && std::isfinite(static_cast<double>(normal->x))
        && std::isfinite(static_cast<double>(normal->y))
        && std::isfinite(static_cast<double>(normal->z))
        && normal->length_squared() > T(0)
        && std::isfinite(static_cast<double>(*side_length))
        && *side_length > T(0);
}

template <typename T>
HitSurface<T>
SquareGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
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
    const T epsilon = std::numeric_limits<T>::epsilon();

    if (std::abs(denominator) <= epsilon) {
        return hit;
    }

    const T distance = ((*center - ray.origin).dot(unit_normal)) / denominator;
    if (distance < T(0)) {
        return hit;
    }

    const Vector3<T> hit_point = ray.point_at(distance);
    if (!is_inside(hit_point, epsilon)) {
        return hit;
    }

    hit.is_intersecting = true;
    hit.distance = distance;
    hit.point = hit_point;
    hit.normal = unit_normal;
    return hit;
}

template <typename T>
HitSurface<T>
SquareGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

template <typename T>
Square<T>::Square() noexcept {
    bind_operator();
}

template <typename T>
Square<T>::Square(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T side_length_) noexcept
    : center(center_)
    , normal(normal_)
    , side_length(side_length_) {
    bind_operator();
}

template <typename T>
Square<T>::Square(const Square& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , side_length(other.side_length) {
    bind_operator();
}

template <typename T>
Square<T>::Square(Square&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , side_length(other.side_length) {
    bind_operator();
}

template <typename T>
Square<T>&
Square<T>::operator=(const Square& other) noexcept {
    if (this == &other) {
        return *this;
    }

    center = other.center;
    normal = other.normal;
    side_length = other.side_length;
    bind_operator();
    return *this;
}

template <typename T>
Square<T>&
Square<T>::operator=(Square&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    normal = std::move(other.normal);
    side_length = other.side_length;
    bind_operator();
    return *this;
}

template <typename T>
void
Square<T>::bind_operator() noexcept {
    _operator.center = &center;
    _operator.normal = &normal;
    _operator.side_length = &side_length;
}

template <typename T>
typename Square<T>::Builder
Square<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Square<T>::make_geometry_operator() const {
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_normal(p);
}

template <typename T>
T
Square<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.signed_distance(p);
}

template <typename T>
bool
Square<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Square<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::centroid() const noexcept {
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Square<T>::bound() const noexcept {
    return _operator.bound();
}

template <typename T>
bool
Square<T>::is_valid() const noexcept {
    return _operator.is_valid();
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
    if (!std::isfinite(static_cast<double>(_center.x))
        || !std::isfinite(static_cast<double>(_center.y))
        || !std::isfinite(static_cast<double>(_center.z))
        || !std::isfinite(static_cast<double>(_normal.x))
        || !std::isfinite(static_cast<double>(_normal.y))
        || !std::isfinite(static_cast<double>(_normal.z))
        || !std::isfinite(static_cast<double>(_side_length))) {
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

} // namespace atlas::geometry
