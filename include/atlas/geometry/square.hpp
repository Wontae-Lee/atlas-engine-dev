#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::axis(const atlas::math::Vector<T, 3>& unit_like_normal) const noexcept {
    // Choose a reference axis that is not nearly parallel to the input normal.
    if (std::abs(unit_like_normal.z) < static_cast<T>(0.9)) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    // Use the y-axis as a fallback when the normal is close to the z-axis.
    return Vector3<T>(T(0), T(1), T(0));
}

template <typename T>
bool
SquareGeometryOperator<T>::build_basis(const atlas::math::Vector<T, 3>& input_normal,
                                       atlas::math::Vector<T, 3>& unit_normal,
                                       atlas::math::Vector<T, 3>& tangent,
                                       atlas::math::Vector<T, 3>& bitangent) const noexcept {
    // Start from the provided normal and normalize it into a unit normal.
    unit_normal = input_normal;

    const T normal_length_squared = unit_normal.length_squared();

    // A zero-length normal cannot define a square plane.
    if (!(normal_length_squared > T(0))) {
        return false;
    }

    unit_normal.normalize();

    // Construct the first in-plane basis vector using a safe reference axis.
    tangent = atlas::math::cross(axis(unit_normal), unit_normal);

    const T tangent_length_squared = tangent.length_squared();

    // Reject degenerate tangent construction.
    if (!(tangent_length_squared > T(0))) {
        return false;
    }

    tangent.normalize();

    // Construct the second in-plane basis vector orthogonal to both normal and tangent.
    bitangent = atlas::math::cross(unit_normal, tangent);

    const T bitangent_length_squared = bitangent.length_squared();

    // Reject degenerate bitangent construction.
    if (!(bitangent_length_squared > T(0))) {
        return false;
    }

    bitangent.normalize();

    return true;
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without valid square parameters, there is no meaningful projection target.
    if (!center || !normal || !side_length) {
        return p;
    }

    // A non-positive side length cannot define a valid square.
    if (!(*side_length > T(0))) {
        return p;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    // Build the local orthonormal basis of the square plane.
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return p;
    }

    const T half_side                = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;

    // Project the query point onto the square plane.
    const T signed_plane_offset      = center_to_point.dot(unit_normal);
    const Vector3<T> projected_point = p - unit_normal * signed_plane_offset;

    // Express the projected point in the square's local 2D basis.
    const Vector3<T> planar_offset = projected_point - *center;

    // Clamp local coordinates to the finite square extent.
    const T u = std::clamp(planar_offset.dot(tangent), -half_side, half_side);
    const T v = std::clamp(planar_offset.dot(bitangent), -half_side, half_side);

    // Reconstruct the closest point in world coordinates.
    return *center + tangent * u + bitangent * v;
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Missing normal falls back to a deterministic upward normal.
    if (!normal) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    const T normal_length_squared = normal->length_squared();

    // A zero-length normal cannot be normalized.
    if (!(normal_length_squared > T(0))) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    // Return the normalized square normal.
    return normal->normalized();
}

template <typename T>
T
SquareGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!center || !normal || !side_length) {
        return std::numeric_limits<T>::infinity();
    }

    // Closest point is computed on the finite square, including edge and corner clamping.
    const Vector3<T> projected_closest_point = closest_point(p);

    // Use the square normal to determine the sign of the distance.
    const Vector3<T> unit_normal = closest_normal(p);

    // Magnitude is the Euclidean distance to the closest finite-square point.
    const T distance_magnitude = (p - projected_closest_point).length();

    // Sign is based on which side of the square plane the query point lies.
    const T sign_test = (p - *center).dot(unit_normal);

    return (sign_test >= T(0)) ? distance_magnitude : -distance_magnitude;
}

template <typename T>
bool
SquareGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p,
                                     const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!center || !normal || !side_length || !(*side_length > T(0))) {
        return false;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    // Build the square's local coordinate frame.
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return false;
    }

    const T half_side                = (*side_length) * T(0.5);
    const Vector3<T> center_to_point = p - *center;

    // Plane offset must be within tolerance for the point to lie on the square surface.
    const T signed_plane_offset = center_to_point.dot(unit_normal);

    // Local in-plane coordinates determine whether the point lies within the square bounds.
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
    // A square is a finite surface, so surface membership is equivalent to the inside test.
    return is_inside(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
SquareGeometryOperator<T>::centroid() const noexcept {
    // Missing center falls back to the origin as a neutral centroid.
    if (!center) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // A square's centroid coincides with its center.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SquareGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!center || !normal || !side_length) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    // A non-positive side length collapses the bound to the center point.
    if (!(*side_length > T(0))) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    // Build the square basis to compute projected axis-aligned extents.
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    const T half_side = (*side_length) * T(0.5);

    // Compute the world-space AABB half extent induced by the two in-plane axes.
    const Vector3<T> extent(
        half_side * (std::abs(tangent.x) + std::abs(bitangent.x)),
        half_side * (std::abs(tangent.y) + std::abs(bitangent.y)),
        half_side * (std::abs(tangent.z) + std::abs(bitangent.z)));

    return atlas::spatial::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
SquareGeometryOperator<T>::is_valid() const noexcept {
    // All parameter pointers must be bound before validation can succeed.
    if (!center || !normal || !side_length) {
        return false;
    }

    // Center, normal, and side length must be finite, with nonzero normal and positive size.
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

    // Invalid square geometry produces a default non-intersecting result.
    if (!is_valid()) {
        return hit;
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    // Build the square's local basis before intersecting the supporting plane.
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return hit;
    }

    // Denominator determines whether the ray is parallel to the square plane.
    const T denominator = unit_normal.dot(ray.direction);
    const T epsilon     = std::numeric_limits<T>::epsilon();

    // Parallel or nearly parallel rays are treated as misses.
    if (std::abs(denominator) <= epsilon) {
        return hit;
    }

    // Solve for the ray parameter where the ray meets the square plane.
    const T distance = ((*center - ray.origin).dot(unit_normal)) / denominator;

    // Ignore intersections behind the ray origin.
    if (distance < T(0)) {
        return hit;
    }

    const Vector3<T> hit_point = ray.point_at(distance);

    // Reject plane hits that fall outside the finite square extent.
    if (!is_inside(hit_point, epsilon)) {
        return hit;
    }

    // Populate the hit record with the valid square intersection.
    hit.is_intersecting = true;
    hit.distance        = distance;
    hit.point           = hit_point;
    hit.normal          = unit_normal;

    return hit;
}

template <typename T>
HitSurface<T>
SquareGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

template <typename T>
Square<T>::Square() noexcept {
    // Bind the operator to this square's default parameter storage.
    bind_operator();
}

template <typename T>
Square<T>::Square(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T side_length_) noexcept
    : center(center_)
    , normal(normal_)
    , side_length(side_length_) {
    // Bind the operator after storing the user-provided square parameters.
    bind_operator();
}

template <typename T>
Square<T>::Square(const Square& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , side_length(other.side_length) {
    // Rebind the operator because copied raw pointers must refer to this object.
    bind_operator();
}

template <typename T>
Square<T>::Square(Square&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , side_length(other.side_length) {
    // Rebind this object after moving member storage.
    bind_operator();
}

template <typename T>
Square<T>&
Square<T>::operator=(const Square& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    center      = other.center;
    normal      = other.normal;
    side_length = other.side_length;

    // Rebind after assignment because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
Square<T>&
Square<T>::operator=(Square&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    center      = std::move(other.center);
    normal      = std::move(other.normal);
    side_length = other.side_length;

    // Rebind after moving because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
void
Square<T>::bind_operator() noexcept {
    // Store non-owning raw pointers to the square parameters used by the operator.
    _operator.center      = &center;
    _operator.normal      = &normal;
    _operator.side_length = &side_length;
}

template <typename T>
typename Square<T>::Builder
Square<T>::builder() noexcept {
    // Return a fresh builder for fluent square construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Square<T>::make_geometry_operator() const {
    // Wrap the concrete square operator in the generic geometry operator type.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound square operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound square operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Square<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound square operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Square<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate containment checks to the bound square operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Square<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-membership checks to the bound square operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Square<T>::centroid() const noexcept {
    // Delegate centroid computation to the bound square operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Square<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound square operator.
    return _operator.bound();
}

template <typename T>
bool
Square<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound square operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Square<T>::type() const noexcept {
    // Identify this geometry as a square.
    return GeometryType::Square;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Store the requested center for the later build() call.
    _center = center_;
    return *this;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store the requested normal for the later build() call.
    _normal = normal_;
    return *this;
}

template <typename T>
typename Square<T>::Builder&
Square<T>::Builder::with_side_length(const T side_length_) noexcept {
    // Store the requested side length; validate() enforces positivity later.
    _side_length = side_length_;
    return *this;
}

template <typename T>
void
Square<T>::Builder::validate() const {
    // All scalar and vector components must be finite before construction.
    if (!std::isfinite(static_cast<double>(_center.x))
        || !std::isfinite(static_cast<double>(_center.y))
        || !std::isfinite(static_cast<double>(_center.z))
        || !std::isfinite(static_cast<double>(_normal.x))
        || !std::isfinite(static_cast<double>(_normal.y))
        || !std::isfinite(static_cast<double>(_normal.z))
        || !std::isfinite(static_cast<double>(_side_length))) {
        throw std::runtime_error("Square::Builder: parameters must be finite.");
    }

    // The square plane requires a nonzero normal.
    if (!(_normal.length_squared() > T(0))) {
        throw std::runtime_error("Square::Builder: normal must be non-zero.");
    }

    // The finite square extent requires a strictly positive side length.
    if (!(_side_length > T(0))) {
        throw std::runtime_error("Square::Builder: side_length must be positive.");
    }
}

template <typename T>
Square<T>
Square<T>::Builder::build() const {
    // Validate all builder parameters before constructing the final square.
    validate();

    return Square<T>(_center, _normal, _side_length);
}

template <typename T>
atlas::host_shared_ptr<Square<T>>
Square<T>::Builder::make_host_shared() const {
    // Build a validated square and store it in host-managed shared ownership.
    auto square = build();
    return atlas::make_host_shared<Square<T>>(std::move(square));
}

} // namespace atlas::geometry
