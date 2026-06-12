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
atlas::Vector<T, 3>
SquareGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>&) const noexcept {
    // Missing normal falls back to a deterministic upward normal.
    if (!normal) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    return atlas::normalized_or(*normal, Vector3<T>(T(0), T(0), T(1)));
}

template <typename T>
T
SquareGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
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
SquareGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p,
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

    return atlas::abs(signed_plane_offset) <= tolerance
        && atlas::abs(u) <= half_side + tolerance
        && atlas::abs(v) <= half_side + tolerance;
}

template <typename T>
bool
SquareGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p,
                                         const T tolerance) const noexcept {
    // A square is a finite surface, so surface membership is equivalent to the inside test.
    return is_inside(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
SquareGeometryOperator<T>::centroid() const noexcept {
    // Missing center falls back to the origin as a neutral centroid.
    if (!center) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // A square's centroid coincides with its center.
    return *center;
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
SquareGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!center || !normal || !side_length) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    // A non-positive side length collapses the bound to the center point.
    if (!(*side_length > T(0))) {
        return atlas::AxisAlignedBoundingBox<T>(*center, *center);
    }

    Vector3<T> unit_normal;
    Vector3<T> tangent;
    Vector3<T> bitangent;

    // Build the square basis to compute projected axis-aligned extents.
    if (!build_basis(*normal, unit_normal, tangent, bitangent)) {
        return atlas::AxisAlignedBoundingBox<T>(*center, *center);
    }

    const T half_side = (*side_length) * T(0.5);

    // Compute the world-space AABB half extent induced by the two in-plane axes.
    const Vector3<T> extent = (atlas::abs(tangent) + atlas::abs(bitangent)) * half_side;

    return atlas::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
SquareGeometryOperator<T>::is_valid() const noexcept {
    // All parameter pointers must be bound before validation can succeed.
    if (!center || !normal || !side_length) {
        return false;
    }

    // Center, normal, and side length must be finite, with nonzero normal and positive size.
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
    if (atlas::abs(denominator) <= epsilon) {
        return hit;
    }

    // Solve for the ray parameter where the ray meets the square plane.
    const T distance = ((*center - ray.origin).dot(unit_normal)) / denominator;

    // Ignore intersections behind the ray origin.
    if (distance < T(0)) {
        return hit;
    }

    const Vector3<T> hit_point = ray.point_at(distance);

    const Vector3<T> center_to_hit = hit_point - *center;
    const T u                      = center_to_hit.dot(tangent);
    const T v                      = center_to_hit.dot(bitangent);
    const T half_side              = (*side_length) * T(0.5);

    // Reject plane hits that fall outside the finite square extent.
    if (atlas::abs(u) > half_side + epsilon || atlas::abs(v) > half_side + epsilon) {
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
SquareGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
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
    // Avoid unnecessary rebinding on self-assignment.
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
    // Avoid self move-assignment.
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
    // Return a fresh builder for fluent square construction.
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
    if (!atlas::isfinite(_center)
        || !atlas::isfinite(_normal)
        || !atlas::isfinite(_side_length)) {
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

} // namespace atlas
