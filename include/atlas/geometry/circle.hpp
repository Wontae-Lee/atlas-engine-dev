#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without all circle parameters, there is no meaningful projection target.
    if (!center || !normal || !radius) {
        return p;
    }

    const T n2 = normal->length_squared();

    // Degenerate normals or non-positive radii cannot define a valid circle disk.
    if (n2 <= T(0) || *radius <= T(0)) {
        return p;
    }

    // Normalize the plane normal before computing the orthogonal projection.
    const atlas::math::Vector<T, 3> n = atlas::math::normalized_or(
        *normal,
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::math::Vector<T, 3> offset = p - *center;
    const T plane_distance                 = offset.dot(n);

    // Project the point offset onto the circle plane.
    const atlas::math::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2                    = planar.length_squared();
    const T rr                             = (*radius) * (*radius);

    // Points projected inside the disk use the direct plane projection.
    if (planar_len2 <= rr) {
        return p - n * plane_distance;
    }

    // If the projected direction is numerically undefined, choose a stable fallback point.
    if (planar_len2 <= std::numeric_limits<T>::epsilon()) {
        return *center + atlas::math::Vector<T, 3>(*radius, T(0), T(0));
    }

    // Clamp exterior projections to the circular boundary.
    const T planar_len = static_cast<T>(std::sqrt(planar_len2));
    return *center + planar * ((*radius) / planar_len);
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Fall back to the global z-axis when the stored normal is unavailable.
    if (!normal) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return atlas::math::normalized_or(
        *normal,
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));
}

template <typename T>
T
CircleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid circle parameters represent an infinitely distant surface.
    if (!center || !normal || !radius) {
        return std::numeric_limits<T>::infinity();
    }

    // Distance magnitude is measured to the closest point on the disk.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T magnitude                  = (p - cp).length();

    // The sign is determined by which side of the disk plane the point lies on.
    const T side = (p - *center).dot(nn);

    return (side >= T(0)) ? magnitude : -magnitude;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p,
                                     const T tolerance) const noexcept {
    // Invalid circle parameters cannot classify points.
    if (!center || !normal || !radius || !(*radius > T(0))) {
        return false;
    }

    const T n2 = normal->length_squared();

    if (!(n2 > T(0))) {
        return false;
    }

    const atlas::math::Vector<T, 3> n = atlas::math::normalized_or(
        *normal,
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::math::Vector<T, 3> offset = p - *center;
    const T plane_distance                 = offset.dot(n);
    const atlas::math::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2                    = planar.length_squared();
    const T rr                             = (*radius) * (*radius);
    const T tolerance2                     = tolerance * tolerance;

    T distance2 = plane_distance * plane_distance;

    if (planar_len2 > rr) {
        const T radial_distance = static_cast<T>(std::sqrt(planar_len2)) - *radius;
        distance2 += radial_distance * radial_distance;
    }

    if (!(plane_distance < T(0))) {
        return tolerance >= T(0) && distance2 <= tolerance2;
    }

    return tolerance >= T(0) || distance2 >= tolerance2;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p,
                                         const T tolerance) const noexcept {
    // Invalid circle parameters or negative tolerances cannot accept surface points.
    if (!center || !normal || !radius || !(*radius > T(0)) || tolerance < T(0)) {
        return false;
    }

    const T n2 = normal->length_squared();

    if (!(n2 > T(0))) {
        return false;
    }

    const atlas::math::Vector<T, 3> n = atlas::math::normalized_or(
        *normal,
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));

    const atlas::math::Vector<T, 3> offset = p - *center;
    const T plane_distance                 = offset.dot(n);
    const atlas::math::Vector<T, 3> planar = offset - n * plane_distance;
    const T planar_len2                    = planar.length_squared();
    const T rr                             = (*radius) * (*radius);
    const T tolerance2                     = tolerance * tolerance;

    T distance2 = plane_distance * plane_distance;

    if (planar_len2 > rr) {
        const T radial_distance = static_cast<T>(std::sqrt(planar_len2)) - *radius;
        distance2 += radial_distance * radial_distance;
    }

    return distance2 <= tolerance2;
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::centroid() const noexcept {
    // Invalid or unbound centers fall back to the origin.
    if (!center) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The centroid of a circle disk is its center.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CircleGeometryOperator<T>::bound() const noexcept {
    // Without all parameters, an axis-aligned bound cannot be constructed.
    if (!center || !normal || !radius) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    const T n2 = normal->length_squared();

    // Degenerate disks collapse to a point bound at the center.
    if (n2 <= T(0) || *radius <= T(0)) {
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    // Normalize the disk normal before projecting the radius onto world axes.
    const atlas::math::Vector<T, 3> n = atlas::math::normalized_or(
        *normal,
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));

    // Each AABB extent is the disk radius scaled by the projection onto that axis.
    const atlas::math::Vector<T, 3> extent(
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.x * n.x))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.y * n.y))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.z * n.z))));

    return atlas::spatial::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
CircleGeometryOperator<T>::is_valid() const noexcept {
    // A valid circle requires bound parameter storage.
    if (!center || !normal || !radius) {
        return false;
    }

    // All geometric parameters must be finite, with a non-zero normal and positive radius.
    return atlas::math::isfinite(*center)
        && atlas::math::isfinite(*normal)
        && normal->length_squared() > T(0)
        && std::isfinite(static_cast<double>(*radius))
        && *radius > T(0);
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};

    // Invalid disks produce a default non-intersecting hit result.
    if (!is_valid()) {
        return result;
    }

    const atlas::math::Vector<T, 3> nn = closest_normal(ray.origin);
    const T denom                      = nn.dot(ray.direction);
    const T eps                        = std::numeric_limits<T>::epsilon();

    if (std::abs(denom) <= eps) {
        // Parallel rays intersect only if they already lie on the circle plane.
        const T plane_distance = (ray.origin - *center).dot(nn);
        if (std::abs(plane_distance) > eps) {
            return result;
        }

        // A coplanar ray is accepted only when its origin lies inside the disk.
        if ((ray.origin - *center).length_squared() > (*radius) * (*radius)) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = nn;

        return result;
    }

    // Intersect the ray with the supporting plane of the disk.
    const T t = ((*center - ray.origin).dot(nn)) / denom;

    // Reject intersections behind the ray origin.
    if (t < T(0)) {
        return result;
    }

    const atlas::math::Vector<T, 3> hit_point = ray.point_at(t);

    // Reject plane hits outside the circular disk radius.
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
CircleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

template <typename T>
Circle<T>::Circle() noexcept {
    // Bind the geometry operator to this circle's parameter storage.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T radius_) noexcept
    : center(center_)
    , normal(normal_)
    , radius(radius_) {
    // Bind the geometry operator after initializing custom circle parameters.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(const Circle& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , radius(other.radius) {
    // Rebind the operator because copied raw pointers must point to this object.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(Circle&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , radius(other.radius) {
    // Rebind both objects so each operator points to its own parameter storage.
    bind_operator();
    other.bind_operator();
}

template <typename T>
Circle<T>&
Circle<T>::operator=(const Circle& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    center = other.center;
    normal = other.normal;
    radius = other.radius;

    // Rebind after assignment because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
Circle<T>&
Circle<T>::operator=(Circle&& other) noexcept {
    // Avoid unnecessary rebinding on self move-assignment.
    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    normal = std::move(other.normal);
    radius = other.radius;

    // Rebind both objects after moving parameter storage.
    bind_operator();
    other.bind_operator();

    return *this;
}

template <typename T>
void
Circle<T>::bind_operator() noexcept {
    // Keep the lightweight operator synchronized with this circle's parameter storage.
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.normal = atlas::raw_pointer_cast(&normal);
    _operator.radius = atlas::raw_pointer_cast(&radius);
}

template <typename T>
typename Circle<T>::Builder
Circle<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the circle fluently.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Circle<T>::make_geometry_operator() const {
    // Return a type-erased geometry operator backed by this circle operator.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound circle operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound circle operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Circle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound circle operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Circle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate inside tests to the bound circle operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Circle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface tests to the bound circle operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::centroid() const noexcept {
    // Delegate centroid queries to the bound circle operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Circle<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound circle operator.
    return _operator.bound();
}

template <typename T>
bool
Circle<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound circle operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Circle<T>::type() const noexcept {
    // Identify this geometry as a circle disk.
    return GeometryType::Circle;
}

template <typename T>
Circle<T>
Circle<T>::Builder::build() const {
    // Validate circle parameters before constructing the geometry.
    validate();

    Circle<T> circle {};
    circle.center = _center;
    circle.normal = _normal;
    circle.radius = _radius;

    // Rebind because the builder assigns parameters after default construction.
    circle.bind_operator();

    return circle;
}

template <typename T>
atlas::host_shared_ptr<Circle<T>>
Circle<T>::Builder::make_host_shared() const {
    // Build a validated circle and store it in host-managed shared ownership.
    auto circle = build();
    return atlas::make_host_shared<Circle<T>>(std::move(circle));
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Store the center point used to position the disk.
    _center = center_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store the disk normal used to define the supporting plane orientation.
    _normal = normal_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_radius(const T radius_) noexcept {
    // Store the positive radius used to define the disk extent.
    _radius = radius_;
    return *this;
}

template <typename T>
void
Circle<T>::Builder::validate() const {
    CircleGeometryOperator<T> op;

    // Validate using the same operator logic used by constructed Circle instances.
    op.center = atlas::raw_pointer_cast(&_center);
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.radius = atlas::raw_pointer_cast(&_radius);

    if (!op.is_valid()) {
        throw std::runtime_error("Circle::Builder: invalid parameters.");
    }
}

} // namespace atlas::geometry
