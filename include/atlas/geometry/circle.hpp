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
    // If the operator is not fully bound to valid circle data, return the query point
    // unchanged as a conservative fallback.
    if (!center || !normal || !radius) return p;

    // Copy the normal locally because it may need to be normalized before use.
    atlas::math::Vector<T, 3> n = *normal;

    // Compute squared normal length once.
    const T n2 = n.length_squared();

    // Reject degenerate circles:
    // - zero-length normal means the supporting plane is undefined,
    // - non-positive radius means the disk geometry is invalid.
    if (n2 <= T(0) || *radius <= T(0)) return p;

    // Normalize the plane normal so all later projections use unit-length geometry.
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    // Compute the query point relative to the circle center.
    const atlas::math::Vector<T, 3> offset = p - *center;

    // Signed distance from the query point to the supporting plane.
    const T plane_distance = offset.dot(n);

    // Orthogonal projection of the query point into the plane of the circle.
    const atlas::math::Vector<T, 3> planar = offset - n * plane_distance;

    // Squared radial distance from the center within the supporting plane.
    const T planar_len2 = planar.length_squared();

    // Squared circle radius for comparison without an immediate square root.
    const T rr = (*radius) * (*radius);

    if (planar_len2 <= rr) {
        // If the projected point lies inside or on the circular disk,
        // the closest point is simply the orthogonal projection onto the plane.
        return p - n * plane_distance;
    }

    if (planar_len2 <= std::numeric_limits<T>::epsilon()) {
        // If the projected point is numerically at the center, the radial direction
        // is undefined. Fall back to an arbitrary point on the circle rim.
        //
        // This picks the +x direction in local/world coordinates as a deterministic fallback.
        return *center + atlas::math::Vector<T, 3>(*radius, T(0), T(0));
    }

    // For points whose projection lies outside the disk, the closest point lies
    // on the circular rim in the direction of the planar projection.
    const T planar_len = static_cast<T>(std::sqrt(planar_len2));
    return *center + planar * ((*radius) / planar_len);
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // If the operator is unbound, return a default +z normal as a safe fallback.
    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    // Compute squared length of the stored normal.
    const T n2 = normal->length_squared();

    // If the stored normal is degenerate, again fall back to +z.
    if (n2 <= T(0)) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    // Return the normalized supporting-plane normal.
    //
    // This operator treats the circle as an oriented planar disk, so the normal
    // does not vary across the surface.
    return (*normal) * (T(1) / static_cast<T>(std::sqrt(n2)));
}

template <typename T>
T
CircleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is unbound, report an infinite distance to indicate failure
    // to evaluate a meaningful geometric query.
    if (!center || !normal || !radius) return std::numeric_limits<T>::infinity();

    // Compute the closest point on the circular disk.
    const atlas::math::Vector<T, 3> cp = closest_point(p);

    // Compute the oriented disk normal.
    const atlas::math::Vector<T, 3> nn = closest_normal(p);

    // Unsigned Euclidean distance from the query point to the closest point.
    const T magnitude = (p - cp).length();

    // Determine on which side of the oriented supporting plane the query lies.
    const T side = (p - *center).dot(nn);

    // Use the plane side to assign a sign:
    // - non-negative side -> positive distance
    // - negative side     -> negative distance
    //
    // This is an oriented signed distance relative to the disk's plane normal,
    // not an "inside/outside" solid distance in the volumetric sense.
    return (side >= T(0)) ? magnitude : -magnitude;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Classify a point as inside when the signed distance does not exceed the
    // requested tolerance threshold.
    //
    // Because the sign is derived from the oriented plane side, this method
    // follows the operator's signed-distance convention.
    return signed_distance(p) <= tolerance;
}

template <typename T>
bool
CircleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the surface when its absolute signed distance
    // lies within the specified tolerance band around zero.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
CircleGeometryOperator<T>::centroid() const noexcept {
    // If the center pointer is missing, return the origin as a safe fallback.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // The centroid of a circular disk is its center.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CircleGeometryOperator<T>::bound() const noexcept {
    // Without complete circle data, return a default-constructed bounding box.
    if (!center || !normal || !radius) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    // Copy and validate the normal.
    atlas::math::Vector<T, 3> n = *normal;
    const T n2                  = n.length_squared();

    if (n2 <= T(0) || *radius <= T(0)) {
        // Degenerate circle data cannot define a proper disk extent.
        // Fall back to a zero-size box centered at the circle center.
        return atlas::spatial::AxisAlignedBoundingBox<T>(*center, *center);
    }

    // Normalize the circle normal so the axis-wise projected radius formula applies.
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    // Compute the maximal absolute extent of the disk along each world axis.
    //
    // For a unit normal component n.x, the projected radius onto x is:
    //   r * sqrt(1 - n.x^2)
    //
    // Analogous formulas apply to y and z.
    const atlas::math::Vector<T, 3> extent(
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.x * n.x))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.y * n.y))),
        (*radius) * static_cast<T>(std::sqrt(std::max(T(0), T(1) - n.z * n.z))));

    // The disk AABB is centered at the circle center and expanded by the
    // computed per-axis extents.
    return atlas::spatial::AxisAlignedBoundingBox<T>(*center - extent, *center + extent);
}

template <typename T>
bool
CircleGeometryOperator<T>::is_valid() const noexcept {
    // All required pointers must be bound.
    if (!center || !normal || !radius) return false;

    // A valid circle requires:
    // - finite center coordinates,
    // - finite normal coordinates,
    // - non-zero normal length,
    // - finite positive radius.
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
    // Initialize the result to the default "no intersection" state.
    HitSurface<T> result {};

    // Reject invalid geometry immediately.
    if (!is_valid()) return result;

    // Use the circle's oriented unit normal for plane intersection.
    const atlas::math::Vector<T, 3> nn = closest_normal(ray.origin);

    // Dot product between ray direction and plane normal.
    // This measures how strongly the ray moves toward or away from the disk plane.
    const T denom = nn.dot(ray.direction);

    // Use machine epsilon as the threshold for near-parallel tests.
    const T eps = std::numeric_limits<T>::epsilon();

    if (std::abs(denom) <= eps) {
        // The ray is parallel or nearly parallel to the disk plane.
        const T plane_distance = (ray.origin - *center).dot(nn);

        if (std::abs(plane_distance) > eps) {
            // Parallel ray but not lying in the plane -> no intersection.
            return result;
        }

        if ((ray.origin - *center).length_squared() > (*radius) * (*radius)) {
            // Ray origin lies in the plane but outside the disk footprint -> no hit.
            return result;
        }

        // The ray starts inside the disk plane region.
        // Report an immediate intersection at distance zero.
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = nn;
        return result;
    }

    // Solve for the ray parameter where the ray intersects the supporting plane.
    const T t = ((*center - ray.origin).dot(nn)) / denom;

    // Reject intersections behind the ray origin.
    if (t < T(0)) return result;

    // Compute the plane hit point.
    const atlas::math::Vector<T, 3> hit_point = ray.point_at(t);

    // The plane hit is valid for the disk only if it lies within the disk radius.
    if ((hit_point - *center).length_squared() > (*radius) * (*radius)) return result;

    // Record the successful disk intersection.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = hit_point;
    result.normal          = nn;
    return result;
}

template <typename T>
HitSurface<T>
CircleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Function-call convenience wrapper around `trace()`.
    return trace(ray);
}

template <typename T>
Circle<T>::Circle() noexcept {
    // Bind the cached geometry operator to this instance's member storage.
    //
    // This ensures that all later delegated geometry queries operate on
    // the current object's actual center/normal/radius data.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(const Vector3<T>& center_,
                  const Vector3<T>& normal_,
                  const T radius_) noexcept
    : center(center_)
    , normal(normal_)
    , radius(radius_) {
    // After initializing the circle parameters directly, bind the cached
    // operator so it references this object's members rather than temporary inputs.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(const Circle& other) noexcept
    : center(other.center)
    , normal(other.normal)
    , radius(other.radius) {
    // Rebind the cached operator after copying so it points to this object's
    // copied members, not the source object's members.
    bind_operator();
}

template <typename T>
Circle<T>::Circle(Circle&& other) noexcept
    : center(std::move(other.center))
    , normal(std::move(other.normal))
    , radius(other.radius) {
    // Bind this object's operator to the moved-in member state.
    bind_operator();

    // Rebind the moved-from object's operator as well so its internal pointers
    // remain self-consistent after the move.
    other.bind_operator();
}

template <typename T>
Circle<T>&
Circle<T>::operator=(const Circle& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy all geometric parameters from the source circle.
    center = other.center;
    normal = other.normal;
    radius = other.radius;

    // Rebind so the cached operator continues to reference this object's members.
    bind_operator();
    return *this;
}

template <typename T>
Circle<T>&
Circle<T>::operator=(Circle&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move or copy all geometric parameters from the source circle.
    center = std::move(other.center);
    normal = std::move(other.normal);
    radius = other.radius;

    // Rebind this object's operator to its current member addresses.
    bind_operator();

    // Rebind the moved-from object's operator to keep it internally consistent.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Circle<T>::bind_operator() noexcept {
    // Make the cached geometry operator reference this circle's actual member data.
    //
    // This must be refreshed after construction, copy, or move so delegated
    // queries always observe the correct storage addresses.
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.normal = atlas::raw_pointer_cast(&normal);
    _operator.radius = atlas::raw_pointer_cast(&radius);
}

template <typename T>
typename Circle<T>::Builder
Circle<T>::builder() noexcept {
    // Return a fresh builder object for staged circle construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Circle<T>::make_geometry_operator() const {
    // Wrap the cached circle-specific operator into the generic geometry operator interface.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-point query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-normal query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Circle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the signed-distance query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Circle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward inside classification to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Circle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward surface-band classification to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Circle<T>::centroid() const noexcept {
    // Forward centroid computation to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Circle<T>::bound() const noexcept {
    // Forward bounding-box computation to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Circle<T>::is_valid() const noexcept {
    // Forward validity testing to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Circle<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete geometry.
    return GeometryType::Circle;
}

template <typename T>
Circle<T>
Circle<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing the final circle.
    validate();

    // Start from a default-constructed circle so its cached operator is already
    // bound to its own member storage.
    Circle<T> circle {};

    // Overwrite the circle parameters with the validated builder state.
    circle.center = _center;
    circle.normal = _normal;
    circle.radius = _radius;

    // No extra bind step is required because the cached operator already points
    // to `circle`'s member storage, and only the values were overwritten.
    return circle;
}

template <typename T>
atlas::host_shared_ptr<Circle<T>>
Circle<T>::Builder::make_host_shared() const {
    // Build the validated circle by value first.
    auto circle = build();

    // Move the built circle into host-shared managed storage.
    return atlas::make_host_shared<Circle<T>>(std::move(circle));
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Store the circle center in the builder's staged state.
    _center = center_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store the circle plane normal in the builder's staged state.
    _normal = normal_;
    return *this;
}

template <typename T>
typename Circle<T>::Builder&
Circle<T>::Builder::with_radius(const T radius_) noexcept {
    // Store the circle radius in the builder's staged state.
    _radius = radius_;
    return *this;
}

template <typename T>
void
Circle<T>::Builder::validate() const {
    // Reuse the runtime geometry-operator validity logic so the definition
    // of a valid circle stays centralized in one place.
    CircleGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&_center);
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.radius = atlas::raw_pointer_cast(&_radius);

    // Reject invalid circle parameters with both a log message and an exception.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Circle::Builder validation failed: center/normal must be finite, normal must be non-zero, radius must be > 0.";
        throw std::runtime_error("Circle::Builder: invalid parameters.");
    }
}

} // namespace atlas::geometry