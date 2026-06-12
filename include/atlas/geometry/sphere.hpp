#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <utility>

namespace atlas {

template <typename T>
Sphere<T>::Sphere() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1)) {
}

template <typename T>
Sphere<T>::Sphere(const Vector3<T>& center_, T radius_) noexcept
    : center(center_)
    , radius(radius_) {
}

template <typename T>
Sphere<T>::Sphere(const Sphere& other) noexcept
    : center(other.center)
    , radius(other.radius) {
}

template <typename T>
Sphere<T>::Sphere(Sphere&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius) {
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(const Sphere& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    center = other.center;
    radius = other.radius;

    return *this;
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(Sphere&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    radius = other.radius;

    return *this;
}

template <typename T>
SphereGeometryOperator<T>
Sphere<T>::make_sphere_operator() const noexcept {
    SphereGeometryOperator<T> op {};
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op;
}

template <typename T>
typename Sphere<T>::Builder
Sphere<T>::builder() noexcept {
    // Return a fresh builder for fluent sphere construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Sphere<T>::make_device_geometry_view() const {
    return GeometryOperator<T>(make_sphere_operator());
}

template <typename T>
atlas::Vector<T, 3>
Sphere<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return make_sphere_operator().closest_point(p);
}

template <typename T>
atlas::Vector<T, 3>
Sphere<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return make_sphere_operator().closest_normal(p);
}

template <typename T>
T
Sphere<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return make_sphere_operator().signed_distance(p);
}

template <typename T>
bool
Sphere<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_sphere_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Sphere<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_sphere_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
Sphere<T>::centroid() const noexcept {
    return make_sphere_operator().centroid();
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
Sphere<T>::bound() const noexcept {
    return make_sphere_operator().bound();
}

template <typename T>
bool
Sphere<T>::is_valid() const noexcept {
    return make_sphere_operator().is_valid();
}

template <typename T>
GeometryType
Sphere<T>::type() const noexcept {
    // Identify this geometry as a sphere.
    return GeometryType::Sphere;
}

template <typename T>
Sphere<T>
Sphere<T>::Builder::build() const {
    // Validate all builder parameters before constructing the final sphere.
    validate();

    Sphere<T> s {};
    s.center = _center;
    s.radius = _radius;

    return s;
}

template <typename T>
atlas::host_shared_ptr<Sphere<T>>
Sphere<T>::Builder::make_host_shared() const {
    // Build a validated sphere and store it in host-managed shared ownership.
    auto s = build();
    return atlas::make_host_shared<Sphere<T>>(std::move(s));
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_center(const Vector3<T>& c) noexcept {
    // Store the requested center for the later build() call.
    _center = c;
    return *this;
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_radius(T r) noexcept {
    // Store the requested radius; validate() enforces positivity later.
    _radius = r;
    return *this;
}

template <typename T>
void
Sphere<T>::Builder::validate() const {
    atlas::SphereGeometryOperator<T> op;

    // Validate through the same operator logic used by constructed Sphere instances.
    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);

    if (!op.is_valid()) {
        throw std::runtime_error("Sphere::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::Vector<T, 3>
SphereGeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    // Without valid sphere parameters, there is no meaningful projection target.
    if (!center || !radius) {
        return p;
    }

    // Vector from the sphere center to the query point.
    const atlas::Vector<T, 3> v = p - *center;
    const T e                         = std::numeric_limits<T>::epsilon();

    // Normalize the radial direction and scale it to the sphere radius.
    const atlas::Vector<T, 3> direction = atlas::normalized_or(
        v,
        atlas::Vector<T, 3>(T(1), T(0), T(0)),
        e);
    return (*center) + direction * (*radius);
}

template <typename T>
atlas::Vector<T, 3>
SphereGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    // Invalid geometry cannot provide a meaningful surface normal.
    if (!center || !radius) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The sphere normal is the normalized radial direction from center to point.
    const atlas::Vector<T, 3> v = p - *center;
    const T e                         = std::numeric_limits<T>::epsilon();

    return atlas::normalized_or(
        v,
        atlas::Vector<T, 3>(T(1), T(0), T(0)),
        e);
}

template <typename T>
T
SphereGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!center || !radius) {
        return std::numeric_limits<T>::infinity();
    }

    // Signed distance is radial distance from center minus the sphere radius.
    return (p - *center).length() - *radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!center || !radius) {
        return false;
    }

    // Expand the radius by tolerance to make boundary checks numerically robust.
    const T expanded_radius = *radius + tolerance;

    // A negative expanded radius cannot contain any point.
    if (expanded_radius < T(0)) {
        return false;
    }

    // Compare squared distances to avoid an unnecessary square root.
    return (p - *center).length_squared() <= expanded_radius * expanded_radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry or negative tolerances cannot accept surface points.
    if (!center || !radius || !(*radius > T(0)) || tolerance < T(0)) {
        return false;
    }

    const T outer_radius = *radius + tolerance;
    const T inner_radius = (*radius > tolerance) ? *radius - tolerance : T(0);
    const T d2           = (p - *center).length_squared();

    return d2 >= inner_radius * inner_radius
        && d2 <= outer_radius * outer_radius;
}

template <typename T>
atlas::Vector<T, 3>
SphereGeometryOperator<T>::centroid() const noexcept {
    // Missing center falls back to the origin as a neutral centroid.
    if (!center) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    // A sphere's centroid coincides with its center.
    return *center;
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
SphereGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!center || !radius) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    // Expand equally along all axes by the sphere radius.
    const atlas::Vector<T, 3> dr(*radius, *radius, *radius);

    return atlas::AxisAlignedBoundingBox<T>((*center) - dr, (*center) + dr);
}

template <typename T>
bool
SphereGeometryOperator<T>::is_valid() const noexcept {
    // Radius must be bound before validation can succeed.
    if (!radius) {
        return false;
    }

    // A valid sphere requires a strictly positive radius.
    return (*radius) > T(0);
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::trace(const atlas::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};

    // Invalid geometry produces a default non-intersecting hit result.
    if (!center || !radius) {
        return result;
    }

    const atlas::Vector<T, 3>& c = *center;
    const T r                          = *radius;

    // Express the ray origin relative to the sphere center.
    const atlas::Vector<T, 3> oc = ray.origin - c;

    // Build the quadratic equation for ray-sphere intersection.
    const T a  = ray.direction.length_squared();
    const T b  = T(2) * oc.dot(ray.direction);
    const T cc = oc.length_squared() - r * r;

    // Negative discriminant means the ray misses the sphere.
    const T disc = b * b - T(4) * a * cc;
    if (disc < T(0)) {
        return result;
    }

    const T sqrt_disc = atlas::sqrt_nonnegative(disc);
    const T inv2a     = T(0.5) / a;

    // Compute the near and far intersection distances.
    const T t0 = (-b - sqrt_disc) * inv2a;
    const T t1 = (-b + sqrt_disc) * inv2a;

    // Reject intersections that are fully behind the ray origin.
    if (t0 < T(0) && t1 < T(0)) {
        return result;
    }

    // Select the nearest non-negative intersection distance.
    T t = std::numeric_limits<T>::infinity();

    if (t0 >= T(0)) {
        t = t0;
    }

    if (t1 >= T(0) && t1 < t) {
        t = t1;
    }

    // Populate the hit record with the closest valid intersection.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);

    // Compute the outward normal from the hit point.
    result.normal = atlas::normalized_or(
        result.point - c,
        atlas::Vector<T, 3>(T(1), T(0), T(0)));

    return result;
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

} // namespace atlas
