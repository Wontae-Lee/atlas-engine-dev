#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <utility>

namespace atlas::geometry {

template <typename T>
Sphere<T>::Sphere() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1)) {
    // Construct a default sphere centered at the origin with unit radius.
    //
    // Default parameters:
    // - center = (0, 0, 0)
    // - radius = 1
    //
    // Bind the cached operator so all delegated geometry queries reference
    // this object's member storage.
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(const Vector3<T>& center_, T radius_) noexcept
    : center(center_)
    , radius(radius_) {
    // Construct a sphere directly from caller-provided center and radius.
    //
    // Bind the cached operator so the internal query adapter points to this
    // sphere's actual member data.
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(const Sphere& other) noexcept
    : center(other.center)
    , radius(other.radius) {
    // Copy the sphere parameters from another instance.
    //
    // Rebind the cached operator because it must refer to this object's
    // members rather than the source object's members.
    bind_operator();
}

template <typename T>
Sphere<T>::Sphere(Sphere&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius) {
    // Move or copy the sphere parameters into this object.
    //
    // Rebind this object's cached operator to the moved-in members.
    bind_operator();

    // Rebind the moved-from object's operator as well so its internal
    // pointers remain consistent after the move.
    other.bind_operator();
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(const Sphere& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy the sphere parameters from the source object.
    center = other.center;
    radius = other.radius;

    // Rebind the cached operator so it points to this object's members.
    bind_operator();
    return *this;
}

template <typename T>
Sphere<T>&
Sphere<T>::operator=(Sphere&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move or copy the sphere parameters from the source object.
    center = std::move(other.center);
    radius = other.radius;

    // Rebind this object's cached operator.
    bind_operator();

    // Rebind the moved-from object's cached operator as well so its
    // internal pointers continue to reference its own storage.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Sphere<T>::bind_operator() noexcept {
    // Bind the cached sphere query operator to this instance's member storage.
    //
    // This keeps delegated geometry queries cheap while ensuring the operator
    // always sees the current center and radius values.
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.radius = atlas::raw_pointer_cast(&radius);
}

template <typename T>
typename Sphere<T>::Builder
Sphere<T>::builder() noexcept {
    // Return a fresh builder object for staged sphere construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Sphere<T>::make_geometry_operator() const {
    // Wrap the cached sphere-specific operator into the generic geometry
    // operator interface expected by the wider geometry system.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-point query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-normal query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Sphere<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the signed-distance query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Sphere<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward inside classification to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Sphere<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward surface-band classification to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::centroid() const noexcept {
    // Forward centroid computation to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Sphere<T>::bound() const noexcept {
    // Forward bounding-box computation to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Sphere<T>::is_valid() const noexcept {
    // Forward validity testing to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Sphere<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete geometry.
    return GeometryType::Sphere;
}

template <typename T>
Sphere<T>
Sphere<T>::Builder::build() const {
    // Validate the staged builder parameters before constructing the sphere.
    validate();

    // Start from a default-constructed sphere so its cached operator is already
    // bound to its own member storage.
    Sphere<T> s {};

    // Overwrite the default sphere parameters with the validated builder state.
    s.center = _center;
    s.radius = _radius;

    // No explicit rebind is required because the cached operator already points
    // to `s`'s member storage and only the values were changed.
    return s;
}

template <typename T>
atlas::host_shared_ptr<Sphere<T>>
Sphere<T>::Builder::make_host_shared() const {
    // Build the sphere by value first.
    auto s = build();

    // Move the built sphere into host-shared managed storage.
    return atlas::make_host_shared<Sphere<T>>(std::move(s));
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_center(const Vector3<T>& c) noexcept {
    // Store the sphere center in the builder's staged state.
    _center = c;
    return *this;
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_radius(T r) noexcept {
    // Store the sphere radius in the builder's staged state.
    _radius = r;
    return *this;
}

template <typename T>
void
Sphere<T>::Builder::validate() const {
    // Reuse the runtime geometry-operator validity logic so the definition
    // of a valid sphere remains centralized in one place.
    atlas::geometry::SphereGeometryOperator<T> op;

    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);

    // Reject invalid staged parameters with both a log message and an exception.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Sphere::Builder validation failed: radius must be > 0; center must be finite.";
        throw std::runtime_error("Sphere::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, return the input point unchanged.
    if (!center || !radius) return p;

    // Compute the vector from the sphere center to the query point.
    const atlas::math::Vector<T, 3> v = p - *center;

    // Squared distance from the center to the query point.
    const T len2 = v.length_squared();

    // Small threshold used to detect the degenerate case where the query point
    // is numerically at the center.
    const T e = std::numeric_limits<T>::epsilon();

    if (len2 <= e) {
        // At the exact center, the radial direction is undefined.
        // Return a deterministic point on the sphere surface along +x.
        return atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y, (*center).z);
    }

    // Normalize the center-to-point direction and scale it by the sphere radius.
    // This yields the closest point on the sphere surface.
    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return (*center) + v * ((*radius) * inv_len);
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, no meaningful normal can be computed.
    if (!center || !radius) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // Vector from the sphere center to the query point.
    const atlas::math::Vector<T, 3> v = p - *center;

    // Squared length of that vector.
    const T len2 = v.length_squared();

    // Threshold for the degenerate case where the direction is undefined.
    const T e = std::numeric_limits<T>::epsilon();

    if (len2 <= e) {
        // At the center, choose a deterministic outward normal along +x.
        return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    }

    // Normalize the radial direction.
    return v * (T(1) / static_cast<T>(std::sqrt(len2)));
}

template <typename T>
T
SphereGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, report an infinite distance so the
    // query can be treated as invalid by the caller.
    if (!center || !radius) return std::numeric_limits<T>::infinity();

    // Standard sphere signed-distance function:
    // - negative inside
    // - zero on the surface
    // - positive outside
    return (p - *center).length() - *radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // If the operator is not fully bound, containment cannot be established.
    if (!center || !radius) return false;

    // Expand the effective radius by the requested tolerance.
    // Positive tolerance enlarges the admissible region.
    const T expanded_radius = *radius + tolerance;

    // If the expanded radius becomes negative, no point can be inside.
    if (expanded_radius < T(0)) return false;

    // Compare squared distances to avoid an unnecessary square root.
    return (p - *center).length_squared() <= expanded_radius * expanded_radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the surface when its absolute signed distance
    // lies within the specified tolerance band around zero.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::centroid() const noexcept {
    // If the center pointer is missing, return the origin as a safe fallback.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // The centroid of a sphere is its center.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SphereGeometryOperator<T>::bound() const noexcept {
    // If the operator is not fully bound, return a default-constructed AABB.
    if (!center || !radius) return atlas::spatial::AxisAlignedBoundingBox<T>();

    // The sphere's AABB extends by exactly `radius` along all three axes.
    const atlas::math::Vector<T, 3> dr(*radius, *radius, *radius);
    return atlas::spatial::AxisAlignedBoundingBox<T>((*center) - dr, (*center) + dr);
}

template <typename T>
bool
SphereGeometryOperator<T>::is_valid() const noexcept {
    // Radius pointer must be present.
    if (!radius) return false;

    // A valid sphere requires a strictly positive radius.
    //
    // Note:
    // - this implementation checks radius only
    // - it does not explicitly verify finiteness of center coordinates here
    return (*radius) > T(0);
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Initialize the return record in the default "no intersection" state.
    HitSurface<T> result {};
    if (!center || !radius) return result;

    // Alias the sphere center and radius for readability.
    const atlas::math::Vector<T, 3>& c = *center;
    const T r                          = *radius;

    // Vector from the sphere center to the ray origin.
    const atlas::math::Vector<T, 3> oc = ray.origin - c;

    // Quadratic coefficients for ray-sphere intersection:
    //   |origin + t * direction - center|^2 = r^2
    const T a  = ray.direction.length_squared();
    const T b  = T(2) * oc.dot(ray.direction);
    const T cc = oc.length_squared() - r * r;

    // Discriminant determines whether a real intersection exists.
    const T disc = b * b - T(4) * a * cc;
    if (disc < T(0)) return result;

    // Compute the two quadratic roots.
    const T sqrt_disc = static_cast<T>(std::sqrt(disc));
    const T inv2a     = T(0.5) / a;
    const T t0        = (-b - sqrt_disc) * inv2a;
    const T t1        = (-b + sqrt_disc) * inv2a;

    // Reject when both roots lie behind the ray origin.
    if (t0 < T(0) && t1 < T(0)) return result;

    // Select the nearest non-negative intersection parameter.
    T t = std::numeric_limits<T>::infinity();
    if (t0 >= T(0)) t = t0;
    if (t1 >= T(0) && t1 < t) t = t1;

    // Populate the hit record.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);

    // Compute the outward normal at the hit point.
    atlas::math::Vector<T, 3> n = result.point - c;
    const T len2                = n.length_squared();

    if (len2 > T(0)) {
        // Normalize the radial direction when it is well-defined.
        n *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {
        // Degenerate fallback if the normal cannot be resolved numerically.
        n = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    }

    result.normal = n;
    return result;
}

template <typename T>
HitSurface<T>
SphereGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Function-call convenience wrapper around the explicit ray-trace routine.
    return trace(ray);
}

} // namespace atlas::geometry