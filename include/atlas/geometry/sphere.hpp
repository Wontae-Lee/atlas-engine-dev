// <atlas/geometry/sphere.hpp>
#pragma once

#include <atlas/memory/raw_pointer_cast.h> // raw_pointer_cast(): convert &T / smart pointers into raw pointer form used by operator PODs
#include <utility>                         // std::move (used when moving a built Sphere into a shared pointer)

// NOTE:
// This header contains inline definitions for atlas::geometry::Sphere<T>.
//
// Key design idea:
// - Sphere<T> stores parameters (center, radius) as normal members.
// - Query/trace functionality is implemented by lightweight operator structs.
// - Sphere<T> constructs those operators by wiring raw pointers to its members.
//   This keeps operators small and avoids copying geometry parameters repeatedly.
//
// Lifetime rule (important):
// - Operators built by make_geometry_operator()/make_geometry_operator() store pointers
//   into this Sphere<T>. They must not outlive the Sphere instance.

namespace atlas::geometry {

template <typename T>
Sphere<T>::Sphere() noexcept
    : center(T(0), T(0), T(0)) // Default center at origin.
    , radius(T(1)) {           // Default radius is 1 (unit sphere).
}

template <typename T>
Sphere<T>::Sphere(const Vector3<T>& center_, T radius_) noexcept
    : center(center_) // Store provided center.
    , radius(radius_) // Store provided radius (expected to be > 0 for a valid sphere).
{ }

template <typename T>
typename Sphere<T>::Builder
Sphere<T>::builder() noexcept {
    // Entry point for Builder-based construction.
    // Builder allows configuration + validation before creating the final Sphere<T>.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Sphere<T>::make_geometry_operator() const {
    // Build a query operator for geometric queries such as:
    // - closest point / normal
    // - signed distance
    // - centroid and bounds
    // - validity checks
    atlas::geometry::SphereGeometryOperator<T> op;

    // Wire pointers to this sphere's parameters (no copies).
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);

    // Wrap into the generic GeometryOperator<T> handle.
    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // Compute the closest point on the sphere surface to point p.
    //
    // We construct a query operator on the fly to reuse the canonical implementation.
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // Compute the outward normal associated with the closest point to p.
    //
    // Typical behavior for a valid sphere:
    // normal = normalize(closest_point(p) - center)
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.closest_normal(p);
}

template <typename T>
T
Sphere<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // Signed distance function (SDF) of the sphere at point p.
    //
    // Typical convention:
    // - < 0 inside the sphere
    // - = 0 on the surface
    // - > 0 outside the sphere
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.signed_distance(p);
}

template <typename T>
bool
Sphere<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Reuse the sphere query operator so host-side classification and raw
    // operator classification remain exactly aligned.
    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Sphere<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate the tolerance-band test to the query operator rather than
    // duplicating sphere boundary logic here.
    return make_geometry_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::centroid() const noexcept {
    // For a sphere, the centroid is its center.
    // Delegated to the query operator to keep all geometric logic in one place.
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Sphere<T>::bound() const noexcept {
    // Compute axis-aligned bounds of the sphere.
    //
    // Typical result:
    // - min = center - (radius, radius, radius)
    // - max = center + (radius, radius, radius)
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.bound();
}

template <typename T>
bool
Sphere<T>::is_valid() const noexcept {
    // Validate sphere parameters.
    //
    // Typical validity rules:
    // - radius must be strictly positive
    // - center components must be finite (no NaN / Inf)
    atlas::geometry::SphereGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.is_valid();
}

template <typename T>
GeometryType
Sphere<T>::type() const noexcept {
    // Runtime type tag for geometry dispatch.
    return GeometryType::Sphere;
}

/* =========================
 * Sphere<T>::Builder
 * ========================= */

template <typename T>
Sphere<T>
Sphere<T>::Builder::build() const {
    // Build a validated Sphere<T>.
    //
    // Steps:
    // 1) validate (throws on failure)
    // 2) default-construct a Sphere<T>
    // 3) assign validated parameters
    validate();

    Sphere<T> s {};
    s.center = _center;
    s.radius = _radius;
    return s;
}

template <typename T>
atlas::host_shared_ptr<Sphere<T>>
Sphere<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - build Sphere<T> by value
    // - move it into a host shared pointer allocation
    //
    // std::move avoids an extra copy when placing the object on the heap.
    auto s = build();
    return atlas::make_host_shared<Sphere<T>>(std::move(s));
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_center(const Vector3<T>& c) noexcept {
    // Set candidate center for the sphere being built.
    _center = c;
    return *this;
}

template <typename T>
typename Sphere<T>::Builder&
Sphere<T>::Builder::with_radius(T r) noexcept {
    // Set candidate radius for the sphere being built.
    _radius = r;
    return *this;
}

template <typename T>
void
Sphere<T>::Builder::validate() const {
    // Validate builder parameters using the same logic as the runtime query operator.
    //
    // Reuse benefits:
    // - Single source of truth for validity checks
    // - Consistent behavior between "built spheres" and "runtime queries"
    atlas::geometry::SphereGeometryOperator<T> op;

    // Wire pointers to builder-owned storage (safe during this call).
    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);

    // If invalid, log and throw to prevent constructing broken geometry.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Sphere::Builder validation failed: radius must be > 0; center must be finite.";
        throw std::runtime_error("Sphere::Builder: invalid parameters.");
    }
}

/* SphereGeometryOperator<T>                                                  */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on sphere surface to p.
    //
    // Sphere definition:
    // - center c
    // - radius r
    //
    // For p != c:
    // - direction v = (p - c)
    // - closest point = c + r * normalize(v)
    //
    // Degenerate case:
    // - If p is extremely close to center, direction is undefined.
    //   We return a point on +X axis of the sphere.
    if (!center || !radius) return p;

    const atlas::math::Vector<T, 3> v = p - *center;
    const T len2                      = v.length_squared();
    const T e                         = std::numeric_limits<T>::epsilon();

    if (len2 <= e) {
        // p is (numerically) at center; choose an arbitrary point on the sphere.
        return atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y, (*center).z);
    }

    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return (*center) + v * ((*radius) * inv_len);
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Outward normal of sphere at the closest point.
    //
    // For p not at center:
    // - normal = normalize(p - c)
    //
    // Degenerate case:
    // - If p ~ center, return +X as arbitrary normal.
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
    // Signed distance to sphere:
    //   |p - c| - r
    //
    // Negative inside, positive outside (common SDF convention).
    if (!center || !radius) return std::numeric_limits<T>::infinity();
    return (p - *center).length() - *radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies inside the sphere, allowing a configurable tolerance.
    //
    // Sphere model:
    // - center : c
    // - radius : r
    //
    // Exact containment test (no tolerance):
    //   |p - c| <= r
    //
    // Tolerance handling:
    // - We interpret tolerance by expanding the radius:
    //     expanded_radius = r + tolerance
    //
    //   Then the test becomes:
    //     |p - c| <= expanded_radius
    //
    // Meaning:
    // - tolerance > 0:
    //     sphere grows outward, so near-outside points can still count as inside
    //
    // - tolerance = 0:
    //     exact inclusive sphere containment
    //
    // - tolerance < 0:
    //     sphere shrinks inward, making the test stricter
    //
    // This is useful for:
    // - robust geometric classification near the boundary
    // - compensating for floating-point error
    // - intentionally shrinking/expanding the accepted region
    //
    // Fallback policy:
    // - If the operator is not bound to a valid center/radius, return false.
    if (!center || !radius) return false;

    // Adjust radius by tolerance.
    const T expanded_radius = *radius + tolerance;

    // If the effective radius becomes negative, the accepted interior is empty.
    // In that case no point can be considered inside.
    if (expanded_radius < T(0)) return false;

    // Use squared-distance comparison to avoid an unnecessary sqrt:
    //
    //   |p - c| <= expanded_radius
    // is equivalent to
    //   |p - c|^2 <= expanded_radius^2
    //
    // This is both cheaper and numerically common for containment tests.
    return (p - *center).length_squared() <= expanded_radius * expanded_radius;
}

template <typename T>
bool
SphereGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the sphere surface within the given tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for a sphere is:
    //
    //     sd = |p - center| - radius
    //
    //   so:
    //   - sd < 0 : inside
    //   - sd = 0 : exactly on the surface
    //   - sd > 0 : outside
    //
    // - A point is treated as "on the surface" if the magnitude of that signed
    //   distance is at most `tolerance`:
    //
    //     |sd| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     exact surface membership
    //
    // - tolerance > 0:
    //     accept a thin shell around the sphere surface, both slightly inside
    //     and slightly outside
    //
    // Design benefit:
    // - This keeps surface classification fully consistent with the signed-distance
    //   convention already used elsewhere in the operator.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
SphereGeometryOperator<T>::centroid() const noexcept {
    // Centroid of sphere is its center.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SphereGeometryOperator<T>::bound() const noexcept {
    // Axis-aligned bounding box of sphere:
    // - min = c - (r,r,r)
    // - max = c + (r,r,r)
    if (!center || !radius) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> dr(*radius, *radius, *radius);
    return atlas::spatial::AxisAlignedBoundingBox<T>((*center) - dr, (*center) + dr);
}

template <typename T>
bool
SphereGeometryOperator<T>::is_valid() const noexcept {
    // Valid if radius pointer exists and radius is non-negative.
    // Note:
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

} // namespace atlas::geometry
