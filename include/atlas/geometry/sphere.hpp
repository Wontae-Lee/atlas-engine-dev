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
// - Operators built by make_trace_operator()/make_query_operator() store pointers
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
TraceOperator<T>
Sphere<T>::make_trace_operator() const {
    // Build a trace operator for intersection / tracing queries.
    //
    // Implementation detail:
    // - The operator stores raw pointers to sphere parameters instead of copying values.
    // - This makes the operator cheap to copy and suitable for passing around (e.g., kernels),
    //   but it couples the operator lifetime to the Sphere<T> lifetime.
    atlas::spatial::SphereTraceOperator<T> op;

    // Wire pointer to this sphere's center and radius.
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);

    // Wrap the concrete operator into the generic TraceOperator<T> handle.
    return TraceOperator<T>(op);
}

template <typename T>
QueryOperator<T>
Sphere<T>::make_query_operator() const {
    // Build a query operator for geometric queries such as:
    // - closest point / normal
    // - signed distance
    // - centroid and bounds
    // - validity checks
    atlas::geometry::SphereQueryOperator<T> op;

    // Wire pointers to this sphere's parameters (no copies).
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);

    // Wrap into the generic QueryOperator<T> handle.
    return QueryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // Compute the closest point on the sphere surface to point p.
    //
    // We construct a query operator on the fly to reuse the canonical implementation.
    atlas::geometry::SphereQueryOperator<T> op;
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
    atlas::geometry::SphereQueryOperator<T> op;
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
    atlas::geometry::SphereQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    return op.signed_distance(p);
}

template <typename T>
bool
Sphere<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Reuse the sphere query operator so host-side classification and raw
    // operator classification remain exactly aligned.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Sphere<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate the tolerance-band test to the query operator rather than
    // duplicating sphere boundary logic here.
    return make_query_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Sphere<T>::centroid() const noexcept {
    // For a sphere, the centroid is its center.
    // Delegated to the query operator to keep all geometric logic in one place.
    atlas::geometry::SphereQueryOperator<T> op;
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
    atlas::geometry::SphereQueryOperator<T> op;
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
    atlas::geometry::SphereQueryOperator<T> op;
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
    atlas::geometry::SphereQueryOperator<T> op;

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

} // namespace atlas::geometry
