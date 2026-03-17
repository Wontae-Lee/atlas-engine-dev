#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* Plane<T>                                                                */
/* ====================================================================== */

template <typename T>
Plane<T>::Plane() noexcept
    : normal(T(0), T(0), T(1))
    , offset(T(0)) {
    // Default plane is the XY plane with +Z normal:
    //   n = (0,0,1), d = 0
    //
    // With the common implicit-plane convention:
    //   n · x + d = 0
    // this corresponds to:
    //   (0,0,1) · (x,y,z) + 0 = z = 0
    //
    // So the plane passes through the origin and points "up" along +Z.
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& normal_, T offset_) noexcept
    : normal(normal_)
    , offset(offset_) {
    // Construct a plane directly from (normal, offset).
    //
    // Assumed implicit form:
    //   n · x + d = 0
    // where:
    //   n = normal_
    //   d = offset_
    //
    // Important:
    // - This constructor does NOT normalize the normal.
    // - Many geometric formulas assume |n| = 1. If your operators assume unit normals,
    //   callers should normalize normal_ before passing it, or use builder validation rules.
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept
    : normal(normal_)
    , offset(-(normal_.dot(point))) {
    // Construct a plane from a point on the plane and a normal.
    //
    // Derivation (implicit form):
    //   We want all x on plane to satisfy:
    //     n · (x - p) = 0
    //   Expand:
    //     n · x - n · p = 0
    //   So:
    //     n · x + d = 0  with  d = -(n · p)
    //
    // Again, the normal is stored as-is (not normalized).
}

template <typename T>
typename Plane<T>::Builder
Plane<T>::builder() noexcept {
    // Builder entry point.
    //
    // Builder is useful because:
    // - It can validate that normal is non-zero / finite.
    // - It can centralize any policy about normalization, if you add it later.
    return Builder {};
}

template <typename T>
TraceOperator<T>
Plane<T>::make_trace_operator() const {
    // Create a trace operator for ray-plane intersection.
    //
    // Implementation:
    // - Build a small POD-like PlaneTraceOperator<T>.
    // - Provide it raw pointers to this Plane<T>'s stored parameters.
    //
    // Lifetime note:
    // - The returned operator holds pointers to *this->normal and *this->offset.
    // - Therefore, the Plane<T> instance must remain alive while the operator is used.
    atlas::spatial::PlaneTraceOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return TraceOperator<T>(op);
}

template <typename T>
QueryOperator<T>
Plane<T>::make_query_operator() const {
    // Create a query operator for closest-point/normal/signed-distance queries.
    //
    // Same pointer/lifetime considerations as make_trace_operator().
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return QueryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the closest point on the plane to p.
    //
    // Delegation pattern:
    // - Create a temporary query operator wired to this plane.
    // - Reuse its implementation for consistency with other code paths.
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Return the plane normal (typically normalized by convention).
    //
    // Some implementations may return:
    // - a normalized version of stored normal, or
    // - the stored normal as-is.
    //
    // Exact behavior is defined by PlaneQueryOperator<T>::closest_normal().
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.closest_normal(p);
}

template <typename T>
T
Plane<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance from point p to the plane.
    //
    // Common convention:
    // - positive if p lies in the half-space pointed to by +normal
    // - negative in the opposite half-space
    //
    // If normal is not unit-length:
    // - "distance" may be scaled by |normal|.
    //
    // Delegates to PlaneQueryOperator<T> for consistent behavior.
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.signed_distance(p);
}

template <typename T>
bool
Plane<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Plane half-space classification is forwarded to PlaneQueryOperator<T>
    // to keep the plane equation convention centralized.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Plane<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // The query operator owns the tolerance-band interpretation for plane
    // membership, so the wrapper simply forwards the request.
    return make_query_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::centroid() const noexcept {
    // "Centroid" of an infinite plane is not uniquely defined.
    //
    // Many libraries define centroid() for infinite primitives as:
    // - the origin projected onto the plane, or
    // - some representative point on the plane (often along the normal).
    //
    // Exact convention is defined by PlaneQueryOperator<T>::centroid().
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Plane<T>::bound() const noexcept {
    // Axis-aligned bounding box of an infinite plane.
    //
    // Since the plane is unbounded, many implementations return:
    // - an "infinite" AABB, or
    // - a very large sentinel box.
    //
    // The exact behavior is defined in PlaneQueryOperator<T>::bound().
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.bound();
}

template <typename T>
bool
Plane<T>::is_valid() const noexcept {
    // Validate plane parameters.
    //
    // Typical validity rules:
    // - normal is finite and non-zero length
    // - offset is finite
    //
    // Exact rules are defined by PlaneQueryOperator<T>::is_valid().
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.is_valid();
}

template <typename T>
GeometryType
Plane<T>::type() const noexcept {
    // Return the geometry type tag for this plane.
    return GeometryType::Plane;
}

/* ====================================================================== */
/* Plane<T>::Builder                                                       */
/* ====================================================================== */

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Set plane normal.
    //
    // Note:
    // - Stored as-is; normalization policy (if any) is enforced in validate()
    //   via the query operator's is_valid().
    _normal = normal_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_offset(T offset_) noexcept {
    // Set plane offset "d" in implicit form n·x + d = 0.
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept {
    // Set both normal and offset in one call.
    _normal = normal_;
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_point_normal(const Vector3<T>& point,
                                     const Vector3<T>& normal_) noexcept {
    // Configure plane from a point and normal.
    //
    // Uses:
    //   d = -(n · p)
    _normal = normal_;
    _offset = -(normal_.dot(point));
    return *this;
}

template <typename T>
void
Plane<T>::Builder::validate() const {
    // Validate builder parameters before producing Plane<T>.
    //
    // Delegation:
    // - Use PlaneQueryOperator<T>::is_valid() to keep a single source of truth
    //   for "valid plane" rules across the codebase.
    atlas::geometry::PlaneQueryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.offset = atlas::raw_pointer_cast(&_offset);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Plane::Builder validation failed: normal must be finite and non-zero; offset must be finite.";
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

template <typename T>
Plane<T>
Plane<T>::Builder::build() const {
    // Build a Plane<T> after validation.
    //
    // Strong exception guarantee:
    // - If validate() throws, no Plane is returned.
    validate();

    Plane<T> p {};

    // Copy validated configuration into the final plane instance.
    p.normal = _normal;
    p.offset = _offset;

    return p;
}

template <typename T>
atlas::host_shared_ptr<Plane<T>>
Plane<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build by value
    // - Move into a shared, heap-allocated plane
    auto p = build();
    return atlas::make_host_shared<Plane<T>>(std::move(p));
}

} // namespace atlas::geometry
