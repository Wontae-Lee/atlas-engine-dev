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
GeometryOperator<T>
Plane<T>::make_geometry_operator() const {
    // Create a query operator for closest-point/normal/signed-distance queries.
    //
    // Same pointer/lifetime considerations as make_geometry_operator().
    atlas::geometry::PlaneGeometryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the closest point on the plane to p.
    //
    // Delegation pattern:
    // - Create a temporary query operator wired to this plane.
    // - Reuse its implementation for consistency with other code paths.
    atlas::geometry::PlaneGeometryOperator<T> op;
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
    // Exact behavior is defined by PlaneGeometryOperator<T>::closest_normal().
    atlas::geometry::PlaneGeometryOperator<T> op;
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
    // Delegates to PlaneGeometryOperator<T> for consistent behavior.
    atlas::geometry::PlaneGeometryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&normal);
    op.offset = atlas::raw_pointer_cast(&offset);
    return op.signed_distance(p);
}

template <typename T>
bool
Plane<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Plane half-space classification is forwarded to PlaneGeometryOperator<T>
    // to keep the plane equation convention centralized.
    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Plane<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // The query operator owns the tolerance-band interpretation for plane
    // membership, so the wrapper simply forwards the request.
    return make_geometry_operator().is_on_surface(p, tolerance);
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
    // Exact convention is defined by PlaneGeometryOperator<T>::centroid().
    atlas::geometry::PlaneGeometryOperator<T> op;
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
    // The exact behavior is defined in PlaneGeometryOperator<T>::bound().
    atlas::geometry::PlaneGeometryOperator<T> op;
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
    // Exact rules are defined by PlaneGeometryOperator<T>::is_valid().
    atlas::geometry::PlaneGeometryOperator<T> op;
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
    // - Use PlaneGeometryOperator<T>::is_valid() to keep a single source of truth
    //   for "valid plane" rules across the codebase.
    atlas::geometry::PlaneGeometryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.offset = atlas::raw_pointer_cast(&_offset);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Plane::Builder validation failed: normal must be finite and non-zero; offset must be finite.";
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

/* PlaneGeometryOperator<T>                                                   */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on plane to p using projection.
    //
    // Plane implicit form:
    //   n · x + d = 0
    //
    // Signed distance (scaled by |n| if n not unit):
    //   s = n · p + d
    //
    // Projection (assuming n is unit):
    //   cp = p - s * n
    //
    // If n is not unit, the correct projection would be:
    //   cp = p - (s / |n|^2) * n
    //
    // This operator assumes the provided normal is already unit-length
    // (or that the caller accepts the scaled behavior).
    if (!normal || !offset) return p;

    const T sdev = ((*normal).dot(p) + (*offset));
    return p - sdev * (*normal);
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Plane normal is constant everywhere.
    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *normal;
}

template <typename T>
T
PlaneGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance (not necessarily normalized):
    //   sd = n · p + d
    if (!normal || !offset) return std::numeric_limits<T>::infinity();
    return (*normal).dot(p) + (*offset);
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the "inside" side of the plane, allowing tolerance.
    //
    // Plane implicit form:
    //   n · x + d = 0
    //
    // The scalar
    //   s = n · p + d
    // classifies the point relative to the plane:
    //
    // - s < 0 : point is on the negative side of the plane
    // - s = 0 : point lies exactly on the plane
    // - s > 0 : point is on the positive side of the plane
    //
    // This operator defines "inside" as:
    //   s <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     inside includes the plane itself and the entire negative half-space
    //
    // - tolerance > 0:
    //     the accepted half-space is expanded slightly into the positive side,
    //     which is useful for numerical robustness near the plane
    //
    // - tolerance < 0:
    //     the accepted region shrinks and becomes stricter
    //
    // Important note:
    // - This is a half-space test, not a bounded-volume test.
    // - For a plane, "inside" always means one side of the infinite plane.
    //
    // Fallback policy:
    // - If either normal or offset is missing, there is no valid plane equation,
    //   so return false.
    if (!normal || !offset) return false;

    // Evaluate the plane equation at p and compare against tolerance.
    return (*normal).dot(p) + (*offset) <= tolerance;
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the plane surface within the given tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for this operator is:
    //
    //     sd = n · p + d
    //
    //   Note:
    //   - This is the signed plane equation value.
    //   - It is the true geometric signed distance only if the normal is unit length.
    //   - If the normal is not normalized, this is still a consistent signed
    //     classification value, just scaled by |n|.
    //
    // Surface test:
    //   |sd| <= tolerance
    //
    // Meaning:
    // - tolerance = 0:
    //     only points exactly satisfying the plane equation count as on-surface
    //
    // - tolerance > 0:
    //     accept a thin slab around the plane, which helps with floating-point
    //     robustness and near-surface classification
    //
    // Design benefit:
    // - This keeps the surface test fully consistent with the same signed-distance
    //   convention used elsewhere by the operator.
    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::centroid() const noexcept {
    // Infinite plane has no unique centroid.
    // This implementation returns the origin as a stable placeholder.
    return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
PlaneGeometryOperator<T>::bound() const noexcept {
    // Infinite plane has an infinite AABB.
    //
    // Implementation uses numeric extremes as a sentinel "infinite" box.
    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>(lo, lo, lo),
        atlas::math::Vector<T, 3>(hi, hi, hi));
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_valid() const noexcept {
    // Validity rules:
    // - both pointers must be present
    // - normal must be non-zero length
    // - offset must be finite
    if (!normal || !offset) return false;

    const T n2 = (*normal).length_squared();
    return (n2 > T(0)) && std::isfinite(static_cast<double>(*offset));
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};
    if (!normal || !offset) return result;

    const atlas::math::Vector<T, 3>& n = *normal;
    const T d                          = *offset;
    const T denom                      = n.dot(ray.direction);
    const T numer                      = -(n.dot(ray.origin) + d);

    if (denom == T(0)) {
        if (numer != T(0)) return result;
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = n;
        return result;
    }

    const T t = numer / denom;
    if (t < T(0)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

} // namespace atlas::geometry
