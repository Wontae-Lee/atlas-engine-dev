#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* Triangle<T>                                                             */
/* ====================================================================== */

template <typename T>
Triangle<T>::Triangle(const Vector3<T>& a_,
                      const Vector3<T>& b_,
                      const Vector3<T>& c_) noexcept
    : a(a_)
    , b(b_)
    , c(c_) {
    // Construct a triangle from three vertices.
    //
    // Vertex ordering (winding):
    // - The order (a,b,c) determines the triangle's oriented normal by the
    //   right-hand rule:
    //     normal ∝ (b - a) × (c - a)
    //
    // Cached normal:
    // - We compute and store a normalized normal so that:
    //   1) trace/query operators can use it cheaply,
    //   2) signed distance has a stable sign reference,
    //   3) repeated normal queries do not re-cross/re-normalize.
    //
    // Degenerate case:
    // - If a,b,c are collinear or duplicated, cross() is ~0 and normalized()
    //   should handle it (either returning a safe default or leaving it zero).
    normal = math::cross(b - a, c - a).normalized();
}

template <typename T>
typename Triangle<T>::Builder
Triangle<T>::builder() noexcept {
    // Builder entry-point:
    //   auto tri = Triangle<T>::builder()
    //                 .with_vertices(a,b,c)
    //                 .build();
    return Builder {};
}

/* ---------------------------------------------------------------------- */
/* Operators: Trace / Query                                                */
/* ---------------------------------------------------------------------- */

template <typename T>
TraceOperator<T>
Triangle<T>::make_trace_operator() const {
    // Build a polymorphic TraceOperator<T> that references this triangle's
    // storage via raw pointers.
    //
    // Design:
    // - The operator is a lightweight POD-like object intended to be copied
    //   into kernels / dispatch structures.
    // - It does NOT own the data; it stores pointers to a,b,c,normal.
    //
    // Lifetime requirement:
    // - The Triangle<T> instance must outlive any use of the returned operator.
    atlas::spatial::TriangleTraceOperator<T> op;
    op.a      = atlas::raw_pointer_cast(&a);
    op.b      = atlas::raw_pointer_cast(&b);
    op.c      = atlas::raw_pointer_cast(&c);
    op.normal = atlas::raw_pointer_cast(&normal);

    // Wrap the concrete operator inside the type-erased TraceOperator<T>.
    return TraceOperator<T>(op);
}

template <typename T>
QueryOperator<T>
Triangle<T>::make_query_operator() const {
    // Build a polymorphic QueryOperator<T> that references this triangle's data.
    //
    // Why provide both make_trace_operator() and make_query_operator()?
    // - Trace operators typically answer "ray hit?" style queries.
    // - Query operators answer "closest point/normal/sdf" style queries.
    //
    // Both are stored in a type-erased dispatcher for flexible geometry usage.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return QueryOperator<T>(op);
}

/* ---------------------------------------------------------------------- */
/* Convenience forwarding methods                                          */
/* ---------------------------------------------------------------------- */

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // - Constructs a TriangleQueryOperator wired to this triangle and forwards.
    // - Keeps the public Triangle API small while reusing shared operator logic.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Returns the normal associated with the closest surface feature.
    //
    // In practice for triangles this is often:
    // - the triangle normal itself (if stored),
    // - or a recomputed geometric normal.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.closest_normal(p);
}

template <typename T>
T
Triangle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to an oriented triangle surface.
    //
    // Note:
    // - "signed" uses the triangle normal direction as the sign reference.
    // - For open surfaces, "inside/outside" isn't globally defined, but
    //   per-triangle signed distance is still useful (e.g., local SDF, contact).
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.signed_distance(p);
}

template <typename T>
bool
Triangle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Orientation-dependent triangle "inside" classification is defined in
    // TriangleQueryOperator<T>; forward to it to avoid divergent rules.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Triangle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-band classification to the query operator so the
    // wrapper does not reimplement point-to-triangle distance logic.
    return make_query_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::centroid() const noexcept {
    // Centroid:
    //   (a + b + c) / 3
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Triangle<T>::bound() const noexcept {
    // Bounding box:
    // - component-wise min/max over vertices.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.bound();
}

template <typename T>
bool
Triangle<T>::is_valid() const noexcept {
    // Validity:
    // - vertices exist (they do, since stored by value)
    // - triangle has non-zero area (cross product magnitude > 0)
    //
    // This forwards to TriangleQueryOperator::is_valid() so that the
    // same rule is used consistently across the system.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return op.is_valid();
}

template <typename T>
GeometryType
Triangle<T>::type() const noexcept {
    // Return the geometry type tag for this class.
    return GeometryType::Triangle;
}

/* ---------------------------------------------------------------------- */
/* Mutators                                                                */
/* ---------------------------------------------------------------------- */

template <typename T>
void
Triangle<T>::set_vertices(const Vector3<T>& a_,
                          const Vector3<T>& b_,
                          const Vector3<T>& c_) noexcept {
    // Update vertices and refresh cached normal.
    //
    // Why recompute normal here?
    // - normal is derived data; storing it avoids repeated cross/normalize work.
    // - After changing vertices, the cached normal must be consistent.
    a = a_;
    b = b_;
    c = c_;

    normal = math::cross(b - a, c - a).normalized();
}

/* ---------------------------------------------------------------------- */
/* Barycentric coordinates                                                 */
/* ---------------------------------------------------------------------- */

template <typename T>
bool
Triangle<T>::barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept {
    // Compute barycentric coordinates of point p w.r.t. triangle (a,b,c).
    //
    // We solve for:
    //   p = u*a + v*b + w*c
    // with:
    //   u + v + w = 1
    //
    // A common approach:
    // - Express p in the basis (b-a, c-a):
    //     p - a = v*(b-a) + w*(c-a)
    // - Solve 2x2 system using dot products.
    //
    // Returns:
    // - true  if the triangle is non-degenerate (denom != 0)
    // - false if degenerate; (u,v,w) are set to a safe fallback (1,0,0).
    //
    // Interpretation:
    // - If u,v,w are all in [0,1] (with tolerance), p lies inside the triangle
    //   (including edges).
    const Vector3<T> v0 = b - a;
    const Vector3<T> v1 = c - a;
    const Vector3<T> v2 = p - a;

    const T d00 = v0.dot(v0);
    const T d01 = v0.dot(v1);
    const T d11 = v1.dot(v1);
    const T d20 = v2.dot(v0);
    const T d21 = v2.dot(v1);

    // Denominator of the barycentric solve:
    //   denom = |v0|^2 |v1|^2 - (v0·v1)^2
    // If denom == 0, v0 and v1 are linearly dependent -> triangle degenerate.
    const T denom = d00 * d11 - d01 * d01;
    if (denom == T(0)) {
        // Degenerate: choose vertex 'a' as the only meaningful reference.
        u = T(1);
        v = T(0);
        w = T(0);
        return false;
    }

    const T inv = T(1) / denom;

    // These formulas correspond to solving:
    //   [d00 d01] [v] = [d20]
    //   [d01 d11] [w]   [d21]
    v = (d11 * d20 - d01 * d21) * inv;
    w = (d00 * d21 - d01 * d20) * inv;

    // u is the remainder to enforce u+v+w=1.
    u = T(1) - v - w;

    return true;
}

/* ====================================================================== */
/* Triangle<T>::Builder                                                    */
/* ====================================================================== */

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_a(const Vector3<T>& a_) noexcept {
    // Set vertex A (by value).
    _a = a_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_b(const Vector3<T>& b_) noexcept {
    // Set vertex B (by value).
    _b = b_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_c(const Vector3<T>& c_) noexcept {
    // Set vertex C (by value).
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_vertices(const Vector3<T>& a_,
                                    const Vector3<T>& b_,
                                    const Vector3<T>& c_) noexcept {
    // Convenience setter for all vertices.
    _a = a_;
    _b = b_;
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Optional override:
    // - Allows the user to provide a custom normal (e.g., smoothed normal).
    //
    // Caveat:
    // - This normal should typically be unit-length to keep consistent behavior
    //   for signed_distance and shading-style uses.
    _normal = normal_;
    return *this;
}

template <typename T>
void
Triangle<T>::Builder::validate() const {
    // Validate triangle geometry before building.
    //
    // Implementation approach:
    // - Reuse TriangleQueryOperator<T>::is_valid() so the same validity rule
    //   is applied across both the "Triangle<T>" owner and operator path.
    atlas::geometry::TriangleQueryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&_a);
    op.b = atlas::raw_pointer_cast(&_b);
    op.c = atlas::raw_pointer_cast(&_c);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Triangle::Builder validation failed: vertices must not be collinear or duplicated.";
        throw std::runtime_error("Triangle::Builder: invalid triangle.");
    }
}

template <typename T>
Triangle<T>
Triangle<T>::Builder::build() const {
    // Build a Triangle<T> value object from builder state.
    //
    // Notes:
    // - We validate first to avoid creating degenerate triangles silently.
    // - We also ensure the cached normal is consistent.
    validate();

    Triangle<T> t {};

    // Store vertices.
    t.a = _a;
    t.b = _b;
    t.c = _c;

    // Decide normal:
    // - If a custom normal was provided, use it.
    // - Otherwise compute geometric normal from vertices.
    //
    // This keeps the builder flexible:
    // - geometric normal for physics/contact
    // - custom normal for shading/artist-authored surfaces
    if (_normal.has_value()) {
        t.normal = *_normal;
    } else {
        t.normal = math::cross(t.b - t.a, t.c - t.a).normalized();
    }

    return t;
}

template <typename T>
atlas::host_shared_ptr<Triangle<T>>
Triangle<T>::Builder::make_host_shared() const {
    // Allocate a shared-owned Triangle<T> on the host.
    //
    // Pattern:
    // - build value object
    // - move into shared wrapper to avoid extra copy when possible
    auto t = build();
    return atlas::make_host_shared<Triangle<T>>(std::move(t));
}

} // namespace atlas::geometry
