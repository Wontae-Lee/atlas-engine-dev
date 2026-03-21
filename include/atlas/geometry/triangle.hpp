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
GeometryOperator<T>
Triangle<T>::make_geometry_operator() const {
    // Build a polymorphic GeometryOperator<T> that references this triangle's data.
    //
    // Why provide both make_geometry_operator() and make_geometry_operator()?
    // - Trace operators typically answer "ray hit?" style queries.
    // - Query operators answer "closest point/normal/sdf" style queries.
    //
    // Both are stored in a type-erased dispatcher for flexible geometry usage.
    atlas::geometry::TriangleGeometryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return GeometryOperator<T>(op);
}

/* ---------------------------------------------------------------------- */
/* Convenience forwarding methods                                          */
/* ---------------------------------------------------------------------- */

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Convenience wrapper:
    // - Constructs a TriangleGeometryOperator wired to this triangle and forwards.
    // - Keeps the public Triangle API small while reusing shared operator logic.
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    // TriangleGeometryOperator<T>; forward to it to avoid divergent rules.
    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Triangle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-band classification to the query operator so the
    // wrapper does not reimplement point-to-triangle distance logic.
    return make_geometry_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::centroid() const noexcept {
    // Centroid:
    //   (a + b + c) / 3
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    // This forwards to TriangleGeometryOperator::is_valid() so that the
    // same rule is used consistently across the system.
    atlas::geometry::TriangleGeometryOperator<T> op;
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
    // - Reuse TriangleGeometryOperator<T>::is_valid() so the same validity rule
    //   is applied across both the "Triangle<T>" owner and operator path.
    atlas::geometry::TriangleGeometryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&_a);
    op.b = atlas::raw_pointer_cast(&_b);
    op.c = atlas::raw_pointer_cast(&_c);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Triangle::Builder validation failed: vertices must not be collinear or duplicated.";
        throw std::runtime_error("Triangle::Builder: invalid triangle.");
    }
}

/* TriangleGeometryOperator<T>                                                */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on a triangle to point p.
    //
    // This is the classic region-based test (Christer Ericson, RTCD):
    // - Check vertex regions outside A, B, C
    // - Check edge regions AB, AC, BC
    // - Otherwise inside face region (use barycentric coordinates)
    //
    // Requires triangle vertices a,b,c.
    if (!a || !b || !c) return p;

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    const atlas::math::Vector<T, 3> ab = v1 - v0;
    const atlas::math::Vector<T, 3> ac = v2 - v0;
    const atlas::math::Vector<T, 3> ap = p - v0;

    const T d1 = ab.dot(ap);
    const T d2 = ac.dot(ap);
    if (d1 <= T(0) && d2 <= T(0)) return v0; // Vertex region A

    const atlas::math::Vector<T, 3> bp = p - v1;
    const T d3                         = ab.dot(bp);
    const T d4                         = ac.dot(bp);
    if (d3 >= T(0) && d4 <= d3) return v1; // Vertex region B

    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        // Edge region AB
        const T vv = d1 / (d1 - d3);
        return v0 + ab * vv;
    }

    const atlas::math::Vector<T, 3> cpv = p - v2;
    const T d5                          = ab.dot(cpv);
    const T d6                          = ac.dot(cpv);
    if (d6 >= T(0) && d5 <= d6) return v2; // Vertex region C

    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        // Edge region AC
        const T ww = d2 / (d2 - d6);
        return v0 + ac * ww;
    }

    const T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        // Edge region BC
        const T ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * ww;
    }

    // Inside face region: compute barycentric coordinates.
    const T denom = T(1) / (va + vb + vc);
    const T vv    = vb * denom;
    const T ww    = vc * denom;
    return v0 + ab * vv + ac * ww;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Return triangle normal.
    //
    // Priority:
    // 1) If explicit normal pointer `n` is provided, return it.
    // 2) Otherwise compute geometric normal cross(b-a, c-a) and normalize.
    if (normal) return *normal;
    if (n) return *n;

    if (!a || !b || !c) {
        // Fallback: arbitrary up normal if geometry not set.
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T len2                 = nn.length_squared();

    if (len2 > T(0)) {
        nn *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {
        // Degenerate triangle: choose a fallback normal.
        nn = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return nn;
}

template <typename T>
T
TriangleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to a triangle (not a plane).
    //
    // Approach:
    // - Compute closest point on triangle.
    // - Use plane-side sign from triangle normal:
    //     sd_plane = dot(p - a, n)
    // - Distance magnitude is |p - cp|.
    //
    // This yields a signed distance that is consistent with the triangle's
    // orientation (normal direction), but note:
    // - For a thin open surface, "inside/outside" is not globally defined.
    if (!a || !b || !c) return std::numeric_limits<T>::infinity();

    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T sd_plane                   = (p - (*a)).dot(nn);

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d                          = (p - cp).length();

    return (sd_plane >= T(0)) ? d : -d;
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p is considered "inside" relative to the triangle,
    // allowing a tolerance.
    //
    // Important note:
    // - A triangle is a 2D surface element in 3D space, not a volumetric region.
    // - Therefore, "inside" here does not mean inside a bounded 3D volume.
    // - Instead, this operator uses:
    //   1) the oriented supporting plane of the triangle, and
    //   2) the closest-point distance to the triangle itself.
    //
    // Triangle definition:
    // - Vertices: a, b, c
    // - Optional stored normal: n
    //
    // If no normal is provided, use the geometric triangle normal:
    //
    //   nn = (b - a) × (c - a)
    //
    // This normal defines the oriented supporting plane of the triangle.
    //
    // Degeneracy:
    // - If the triangle normal has zero length, the triangle is degenerate
    //   (zero area), so classification is undefined and this function returns false.
    //
    // Closest-point strategy:
    // - Compute the closest point cp on the triangle to p.
    //
    //     cp = closest_point(p)
    //
    // - Then compute the squared Euclidean distance:
    //
    //     d2 = |p - cp|^2
    //
    // This gives the shortest distance from p to the finite triangle
    // (including its interior, edges, and vertices).
    //
    // Oriented side test:
    // - Compute:
    //
    //     side = (p - a) · nn
    //
    // Interpretation:
    // - side < 0 : p lies on the negative side of the triangle plane
    // - side = 0 : p lies on the supporting plane
    // - side > 0 : p lies on the positive side of the triangle plane
    //
    // Inside convention:
    // - This operator treats the negative side of the oriented triangle plane
    //   as the "inside" side.
    //
    // Cases:
    //
    // 1) side <= 0
    //
    //    The point is on the plane or on the negative side.
    //
    //    - If tolerance >= 0:
    //        accept immediately
    //
    //      This means the entire negative half-space is considered inside,
    //      independent of the finite-triangle distance.
    //
    //    - If tolerance < 0:
    //        require:
    //
    //            d2 >= tolerance^2
    //
    //      Since tolerance^2 is positive, this excludes points that are too
    //      close to the triangle surface while still remaining on the inside side.
    //
    // 2) side > 0
    //
    //    The point is on the positive side of the plane.
    //
    //    - If tolerance < 0:
    //        reject immediately
    //
    //    - If tolerance >= 0:
    //        accept only if the point lies within tolerance distance of the
    //        finite triangle:
    //
    //            d2 <= tolerance^2
    //
    // Tolerance interpretation:
    // - tolerance = 0:
    //     accept the full negative half-space and the triangle surface itself;
    //     on the positive side, only exact surface points are accepted
    //
    // - tolerance > 0:
    //     expand acceptance slightly into the positive side, but only near the
    //     finite triangle surface
    //
    // - tolerance < 0:
    //     shrink acceptance on the inside side by excluding points too close
    //     to the triangle surface, and reject all points on the positive side
    //
    // Important note:
    // - This is not a pure point-to-triangle inclusion test.
    // - It is an oriented half-space classification combined with a finite-triangle
    //   proximity test near the positive side of the surface.
    //
    // Fallback policy:
    // - If any vertex is missing, or if the triangle is degenerate,
    //   return false.
    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d2                         = (p - cp).length_squared();

    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> nn = normal_ptr ? *normal_ptr : atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T nn_len2              = nn.length_squared();
    if (nn_len2 <= T(0)) return false;

    const T side = (p - (*a)).dot(nn);

    if (side <= T(0)) {
        if (tolerance >= T(0)) return true;
        return d2 >= (tolerance * tolerance);
    }

    if (tolerance < T(0)) return false;
    return d2 <= (tolerance * tolerance);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the triangle surface within tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for this operator is expected to provide
    //   a signed-distance-style value consistent with the triangle surface.
    //
    // Surface test:
    //   |signed_distance(p)| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     only points exactly on the triangle surface are accepted
    //
    // - tolerance > 0:
    //     accept a thin neighborhood around the triangle, including regions
    //     near its interior, edges, and vertices
    //
    // Important note:
    // - For full consistency, signed_distance(p) should follow the same surface
    //   convention as closest_point(p) and the finite triangle geometry.
    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::centroid() const noexcept {
    // Centroid of triangle:
    //   (a + b + c) / 3
    if (!a || !b || !c) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return ((*a) + (*b) + (*c)) * (T(1) / T(3));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleGeometryOperator<T>::bound() const noexcept {
    // AABB of triangle = component-wise min/max over vertices.
    if (!a || !b || !c) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> mn = atlas::math::cmin(*a, atlas::math::cmin(*b, *c));
    const atlas::math::Vector<T, 3> mx = atlas::math::cmax(*a, atlas::math::cmax(*b, *c));
    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_valid() const noexcept {
    // Valid if:
    // - all three vertices exist
    // - triangle has non-zero area (cross product magnitude > 0)
    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    return nn.length_squared() > T(0);
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    HitSurface<T> result {};
    if (!a || !b || !c) return result;

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;
    const atlas::math::Vector<T, 3> e1 = v1 - v0;
    const atlas::math::Vector<T, 3> e2 = v2 - v0;
    const atlas::math::Vector<T, 3> pvec = atlas::math::cross(r.direction, e2);
    const T det = e1.dot(pvec);
    if (static_cast<T>(std::fabs(static_cast<double>(det))) <= T(eps)) return result;

    const T inv_det = T(1) / det;
    const atlas::math::Vector<T, 3> tvec = r.origin - v0;
    const T u = tvec.dot(pvec) * inv_det;
    if (u < T(0) || u > T(1)) return result;

    const atlas::math::Vector<T, 3> qvec = atlas::math::cross(tvec, e1);
    const T v = r.direction.dot(qvec) * inv_det;
    if (v < T(0) || (u + v) > T(1)) return result;

    const T t = e2.dot(qvec) * inv_det;
    if (t < T(eps)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);

    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> normal_vec = normal_ptr ? *normal_ptr : atlas::math::cross(e1, e2);
    const T n2 = normal_vec.length_squared();
    if (n2 > T(0)) normal_vec *= (T(1) / static_cast<T>(std::sqrt(n2)));
    else
        normal_vec = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    result.normal = normal_vec;
    return result;
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

} // namespace atlas::geometry
