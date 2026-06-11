#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Triangle<T>::Triangle() noexcept {
    // Bind the operator to this triangle's default vertex and normal storage.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(const Vector3<T>& a_,
                      const Vector3<T>& b_,
                      const Vector3<T>& c_) noexcept
    : a(a_)
    , b(b_)
    , c(c_) {
    // Compute the geometric normal from the vertex winding.
    normal = atlas::math::normalized_or(
        math::cross(b - a, c - a),
        Vector3<T>(T(0), T(0), T(0)));

    // Bind the operator after initializing vertices and normal.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(const Triangle& other) noexcept
    : a(other.a)
    , b(other.b)
    , c(other.c)
    , normal(other.normal) {
    // Rebind the operator because copied raw pointers must refer to this object.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(Triangle&& other) noexcept
    : a(std::move(other.a))
    , b(std::move(other.b))
    , c(std::move(other.c))
    , normal(std::move(other.normal)) {
    // Rebind this object after moving member storage.
    bind_operator();

    // Keep the moved-from object internally consistent.
    other.bind_operator();
}

template <typename T>
Triangle<T>&
Triangle<T>::operator=(const Triangle& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    a      = other.a;
    b      = other.b;
    c      = other.c;
    normal = other.normal;

    // Rebind after assignment because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
Triangle<T>&
Triangle<T>::operator=(Triangle&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    a      = std::move(other.a);
    b      = std::move(other.b);
    c      = std::move(other.c);
    normal = std::move(other.normal);

    // Rebind both objects so each operator points to its own member storage.
    bind_operator();
    other.bind_operator();

    return *this;
}

template <typename T>
void
Triangle<T>::bind_operator() noexcept {
    // Store non-owning raw pointers to the triangle vertices and normal.
    _operator.a      = atlas::raw_pointer_cast(&a);
    _operator.b      = atlas::raw_pointer_cast(&b);
    _operator.c      = atlas::raw_pointer_cast(&c);
    _operator.n      = atlas::raw_pointer_cast(&normal);
    _operator.normal = atlas::raw_pointer_cast(&normal);
}

template <typename T>
typename Triangle<T>::Builder
Triangle<T>::builder() noexcept {
    // Return a fresh builder for fluent triangle construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Triangle<T>::make_geometry_operator() const {
    // Wrap the concrete triangle operator in the generic geometry operator type.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound triangle operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound triangle operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Triangle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound triangle operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Triangle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate containment checks to the bound triangle operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Triangle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-membership checks to the bound triangle operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::centroid() const noexcept {
    // Delegate centroid computation to the bound triangle operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Triangle<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound triangle operator.
    return _operator.bound();
}

template <typename T>
bool
Triangle<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound triangle operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Triangle<T>::type() const noexcept {
    // Identify this geometry as a triangle.
    return GeometryType::Triangle;
}

template <typename T>
void
Triangle<T>::set_vertices(const Vector3<T>& a_,
                          const Vector3<T>& b_,
                          const Vector3<T>& c_) noexcept {
    // Replace all triangle vertices at once.
    a = a_;
    b = b_;
    c = c_;

    // Recompute the normal from the updated vertex winding.
    normal = atlas::math::normalized_or(
        math::cross(b - a, c - a),
        Vector3<T>(T(0), T(0), T(0)));

    // Rebind to keep operator pointers synchronized with this object.
    bind_operator();
}

template <typename T>
bool
Triangle<T>::barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept {
    // Build edge vectors and the point offset from vertex a.
    const Vector3<T> v0 = b - a;
    const Vector3<T> v1 = c - a;
    const Vector3<T> v2 = p - a;

    // Precompute dot products used by the barycentric coordinate solve.
    const T d00   = v0.dot(v0);
    const T d01   = v0.dot(v1);
    const T d11   = v1.dot(v1);
    const T d20   = v2.dot(v0);
    const T d21   = v2.dot(v1);
    const T denom = d00 * d11 - d01 * d01;

    if (denom == T(0)) {
        // Degenerate triangles cannot provide reliable barycentric coordinates.
        u = T(1);
        v = T(0);
        w = T(0);
        return false;
    }

    // Solve barycentric coordinates in the triangle basis.
    const T inv = T(1) / denom;
    v           = (d11 * d20 - d01 * d21) * inv;
    w           = (d00 * d21 - d01 * d20) * inv;
    u           = T(1) - v - w;

    return true;
}

template <typename T>
Triangle<T>
Triangle<T>::Builder::build() const {
    // Validate the triangle vertices before constructing the final object.
    validate();

    Triangle<T> t {};
    t.a = _a;
    t.b = _b;
    t.c = _c;

    if (_normal.has_value()) {
        // Use the explicitly supplied normal when provided.
        t.normal = *_normal;
    } else {
        // Otherwise derive the normal from the triangle vertex winding.
        t.normal = atlas::math::normalized_or(
            math::cross(t.b - t.a, t.c - t.a),
            Vector3<T>(T(0), T(0), T(0)));
    }

    // Rebind because vertices and normal are assigned after default construction.
    t.bind_operator();

    return t;
}

template <typename T>
atlas::host_shared_ptr<Triangle<T>>
Triangle<T>::Builder::make_host_shared() const {
    // Build a validated triangle and store it in host-managed shared ownership.
    auto t = build();
    return atlas::make_host_shared<Triangle<T>>(std::move(t));
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_a(const Vector3<T>& a_) noexcept {
    // Store vertex a for the later build() call.
    _a = a_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_b(const Vector3<T>& b_) noexcept {
    // Store vertex b for the later build() call.
    _b = b_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_c(const Vector3<T>& c_) noexcept {
    // Store vertex c for the later build() call.
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_vertices(const Vector3<T>& a_,
                                    const Vector3<T>& b_,
                                    const Vector3<T>& c_) noexcept {
    // Store all triangle vertices at once.
    _a = a_;
    _b = b_;
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store an optional explicit normal for the later build() call.
    _normal = normal_;
    return *this;
}

template <typename T>
void
Triangle<T>::Builder::validate() const {
    atlas::geometry::TriangleGeometryOperator<T> op;

    // Validate through the same operator logic used by constructed Triangle instances.
    op.a = atlas::raw_pointer_cast(&_a);
    op.b = atlas::raw_pointer_cast(&_b);
    op.c = atlas::raw_pointer_cast(&_c);

    if (!op.is_valid()) {
        throw std::runtime_error("Triangle::Builder: invalid triangle.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without valid vertices, there is no meaningful projection target.
    if (!a || !b || !c) {
        return p;
    }

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    // Triangle edge vectors from v0.
    const atlas::math::Vector<T, 3> ab = v1 - v0;
    const atlas::math::Vector<T, 3> ac = v2 - v0;
    const atlas::math::Vector<T, 3> ap = p - v0;

    // Test whether the closest point is vertex v0.
    const T d1 = ab.dot(ap);
    const T d2 = ac.dot(ap);

    if (d1 <= T(0) && d2 <= T(0)) {
        return v0;
    }

    // Test whether the closest point is vertex v1.
    const atlas::math::Vector<T, 3> bp = p - v1;
    const T d3                         = ab.dot(bp);
    const T d4                         = ac.dot(bp);

    if (d3 >= T(0) && d4 <= d3) {
        return v1;
    }

    // Test whether the closest point lies on edge v0-v1.
    const T vc = d1 * d4 - d3 * d2;

    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        const T vv = d1 / (d1 - d3);
        return v0 + ab * vv;
    }

    // Test whether the closest point is vertex v2.
    const atlas::math::Vector<T, 3> cpv = p - v2;
    const T d5                          = ab.dot(cpv);
    const T d6                          = ac.dot(cpv);

    if (d6 >= T(0) && d5 <= d6) {
        return v2;
    }

    // Test whether the closest point lies on edge v0-v2.
    const T vb = d5 * d2 - d1 * d6;

    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        const T ww = d2 / (d2 - d6);
        return v0 + ac * ww;
    }

    // Test whether the closest point lies on edge v1-v2.
    const T va = d3 * d6 - d5 * d4;

    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        const T ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * ww;
    }

    // The closest point lies inside the triangle face region.
    const T denom = T(1) / (va + vb + vc);
    const T vv    = vb * denom;
    const T ww    = vc * denom;

    return v0 + ab * vv + ac * ww;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Prefer the primary stored normal pointer when available.
    if (normal) {
        return *normal;
    }

    // Fall back to the legacy/alias normal pointer when available.
    if (n) {
        return *n;
    }

    // Without valid vertices, return a deterministic default normal.
    if (!a || !b || !c) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return atlas::math::normalized_or(
        atlas::math::cross((*b) - (*a), (*c) - (*a)),
        atlas::math::Vector<T, 3>(T(0), T(0), T(1)));
}

template <typename T>
T
TriangleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!a || !b || !c) {
        return std::numeric_limits<T>::infinity();
    }

    // Use the triangle normal to determine the side of the supporting plane.
    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T sd_plane                   = (p - (*a)).dot(nn);

    // Distance magnitude is measured to the finite triangle.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d                          = (p - cp).length();

    return (sd_plane >= T(0)) ? d : -d;
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!a || !b || !c) {
        return false;
    }

    // Use the bound normal if available; otherwise compute one from the vertices.
    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> nn                = normal_ptr ? *normal_ptr
                                                             : atlas::math::cross((*b) - (*a), (*c) - (*a));

    const T nn_len2 = nn.length_squared();

    // Degenerate normals cannot define a valid inside half-space.
    if (nn_len2 <= T(0)) {
        return false;
    }

    // Determine which side of the triangle plane the point lies on.
    const T side = (p - (*a)).dot(nn);

    if (side <= T(0)) {
        // Points on the inward side are accepted for non-negative tolerance.
        if (tolerance >= T(0)) {
            return true;
        }
    } else if (tolerance < T(0)) {
        return false;
    }

    // Closest point on the finite triangle is needed only for tolerance-band checks.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d2                         = (p - cp).length_squared();

    return side <= T(0) ? d2 >= tolerance * tolerance
                        : d2 <= tolerance * tolerance;
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Surface membership only needs unsigned distance to the finite triangle.
    if (!a || !b || !c || tolerance < T(0)) {
        return false;
    }

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    return (p - cp).length_squared() <= tolerance * tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::centroid() const noexcept {
    // Invalid geometry falls back to the origin as a neutral centroid.
    if (!a || !b || !c) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The centroid is the arithmetic mean of the three vertices.
    return ((*a) + (*b) + (*c)) * (T(1) / T(3));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!a || !b || !c) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    // Compute component-wise minimum and maximum corners over all vertices.
    const atlas::math::Vector<T, 3> mn = atlas::math::cmin(*a, atlas::math::cmin(*b, *c));
    const atlas::math::Vector<T, 3> mx = atlas::math::cmax(*a, atlas::math::cmax(*b, *c));

    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_valid() const noexcept {
    // All three vertex pointers must be bound before validation can succeed.
    if (!a || !b || !c) {
        return false;
    }

    // A valid triangle must have nonzero area.
    const atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    return nn.length_squared() > T(0);
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    HitSurface<T> result {};

    // Invalid geometry produces a default non-intersecting hit result.
    if (!a || !b || !c) {
        return result;
    }

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    // Edge vectors used by the Moller-Trumbore intersection test.
    const atlas::math::Vector<T, 3> e1 = v1 - v0;
    const atlas::math::Vector<T, 3> e2 = v2 - v0;

    // Compute determinant term from the ray direction and second edge.
    const atlas::math::Vector<T, 3> pvec = atlas::math::cross(r.direction, e2);
    const T det                          = e1.dot(pvec);

    // Near-zero determinant means the ray is parallel to the triangle plane.
    if (static_cast<T>(std::fabs(static_cast<double>(det))) <= T(eps)) {
        return result;
    }

    const T inv_det = T(1) / det;

    // Compute the first barycentric coordinate.
    const atlas::math::Vector<T, 3> tvec = r.origin - v0;
    const T u                            = tvec.dot(pvec) * inv_det;

    if (u < T(0) || u > T(1)) {
        return result;
    }

    // Compute the second barycentric coordinate.
    const atlas::math::Vector<T, 3> qvec = atlas::math::cross(tvec, e1);
    const T v                            = r.direction.dot(qvec) * inv_det;

    if (v < T(0) || (u + v) > T(1)) {
        return result;
    }

    // Compute the ray distance to the triangle plane.
    const T t = e2.dot(qvec) * inv_det;

    // Ignore hits at or behind the ray origin.
    if (t < T(eps)) {
        return result;
    }

    // Populate the hit record with the valid triangle intersection.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);

    // Use the bound normal if present; otherwise compute the geometric normal.
    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> normal_vec        = normal_ptr ? *normal_ptr : atlas::math::cross(e1, e2);

    result.normal = atlas::math::normalized_or(
        normal_vec,
        atlas::math::Vector<T, 3>(T(1), T(0), T(0)));

    return result;
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

} // namespace atlas::geometry
