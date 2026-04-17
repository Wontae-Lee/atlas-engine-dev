#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept>
#include <utility>
#include <atlas/logging/logging.h>

namespace atlas::geometry {

template <typename T>
Triangle<T>::Triangle() noexcept {
    // Bind the cached geometry operator to this instance's member storage.
    //
    // The default constructor relies on the class's default-initialized vertex
    // and normal members, and the operator must immediately point to those
    // members so all delegated geometric queries stay valid.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(const Vector3<T>& a_,
                      const Vector3<T>& b_,
                      const Vector3<T>& c_) noexcept
    : a(a_)
    , b(b_)
    , c(c_) {
    // Store the three triangle vertices exactly as provided by the caller.
    //
    // Vertex naming convention:
    // - `a` : first vertex
    // - `b` : second vertex
    // - `c` : third vertex

    // Compute the geometric face normal from the oriented triangle edges:
    //     normal = normalize((b - a) x (c - a))
    //
    // This produces a unit-length normal when the triangle is non-degenerate
    // and preserves the winding-dependent orientation.
    normal = math::cross(b - a, c - a).normalized();

    // Bind the cached operator after all geometric members have been initialized.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(const Triangle& other) noexcept
    : a(other.a)
    , b(other.b)
    , c(other.c)
    , normal(other.normal) {
    // Copy the full geometric state from the source triangle.
    //
    // Rebinding is required because the cached operator must reference this
    // object's own member storage, not the source object's storage.
    bind_operator();
}

template <typename T>
Triangle<T>::Triangle(Triangle&& other) noexcept
    : a(std::move(other.a))
    , b(std::move(other.b))
    , c(std::move(other.c))
    , normal(std::move(other.normal)) {
    // Move the triangle's vertex and normal state into this object.

    // Bind this object's cached operator to its newly moved-in members.
    bind_operator();

    // Rebind the moved-from object's cached operator as well so its internal
    // pointers remain self-consistent after the move.
    other.bind_operator();
}

template <typename T>
Triangle<T>&
Triangle<T>::operator=(const Triangle& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy all geometric members from the source triangle.
    a      = other.a;
    b      = other.b;
    c      = other.c;
    normal = other.normal;

    // Rebind the cached operator so it continues to reference this object's members.
    bind_operator();
    return *this;
}

template <typename T>
Triangle<T>&
Triangle<T>::operator=(Triangle&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move the geometric state from the source triangle.
    a      = std::move(other.a);
    b      = std::move(other.b);
    c      = std::move(other.c);
    normal = std::move(other.normal);

    // Rebind this object's cached operator.
    bind_operator();

    // Rebind the moved-from object's cached operator to keep its internal
    // pointers aligned with its own member storage.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Triangle<T>::bind_operator() noexcept {
    // Make the cached operator reference this triangle's actual vertex members.
    _operator.a = atlas::raw_pointer_cast(&a);
    _operator.b = atlas::raw_pointer_cast(&b);
    _operator.c = atlas::raw_pointer_cast(&c);

    // Publish the same stored face normal under both aliases expected by the
    // operator implementation.
    //
    // This preserves compatibility with code paths that read either `n` or `normal`.
    _operator.n      = atlas::raw_pointer_cast(&normal);
    _operator.normal = atlas::raw_pointer_cast(&normal);
}

template <typename T>
typename Triangle<T>::Builder
Triangle<T>::builder() noexcept {
    // Return a fresh builder object for staged triangle construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Triangle<T>::make_geometry_operator() const {
    // Wrap the cached triangle-specific operator in the generic geometry
    // operator interface used by the wider geometry system.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-point query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-normal query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Triangle<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the signed-distance query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Triangle<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward inside classification to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Triangle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward surface classification to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::centroid() const noexcept {
    // Forward centroid computation to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Triangle<T>::bound() const noexcept {
    // Forward bounding-box computation to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Triangle<T>::is_valid() const noexcept {
    // Forward validity testing to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Triangle<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete primitive.
    return GeometryType::Triangle;
}

template <typename T>
void
Triangle<T>::set_vertices(const Vector3<T>& a_,
                          const Vector3<T>& b_,
                          const Vector3<T>& c_) noexcept {
    // Overwrite all triangle vertices with the new input geometry.
    a = a_;
    b = b_;
    c = c_;

    // Recompute the face normal from the updated vertices so the stored normal
    // remains consistent with the current triangle geometry.
    normal = math::cross(b - a, c - a).normalized();

    // Rebind the cached operator in case any code relies on refreshed member pointers.
    bind_operator();
}

template <typename T>
bool
Triangle<T>::barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept {
    // Compute barycentric coordinates of `p` relative to triangle (a, b, c).
    //
    // Coordinate convention:
    //     p = u * a + v * b + w * c
    // with:
    //     u + v + w = 1

    // Build the standard edge basis from vertex `a`.
    const Vector3<T> v0 = b - a;
    const Vector3<T> v1 = c - a;
    const Vector3<T> v2 = p - a;

    // Precompute the dot products needed for the 2x2 barycentric solve.
    const T d00 = v0.dot(v0);
    const T d01 = v0.dot(v1);
    const T d11 = v1.dot(v1);
    const T d20 = v2.dot(v0);
    const T d21 = v2.dot(v1);

    // Determinant of the Gram matrix.
    //
    // This becomes zero when the triangle is degenerate in the barycentric basis.
    const T denom = d00 * d11 - d01 * d01;
    if (denom == T(0)) {
        // Degenerate fallback:
        // - choose the first vertex weight as 1
        // - mark the solve as failed
        u = T(1);
        v = T(0);
        w = T(0);
        return false;
    }

    // Solve barycentric coordinates for vertices b and c first.
    const T inv = T(1) / denom;

    v = (d11 * d20 - d01 * d21) * inv;
    w = (d00 * d21 - d01 * d20) * inv;

    // Recover the remaining coordinate for vertex a.
    u = T(1) - v - w;

    return true;
}

template <typename T>
Triangle<T>
Triangle<T>::Builder::build() const {
    // Validate the staged builder parameters before constructing the triangle.
    validate();

    // Start from a default-constructed triangle so its cached operator is already bound.
    Triangle<T> t {};

    // Overwrite the default vertices with the staged builder values.
    t.a = _a;
    t.b = _b;
    t.c = _c;

    if (_normal.has_value()) {
        // If the caller explicitly supplied a normal, preserve it exactly.
        t.normal = *_normal;
    } else {
        // Otherwise derive the face normal from the vertex winding.
        t.normal = math::cross(t.b - t.a, t.c - t.a).normalized();
    }

    // No explicit rebind is required because the cached operator already points
    // to `t`'s own members and only the stored values changed.
    return t;
}

template <typename T>
atlas::host_shared_ptr<Triangle<T>>
Triangle<T>::Builder::make_host_shared() const {
    // Build the triangle by value first.
    auto t = build();

    // Move the built triangle into host-shared managed storage.
    return atlas::make_host_shared<Triangle<T>>(std::move(t));
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_a(const Vector3<T>& a_) noexcept {
    // Store vertex `a` in the builder's staged state.
    _a = a_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_b(const Vector3<T>& b_) noexcept {
    // Store vertex `b` in the builder's staged state.
    _b = b_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_c(const Vector3<T>& c_) noexcept {
    // Store vertex `c` in the builder's staged state.
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_vertices(const Vector3<T>& a_,
                                    const Vector3<T>& b_,
                                    const Vector3<T>& c_) noexcept {
    // Store all three vertices at once in the builder's staged state.
    _a = a_;
    _b = b_;
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store an explicit face normal override in the builder.
    _normal = normal_;
    return *this;
}

template <typename T>
void
Triangle<T>::Builder::validate() const {
    // Reuse the runtime triangle-operator validity logic so the definition
    // of a valid triangle stays centralized in one place.
    atlas::geometry::TriangleGeometryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&_a);
    op.b = atlas::raw_pointer_cast(&_b);
    op.c = atlas::raw_pointer_cast(&_c);

    // Reject invalid staged geometry with both a log message and an exception.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Triangle::Builder validation failed: vertices must not be collinear or duplicated.";
        throw std::runtime_error("Triangle::Builder: invalid triangle.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, return the query point unchanged.
    if (!a || !b || !c) return p;

    // Alias the triangle vertices for readability.
    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    // Edge vectors from vertex v0.
    const atlas::math::Vector<T, 3> ab = v1 - v0;
    const atlas::math::Vector<T, 3> ac = v2 - v0;
    const atlas::math::Vector<T, 3> ap = p - v0;

    // Test the vertex region around v0.
    const T d1 = ab.dot(ap);
    const T d2 = ac.dot(ap);
    if (d1 <= T(0) && d2 <= T(0)) return v0;

    // Test the vertex region around v1.
    const atlas::math::Vector<T, 3> bp = p - v1;
    const T d3                         = ab.dot(bp);
    const T d4                         = ac.dot(bp);
    if (d3 >= T(0) && d4 <= d3) return v1;

    // Test the edge region of edge (v0, v1).
    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        // Project onto edge (v0, v1) using the corresponding edge parameter.
        const T vv = d1 / (d1 - d3);
        return v0 + ab * vv;
    }

    // Test the vertex region around v2.
    const atlas::math::Vector<T, 3> cpv = p - v2;
    const T d5                          = ab.dot(cpv);
    const T d6                          = ac.dot(cpv);
    if (d6 >= T(0) && d5 <= d6) return v2;

    // Test the edge region of edge (v0, v2).
    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        // Project onto edge (v0, v2).
        const T ww = d2 / (d2 - d6);
        return v0 + ac * ww;
    }

    // Test the edge region of edge (v1, v2).
    const T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        // Project onto edge (v1, v2).
        const T ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * ww;
    }

    // The point projects inside the face region.
    //
    // Compute the barycentric combination of v0, ab, and ac.
    const T denom = T(1) / (va + vb + vc);
    const T vv    = vb * denom;
    const T ww    = vc * denom;
    return v0 + ab * vv + ac * ww;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Prefer an explicitly bound normal pointer when available.
    if (normal) return *normal;

    // Fall back to the legacy alias if it is present.
    if (n) return *n;

    if (!a || !b || !c) {
        // Without vertices, return a safe fallback normal.
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    // Recompute the face normal directly from the triangle geometry.
    atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T len2                 = nn.length_squared();

    if (len2 > T(0)) {
        // Normalize when the geometric normal is well-defined.
        nn *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {
        // Degenerate fallback normal.
        nn = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return nn;
}

template <typename T>
T
TriangleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, report an infinite distance.
    if (!a || !b || !c) return std::numeric_limits<T>::infinity();

    // Use the triangle face normal to determine the oriented side of the query point.
    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T sd_plane                   = (p - (*a)).dot(nn);

    // Measure the Euclidean distance to the closest point on the finite triangle.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d                          = (p - cp).length();

    // Assign sign according to the oriented side of the supporting plane.
    return (sd_plane >= T(0)) ? d : -d;
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // If the operator is not fully bound, containment cannot be established.
    if (!a || !b || !c) return false;

    // Find the closest point on the finite triangle and compute squared distance to it.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d2                         = (p - cp).length_squared();

    // Prefer a stored normal if available; otherwise derive one from the triangle edges.
    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> nn                = normal_ptr ? *normal_ptr : atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T nn_len2                             = nn.length_squared();
    if (nn_len2 <= T(0)) return false;

    // Determine which side of the oriented supporting plane the query lies on.
    const T side = (p - (*a)).dot(nn);

    if (side <= T(0)) {
        // Points on or behind the oriented plane are considered "inside" under
        // this operator's half-space style convention.
        if (tolerance >= T(0)) return true;

        // With negative tolerance, require the point to be at least a certain
        // distance away from the triangle surface.
        return d2 >= (tolerance * tolerance);
    }

    // For points in front of the oriented plane, negative tolerance never admits them.
    if (tolerance < T(0)) return false;

    // Otherwise accept only if the point is within tolerance distance of the triangle.
    return d2 <= (tolerance * tolerance);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the surface when its absolute signed distance
    // lies within the specified tolerance band.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::centroid() const noexcept {
    // If any vertex pointer is missing, return the origin as a safe fallback.
    if (!a || !b || !c) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // The centroid of a triangle is the arithmetic mean of its three vertices.
    return ((*a) + (*b) + (*c)) * (T(1) / T(3));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleGeometryOperator<T>::bound() const noexcept {
    // If the operator is not fully bound, return a default-constructed AABB.
    if (!a || !b || !c) return atlas::spatial::AxisAlignedBoundingBox<T>();

    // Compute component-wise minimum and maximum over the three vertices.
    const atlas::math::Vector<T, 3> mn = atlas::math::cmin(*a, atlas::math::cmin(*b, *c));
    const atlas::math::Vector<T, 3> mx = atlas::math::cmax(*a, atlas::math::cmax(*b, *c));
    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_valid() const noexcept {
    // All three vertex pointers must be present.
    if (!a || !b || !c) return false;

    // A valid triangle must have non-zero area.
    //
    // Equivalently, the cross product of its two edge vectors must be non-zero.
    const atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    return nn.length_squared() > T(0);
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    // Initialize the return record to the default "no hit" state.
    HitSurface<T> result {};
    if (!a || !b || !c) return result;

    // Alias the triangle vertices.
    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    // Triangle edges from v0.
    const atlas::math::Vector<T, 3> e1 = v1 - v0;
    const atlas::math::Vector<T, 3> e2 = v2 - v0;

    // Begin the Moller-Trumbore intersection test.
    const atlas::math::Vector<T, 3> pvec = atlas::math::cross(r.direction, e2);
    const T det                          = e1.dot(pvec);

    // Reject near-parallel rays.
    if (static_cast<T>(std::fabs(static_cast<double>(det))) <= T(eps)) return result;

    const T inv_det                      = T(1) / det;
    const atlas::math::Vector<T, 3> tvec = r.origin - v0;

    // First barycentric coordinate.
    const T u = tvec.dot(pvec) * inv_det;
    if (u < T(0) || u > T(1)) return result;

    const atlas::math::Vector<T, 3> qvec = atlas::math::cross(tvec, e1);

    // Second barycentric coordinate.
    const T v = r.direction.dot(qvec) * inv_det;
    if (v < T(0) || (u + v) > T(1)) return result;

    // Parametric ray distance to the intersection point.
    const T t = e2.dot(qvec) * inv_det;
    if (t < T(eps)) return result;

    // Populate the hit record.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);

    // Use a stored face normal if available; otherwise derive it from the edges.
    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> normal_vec        = normal_ptr ? *normal_ptr : atlas::math::cross(e1, e2);
    const T n2                                  = normal_vec.length_squared();

    if (n2 > T(0)) {
        // Normalize the hit normal when possible.
        normal_vec *= (T(1) / static_cast<T>(std::sqrt(n2)));
    } else {
        // Degenerate fallback normal.
        normal_vec = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    }

    result.normal = normal_vec;
    return result;
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Function-call convenience wrapper around the explicit ray-trace routine.
    return trace(ray);
}

} // namespace atlas::geometry