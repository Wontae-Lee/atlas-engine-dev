#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Triangle<T>::Triangle(const Vector3<T>& a_,
                      const Vector3<T>& b_,
                      const Vector3<T>& c_) noexcept
    : a(a_)
    , b(b_)
    , c(c_) {

    normal = math::cross(b - a, c - a).normalized();
}

template <typename T>
typename Triangle<T>::Builder
Triangle<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Triangle<T>::make_geometry_operator() const {

    atlas::geometry::TriangleGeometryOperator<T> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&normal);

    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

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

    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Triangle<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    return make_geometry_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Triangle<T>::centroid() const noexcept {

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

    return GeometryType::Triangle;
}

template <typename T>
void
Triangle<T>::set_vertices(const Vector3<T>& a_,
                          const Vector3<T>& b_,
                          const Vector3<T>& c_) noexcept {

    a = a_;
    b = b_;
    c = c_;

    normal = math::cross(b - a, c - a).normalized();
}

template <typename T>
bool
Triangle<T>::barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept {

    const Vector3<T> v0 = b - a;
    const Vector3<T> v1 = c - a;
    const Vector3<T> v2 = p - a;

    const T d00 = v0.dot(v0);
    const T d01 = v0.dot(v1);
    const T d11 = v1.dot(v1);
    const T d20 = v2.dot(v0);
    const T d21 = v2.dot(v1);

    const T denom = d00 * d11 - d01 * d01;
    if (denom == T(0)) {

        u = T(1);
        v = T(0);
        w = T(0);
        return false;
    }

    const T inv = T(1) / denom;

    v = (d11 * d20 - d01 * d21) * inv;
    w = (d00 * d21 - d01 * d20) * inv;

    u = T(1) - v - w;

    return true;
}

template <typename T>
Triangle<T>
Triangle<T>::Builder::build() const {

    validate();

    Triangle<T> t {};

    t.a = _a;
    t.b = _b;
    t.c = _c;

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

    auto t = build();
    return atlas::make_host_shared<Triangle<T>>(std::move(t));
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_a(const Vector3<T>& a_) noexcept {

    _a = a_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_b(const Vector3<T>& b_) noexcept {

    _b = b_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_c(const Vector3<T>& c_) noexcept {

    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_vertices(const Vector3<T>& a_,
                                    const Vector3<T>& b_,
                                    const Vector3<T>& c_) noexcept {

    _a = a_;
    _b = b_;
    _c = c_;
    return *this;
}

template <typename T>
typename Triangle<T>::Builder&
Triangle<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {

    _normal = normal_;
    return *this;
}

template <typename T>
void
Triangle<T>::Builder::validate() const {

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

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!a || !b || !c) return p;

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    const atlas::math::Vector<T, 3> ab = v1 - v0;
    const atlas::math::Vector<T, 3> ac = v2 - v0;
    const atlas::math::Vector<T, 3> ap = p - v0;

    const T d1 = ab.dot(ap);
    const T d2 = ac.dot(ap);
    if (d1 <= T(0) && d2 <= T(0)) return v0;

    const atlas::math::Vector<T, 3> bp = p - v1;
    const T d3                         = ab.dot(bp);
    const T d4                         = ac.dot(bp);
    if (d3 >= T(0) && d4 <= d3) return v1;

    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {

        const T vv = d1 / (d1 - d3);
        return v0 + ab * vv;
    }

    const atlas::math::Vector<T, 3> cpv = p - v2;
    const T d5                          = ab.dot(cpv);
    const T d6                          = ac.dot(cpv);
    if (d6 >= T(0) && d5 <= d6) return v2;

    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {

        const T ww = d2 / (d2 - d6);
        return v0 + ac * ww;
    }

    const T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {

        const T ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * ww;
    }

    const T denom = T(1) / (va + vb + vc);
    const T vv    = vb * denom;
    const T ww    = vc * denom;
    return v0 + ab * vv + ac * ww;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {

    if (normal) return *normal;
    if (n) return *n;

    if (!a || !b || !c) {

        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T len2                 = nn.length_squared();

    if (len2 > T(0)) {
        nn *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {

        nn = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return nn;
}

template <typename T>
T
TriangleGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {

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

    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d2                         = (p - cp).length_squared();

    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> nn                = normal_ptr ? *normal_ptr : atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T nn_len2                             = nn.length_squared();
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

    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
TriangleGeometryOperator<T>::centroid() const noexcept {

    if (!a || !b || !c) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return ((*a) + (*b) + (*c)) * (T(1) / T(3));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleGeometryOperator<T>::bound() const noexcept {

    if (!a || !b || !c) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> mn = atlas::math::cmin(*a, atlas::math::cmin(*b, *c));
    const atlas::math::Vector<T, 3> mx = atlas::math::cmax(*a, atlas::math::cmax(*b, *c));
    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleGeometryOperator<T>::is_valid() const noexcept {

    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    return nn.length_squared() > T(0);
}

template <typename T>
HitSurface<T>
TriangleGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    HitSurface<T> result {};
    if (!a || !b || !c) return result;

    const atlas::math::Vector<T, 3> v0   = *a;
    const atlas::math::Vector<T, 3> v1   = *b;
    const atlas::math::Vector<T, 3> v2   = *c;
    const atlas::math::Vector<T, 3> e1   = v1 - v0;
    const atlas::math::Vector<T, 3> e2   = v2 - v0;
    const atlas::math::Vector<T, 3> pvec = atlas::math::cross(r.direction, e2);
    const T det                          = e1.dot(pvec);
    if (static_cast<T>(std::fabs(static_cast<double>(det))) <= T(eps)) return result;

    const T inv_det                      = T(1) / det;
    const atlas::math::Vector<T, 3> tvec = r.origin - v0;
    const T u                            = tvec.dot(pvec) * inv_det;
    if (u < T(0) || u > T(1)) return result;

    const atlas::math::Vector<T, 3> qvec = atlas::math::cross(tvec, e1);
    const T v                            = r.direction.dot(qvec) * inv_det;
    if (v < T(0) || (u + v) > T(1)) return result;

    const T t = e2.dot(qvec) * inv_det;
    if (t < T(eps)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);

    const atlas::math::Vector<T, 3>* normal_ptr = normal ? normal : n;
    atlas::math::Vector<T, 3> normal_vec        = normal_ptr ? *normal_ptr : atlas::math::cross(e1, e2);
    const T n2                                  = normal_vec.length_squared();
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

}