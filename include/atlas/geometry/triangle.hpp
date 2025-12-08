#pragma once
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::geometry {
template <typename T>
ATLAS_DEVICE HitSurface<T>

TriangleTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> result;
    const Vector3<T> ab = *b - *a;
    const Vector3<T> ac = *c - *a;
    const Matrix3x3<T> M {
        -r.direction.x,
        ab.x,
        ac.x,
        -r.direction.y,
        ab.y,
        ac.y,
        -r.direction.z,
        ab.z,
        ac.z
    };
    const Vector3<T> rhs = r.origin - *a;
    const Vector3<T> tuv = M.solved(rhs);
    const T t            = tuv[0];
    const T u            = tuv[1];
    const T v            = tuv[2];
    if (t < T(eps) || u < T(0) || v < T(0) || (u + v) > T(1)) return result;
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    Vector3<T> n           = cross(ab, ac).normalized();
    result.normal          = n;
    return result;
}

template <typename T>
Triangle<T>::Triangle(const Vector3<T>& a_,
                      const Vector3<T>& b_,
                      const Vector3<T>& c_) noexcept
    : a(a_)
    , b(b_)
    , c(c_) {
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    normal              = math::normalize(math::cross(ab, ac));
}

template <typename T>
T
Triangle<T>::signed_distance(const Vector3<T>& p) const {
    T sd_plane          = math::dot(p - a, normal);
    const Vector3<T> cp = closest_point(p);
    const T d           = math::length(p - cp);
    T sd                = (sd_plane >= T(0)) ? d : -d;
    return sd;
}

template <typename T>
Vector3<T>
Triangle<T>::closest_point(const Vector3<T>& p) const {
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const Vector3<T> ap = p - a;
    const T d1          = math::dot(ab, ap);
    const T d2          = math::dot(ac, ap);
    if (d1 <= T(0) && d2 <= T(0)) return a;
    const Vector3<T> bp = p - b;
    const Vector3<T> bc = c - b;
    const T d3          = math::dot(bp, bc);
    const T d4          = math::dot(bc, bc);
    if (d3 >= -T(0) && d4 <= d3 + T(0)) return b;
    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= -T(0) && d3 <= T(0)) {
        const T v = d1 / (d1 - d3);
        return a + ab * v;
    }
    const Vector3<T> cp = p - c;
    const T d5          = math::dot(ab, cp);
    const T d6          = math::dot(ac, cp);
    if (d6 >= -T(0) && d5 <= d6 + T(0)) return c;
    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= -T(0) && d6 <= T(0)) {
        const T w = d2 / (d2 - d6);
        return a + ac * w;
    }
    const T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= -T(0) && (d5 - d6) >= -T(0)) {
        const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    const Matrix2x2<T> G {
        math::dot(ab, ab),
        math::dot(ab, ac),
        math::dot(ab, ac),
        math::dot(ac, ac)
    };
    const Vector2<T> r { math::dot(ab, ap), math::dot(ac, ap) };
    const Vector2<T> vw = G.solved(r);
    const T v           = vw[0];
    const T w           = vw[1];
    return a + ab * v + ac * w;
}

template <typename T>
Vector3<T>
Triangle<T>::closest_normal(const Vector3<T>&) const {
    return normal;
}

template <typename T>
T
Triangle<T>::closest_distance(const Vector3<T>& p) const {
    T sd = signed_distance(p);
    return sd >= T(0) ? sd : -sd;
}

template <typename T>
AABB<T>
Triangle<T>::bound() const {
    Vector3<T> min = math::cmin(a, math::cmin(b, c));
    Vector3<T> max = math::cmax(a, math::cmax(b, c));
    return AABB<T>(min, max);
}

template <typename T>
bool
Triangle<T>::intersects(const Ray<T>& ray) const {
    const Vector3<T> ab  = b - a;
    const Vector3<T> ac  = c - a;
    const Vector3<T> rhs = ray.origin - a;
    const Matrix3x3<T> M {
        -ray.direction.x,
        ab.x,
        ac.x,
        -ray.direction.y,
        ab.y,
        ac.y,
        -ray.direction.z,
        ab.z,
        ac.z
    };
    const Vector3<T> tuv = M.solved(rhs);
    const T t            = tuv[0];
    const T u            = tuv[1];
    const T v            = tuv[2];
    if (t < T(eps)) return false;
    if (u < T(eps)) return false;
    if (v < T(eps)) return false;
    if (u + v > T(1)) return false;
    return true;
}

template <typename T>
TriangleTraceOperator<T>
Triangle<T>::make_trace_operator() const {
    TriangleTraceOperator<T> op;
    op.a      = atlas::raw_pointer_cast(&a);
    op.b      = atlas::raw_pointer_cast(&b);
    op.c      = atlas::raw_pointer_cast(&c);
    op.normal = atlas::raw_pointer_cast(&normal);
    return op;
}

template <typename T>
bool
Triangle<T>::is_inside(const Vector3<T>& p) const {
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const Vector3<T> ap = p - a;
    const T d           = dot(normal, ap);
    if (std::abs(d) > T(eps)) return false;
    const Matrix2x2<T> G {
        dot(ab, ab),
        dot(ab, ac),
        dot(ab, ac),
        dot(ac, ac)
    };
    const Vector2<T> r { dot(ab, ap), dot(ac, ap) };
    const Vector2<T> uv = G.solved(r);
    const T u           = uv[0];
    const T v           = uv[1];
    const T w           = T(1) - u - v;
    return (u >= -T(eps)) & (v >= -T(eps)) & (w >= -T(eps)) & (u <= T(1) + T(eps)) & (v <= T(1) + T(eps)) & (w <= T(1) + T(eps));
}

template <typename T>
void
Triangle<T>::set_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept {
    a                   = a_;
    b                   = b_;
    c                   = c_;
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    normal              = math::normalize(math::cross(ab, ac));
}

template <typename T>
Vector3<T>
Triangle<T>::extents() const noexcept {
    Vector3<T> min = math::cmin(a, math::cmin(b, c));
    Vector3<T> max = math::cmax(a, math::cmax(b, c));
    return max - min;
}

template <typename T>
bool
Triangle<T>::is_valid() const noexcept {
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const T area        = math::cross(ab, ac).length();
    return area > T(0);
}

template <typename T>
bool
Triangle<T>::barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept {
    const Vector3<T> v0 = b - a;
    const Vector3<T> v1 = c - a;
    const Vector3<T> v2 = p - a;
    const T d00         = v0.dot(v0);
    const T d01         = v0.dot(v1);
    const T d11         = v1.dot(v1);
    const T d20         = v2.dot(v0);
    const T d21         = v2.dot(v1);
    const T denom       = d00 * d11 - d01 * d01;
    if (denom == T(0)) {
        u = T(1);
        v = T(0);
        w = T(0);
        return false;
    }
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = T(1) - v - w;
    return true;
}
}