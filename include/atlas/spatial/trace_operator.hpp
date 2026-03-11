#pragma once

#include <cmath>
#include <limits>

namespace atlas::spatial {

// -------- BoxTraceOperator --------

template <typename T>
HitSurface<T>
BoxTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> result;
    if (!lower_corner || !upper_corner) return result;

    const Vector3<T>& lo = *lower_corner;
    const Vector3<T>& hi = *upper_corner;

    const Vector3<T> inv_dir = T(1) / r.direction;

    const Vector3<T> t0 = (lo - r.origin) * inv_dir;
    const Vector3<T> t1 = (hi - r.origin) * inv_dir;

    const Vector3<T> tmin_v = math::cmin(t0, t1);
    const Vector3<T> tmax_v = math::cmax(t0, t1);

    const T t_enter = tmin_v.max();
    const T t_exit  = tmax_v.min();

    if (t_exit < t_enter) return result;
    if (t_exit < T(eps)) return result;

    const bool use_enter = (t_enter >= T(eps));
    const T t            = use_enter ? t_enter : t_exit;

    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    Vector3<T> n(T(0));
    const T dir = r.direction[axis];

    if (use_enter) n[axis] = (dir >= T(0)) ? T(-1) : T(1);
    else
        n[axis] = (dir >= T(0)) ? T(1) : T(-1);

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    result.normal          = n;
    return result;
}

// -------- CylinderTraceOperator --------

template <typename T>
HitSurface<T>
CylinderTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> out;
    if (!center || !radius || !height) return out;

    const Vector3<T> ro = ray.origin - *center;
    const Vector3<T> rd = ray.direction;

    const T r    = *radius;
    const T hz   = (*height) * T(0.5);
    const T zmin = -hz;
    const T zmax = hz;

    T best_t = std::numeric_limits<T>::infinity();
    Vector3<T> best_n(T(0), T(0), T(0));

    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();

    if (rd.z == T(0)) {
        if (ro.z < zmin || ro.z > zmax) z_ok = false;
    } else {
        const T inv_dz = T(1) / rd.z;
        T a            = (zmin - ro.z) * inv_dz;
        T b            = (zmax - ro.z) * inv_dz;
        if (a > b) {
            const T tmp = a;
            a           = b;
            b           = tmp;
        }
        t_z_enter = a;
        t_z_exit  = b;
    }

    if (z_ok) {
        const T A = rd.x * rd.x + rd.y * rd.y;
        const T B = T(2) * (ro.x * rd.x + ro.y * rd.y);
        const T C = ro.x * ro.x + ro.y * ro.y - r * r;

        if (A > T(0)) {
            const T disc = B * B - T(4) * A * C;
            if (disc >= T(0)) {
                const T sqrt_disc = static_cast<T>(std::sqrt(disc));

                const T sign_b = (B >= T(0)) ? T(1) : T(-1);
                const T q      = -T(0.5) * (B + sign_b * sqrt_disc);

                T t0 = (q == T(0)) ? std::numeric_limits<T>::infinity() : (C / q);
                T t1 = (A == T(0)) ? std::numeric_limits<T>::infinity() : (q / A);
                if (t0 > t1) {
                    const T tmp = t0;
                    t0          = t1;
                    t1          = tmp;
                }

                auto accept_side = [&](T t) -> bool {
                    if (!(t >= T(0))) return false;
                    if (rd.z != T(0)) {
                        if (t < t_z_enter || t > t_z_exit) return false;
                    }
                    return true;
                };

                auto set_side_hit = [&](T t) {
                    best_t              = t;
                    const Vector3<T> ph = ro + rd * t;

                    const T rr2 = ph.x * ph.x + ph.y * ph.y;
                    if (rr2 > T(0)) {
                        const T inv_rr = T(1) / static_cast<T>(std::sqrt(rr2));
                        best_n         = Vector3<T>(ph.x * inv_rr, ph.y * inv_rr, T(0));
                    } else {
                        best_n = Vector3<T>(T(1), T(0), T(0));
                    }
                };

                if (accept_side(t0)) set_side_hit(t0);
                if (!std::isfinite(best_t) && accept_side(t1)) set_side_hit(t1);
            }
        }
    }

    if (rd.z != T(0)) {
        auto try_cap = [&](T zplane, T nz) {
            const T t = (zplane - ro.z) / rd.z;
            if (!(t >= T(0))) return;
            if (t >= best_t) return;

            const Vector3<T> ph = ro + rd * t;
            if (ph.x * ph.x + ph.y * ph.y <= r * r) {
                best_t = t;
                best_n = Vector3<T>(T(0), T(0), nz);
            }
        };

        const T t_min = (zmin - ro.z) / rd.z;
        const T t_max = (zmax - ro.z) / rd.z;
        if (t_min < t_max) {
            try_cap(zmin, -T(1));
            try_cap(zmax, T(1));
        } else {
            try_cap(zmax, T(1));
            try_cap(zmin, -T(1));
        }
    }

    if (!std::isfinite(best_t)) return out;

    out.is_intersecting = true;
    out.distance        = best_t;
    out.point           = ray.point_at(best_t);
    out.normal          = best_n;
    return out;
}

// -------- PlaneTraceOperator --------

template <typename T>
HitSurface<T>
PlaneTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> result;
    if (!normal || !offset) return result;

    const Vector3<T>& n = *normal;
    const T d           = *offset;

    const T denom = n.dot(ray.direction);
    const T numer = -(n.dot(ray.origin) + d);

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

// -------- SphereTraceOperator --------

template <typename T>
HitSurface<T>
SphereTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> result;
    if (!center || !radius) return result;

    const Vector3<T>& c = *center;
    const T r           = *radius;

    const Vector3<T> oc = ray.origin - c;

    const T a    = ray.direction.length_squared();
    const T b    = T(2) * oc.dot(ray.direction);
    const T cc   = oc.length_squared() - r * r;
    const T disc = b * b - T(4) * a * cc;

    if (disc < T(0)) return result;

    const T sqrt_disc = static_cast<T>(std::sqrt(disc));
    const T inv2a     = T(0.5) / a;

    const T t0 = (-b - sqrt_disc) * inv2a;
    const T t1 = (-b + sqrt_disc) * inv2a;

    if (t0 < T(0) && t1 < T(0)) return result;

    T t = std::numeric_limits<T>::infinity();
    if (t0 >= T(0)) t = t0;
    if (t1 >= T(0) && t1 < t) t = t1;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);

    Vector3<T> n = result.point - c;
    const T len2 = n.length_squared();
    if (len2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(len2)));
    else
        n = Vector3<T>(T(1), T(0), T(0));
    result.normal = n;

    return result;
}

// -------- TriangleTraceOperator --------

template <typename T>
HitSurface<T>
TriangleTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> result;
    if (!a || !b || !c) return result;

    const Vector3<T> v0 = *a;
    const Vector3<T> v1 = *b;
    const Vector3<T> v2 = *c;

    const Vector3<T> e1 = v1 - v0;
    const Vector3<T> e2 = v2 - v0;

    const Vector3<T> pvec = math::cross(r.direction, e2);
    const T det           = e1.dot(pvec);

    if (static_cast<T>(std::fabs(static_cast<double>(det))) <= T(eps)) return result;

    const T inv_det = T(1) / det;

    const Vector3<T> tvec = r.origin - v0;
    const T u             = tvec.dot(pvec) * inv_det;
    if (u < T(0) || u > T(1)) return result;

    const Vector3<T> qvec = math::cross(tvec, e1);
    const T v             = r.direction.dot(qvec) * inv_det;
    if (v < T(0) || (u + v) > T(1)) return result;

    const T t = e2.dot(qvec) * inv_det;
    if (t < T(eps)) return result;

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);

    Vector3<T> n = math::cross(e1, e2);
    const T n2   = n.length_squared();
    if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
    else
        n = Vector3<T>(T(1), T(0), T(0));

    result.normal = n;
    return result;
}

// -------- BvhTraceOperator --------

template <typename T>
HitSurface<T>
BvhTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> out {};
    if (!nodes || !indices || !tris) return out;
    if (root < 0) return out;

    T best_t = std::numeric_limits<T>::max();

    Vector3<T> best_p {};
    Vector3<T> best_n {};
    bool found = false;

    int stack[64];
    int sp      = 0;
    stack[sp++] = root;

    while (sp) {
        const int ni         = stack[--sp];
        const BVHNode<T>& nd = nodes[ni];

        HitAABB hit = nd.bounds.trace(r);
        if (!hit.is_intersecting || hit.enter > best_t) continue;

        if (nd.is_leaf) {
            TriangleTraceOperator<T> tri_op;

            ATLAS_UNROLL
            for (int k = 0; k < nd.count; ++k) {
                const int pid                    = indices[nd.start + k];
                const TriangleContainer4<T>& tri = tris[pid];

                tri_op.a      = &tri.a();
                tri_op.b      = &tri.b();
                tri_op.c      = &tri.c();
                tri_op.normal = &tri.d();

                HitSurface<T> h = tri_op(r);

                if (h.is_intersecting && h.distance < best_t) {
                    best_t = h.distance;
                    best_p = h.point;
                    best_n = h.normal;
                    found  = true;
                }
            }
        } else {
            if (sp < 63) stack[sp++] = nd.left;
            else
                stack[63] = nd.left;

            if (sp < 63) stack[sp++] = nd.right;
            else
                stack[63] = nd.right;
        }
    }

    if (found) {
        out.is_intersecting = true;
        out.distance        = best_t;
        out.point           = best_p;
        out.normal          = best_n;
    }
    return out;
}

/* ====================================================================== */
/* TraceOperator special members                                           */
/* ====================================================================== */

template <typename T>
TraceOperator<T>::TraceOperator() noexcept
    : type(atlas::geometry::GeometryType::Sphere) {
    // Construct a safe default active member.
    new (&sphere) SphereTraceOperator<T> {};
}

template <typename T>
TraceOperator<T>::TraceOperator(const TraceOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
TraceOperator<T>&
TraceOperator<T>::operator=(const TraceOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
TraceOperator<T>::~TraceOperator() noexcept {
    destroy_active();
}

template <typename T>
void
TraceOperator<T>::destroy_active() noexcept {
    // If all leaf operators remain trivially destructible, this is effectively a no-op,
    // but keeping it explicit prevents toolchain-dependent "deleted" special members.
    switch (type) {
    case atlas::geometry::GeometryType::Sphere:
        sphere.~SphereTraceOperator<T>();
        return;
    case atlas::geometry::GeometryType::Cylinder:
        cylinder.~CylinderTraceOperator<T>();
        return;
    case atlas::geometry::GeometryType::Plane:
        plane.~PlaneTraceOperator<T>();
        return;
    case atlas::geometry::GeometryType::Box:
        box.~BoxTraceOperator<T>();
        return;
    case atlas::geometry::GeometryType::Triangle:
        triangle.~TriangleTraceOperator<T>();
        return;
    case atlas::geometry::GeometryType::TriangleMesh:
        triangle_mesh.~BvhTraceOperator<T>();
        return;
    default:
        sphere.~SphereTraceOperator<T>();
        return;
    }
}

template <typename T>
void
TraceOperator<T>::copy_from(const TraceOperator& other) noexcept {
    switch (type) {
    case atlas::geometry::GeometryType::Sphere:
        new (&sphere) SphereTraceOperator<T>(other.sphere);
        return;
    case atlas::geometry::GeometryType::Cylinder:
        new (&cylinder) CylinderTraceOperator<T>(other.cylinder);
        return;
    case atlas::geometry::GeometryType::Plane:
        new (&plane) PlaneTraceOperator<T>(other.plane);
        return;
    case atlas::geometry::GeometryType::Box:
        new (&box) BoxTraceOperator<T>(other.box);
        return;
    case atlas::geometry::GeometryType::Triangle:
        new (&triangle) TriangleTraceOperator<T>(other.triangle);
        return;
    case atlas::geometry::GeometryType::TriangleMesh:
        new (&triangle_mesh) BvhTraceOperator<T>(other.triangle_mesh);
        return;
    default:
        type = atlas::geometry::GeometryType::Sphere;
        new (&sphere) SphereTraceOperator<T>(other.sphere);
        return;
    }
}

/* ====================================================================== */
/* TraceOperator tagged constructors                                       */
/* ====================================================================== */

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const SphereTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::Sphere) {
    new (&sphere) SphereTraceOperator<T>(op);
}

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const CylinderTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::Cylinder) {
    new (&cylinder) CylinderTraceOperator<T>(op);
}

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const PlaneTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::Plane) {
    new (&plane) PlaneTraceOperator<T>(op);
}

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const BoxTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::Box) {
    new (&box) BoxTraceOperator<T>(op);
}

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const TriangleTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::Triangle) {
    new (&triangle) TriangleTraceOperator<T>(op);
}

template <typename T>
ATLAS_HOST
TraceOperator<T>::TraceOperator(const BvhTraceOperator<T>& op)
    : type(atlas::geometry::GeometryType::TriangleMesh) {
    new (&triangle_mesh) BvhTraceOperator<T>(op);
}

/* ====================================================================== */
/* Trace dispatch                                                          */
/* ====================================================================== */

template <typename T>
HitSurface<T>
TraceOperator<T>::trace(const Ray<T>& ray) const {
    switch (type) {
    case atlas::geometry::GeometryType::Sphere:
        return sphere(ray);
    case atlas::geometry::GeometryType::Cylinder:
        return cylinder(ray);
    case atlas::geometry::GeometryType::Plane:
        return plane(ray);
    case atlas::geometry::GeometryType::Box:
        return box(ray);
    case atlas::geometry::GeometryType::Triangle:
        return triangle(ray);
    case atlas::geometry::GeometryType::TriangleMesh:
        return triangle_mesh(ray);
    default: {
        HitSurface<T> miss {};
        miss.is_intersecting = false;
        return miss;
    }
    }
}

template <typename T>
HitSurface<T>
TraceOperator<T>::operator()(const Ray<T>& ray) const {
    return trace(ray);
}

} // namespace atlas::spatial
