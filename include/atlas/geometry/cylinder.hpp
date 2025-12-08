#pragma once
#include <atlas/memory/raw_pointer_cast.h>
#include <cmath>
#include <limits>

namespace atlas::geometry {
template <typename T>
ATLAS_DEVICE HitSurface<T>

CylinderTraceOperator<T>::operator()(const Ray<T>& ray) const {
    HitSurface<T> out;
    if (!center || !radius || !height) return out;
    const Vector3<T> ro = ray.origin - *center;
    const Vector3<T> rd = ray.direction;
    const T hz          = (*height) * T(0.5);
    const T zmin        = -hz;
    const T zmax        = hz;
    const T ox = ro.x, oy = ro.y, oz = ro.z;
    const T dx = rd.x, dy = rd.y, dz = rd.z;
    T best_t = std::numeric_limits<T>::infinity();
    Vector3<T> best_n(T(0), T(0), T(0));
    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();
    if (dz == T(0)) {
        if (oz < zmin || oz > zmax) z_ok = false;
    } else {
        const T inv_dz = T(1) / dz;
        t_z_enter      = (zmin - oz) * inv_dz;
        t_z_exit       = (zmax - oz) * inv_dz;
        if (t_z_enter > t_z_exit) {
            const T tmp = t_z_enter;
            t_z_enter   = t_z_exit;
            t_z_exit    = tmp;
        }
    }
    if (z_ok) {
        const T a = dx * dx + dy * dy;
        const T b = T(2) * (ox * dx + oy * dy);
        const T c = ox * ox + oy * oy - (*radius) * (*radius);
        if (a > T(0)) {
            const T disc = b * b - T(4) * a * c;
            if (disc >= T(0)) {
                const T sqrt_disc = static_cast<T>(std::sqrt(disc));
                const T sign_b    = (b >= T(0)) ? T(1) : T(-1);
                const T q         = -T(0.5) * (b + sign_b * sqrt_disc);
                T t0              = (q == T(0)) ? std::numeric_limits<T>::infinity() : (c / q);
                T t1              = (a == T(0)) ? std::numeric_limits<T>::infinity() : (q / a);
                if (t0 > t1) {
                    const T tmp = t0;
                    t0          = t1;
                    t1          = tmp;
                }
                auto accept_side = [&](T t) -> bool {
                    if (!(t >= T(0))) return false;
                    if (dz != T(0)) {
                        if (t < t_z_enter || t > t_z_exit) return false;
                    }
                    return true;
                };
                if (accept_side(t0)) {
                    best_t              = t0;
                    const Vector3<T> ph = ro + rd * best_t;
                    const T rr          = static_cast<T>(std::sqrt(ph.x * ph.x + ph.y * ph.y));
                    best_n              = (rr > T(0)) ? Vector3<T>(ph.x / rr, ph.y / rr, T(0)) : Vector3<T>(T(1), T(0), T(0));
                }
                if (!std::isfinite(best_t) && accept_side(t1)) {
                    best_t              = t1;
                    const Vector3<T> ph = ro + rd * best_t;
                    const T rr          = static_cast<T>(std::sqrt(ph.x * ph.x + ph.y * ph.y));
                    best_n              = (rr > T(0)) ? Vector3<T>(ph.x / rr, ph.y / rr, T(0)) : Vector3<T>(T(1), T(0), T(0));
                }
            }
        }
    }
    if (dz != T(0)) {
        auto try_cap = [&](T zplane, T nz) {
            const T t = (zplane - oz) / dz;
            if (!(t >= T(0))) return;
            if (t >= best_t) return;
            const T xh = ox + t * dx;
            const T yh = oy + t * dy;
            if (xh * xh + yh * yh <= (*radius) * (*radius)) {
                best_t = t;
                best_n = Vector3<T>(T(0), T(0), nz);
            }
        };
        const T t_cap_min = (zmin - oz) / dz;
        const T t_cap_max = (zmax - oz) / dz;
        if (t_cap_min < t_cap_max) {
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

template <typename T>
Cylinder<T>::Cylinder() noexcept {
    center = Vector3<T>(T(0), T(0), T(0));
    radius = T(1);
    height = T(1);
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_) {
    ATLAS_ASSERT(is_valid());
}

template <typename T>
T
Cylinder<T>::signed_distance(const Vector3<T>& point) const {
    const T hz      = height * T(0.5);
    const T dx      = point.x - center.x;
    const T dy      = point.y - center.y;
    const T rho     = std::sqrt(dx * dx + dy * dy);
    const T qx      = rho - radius;
    const T qy      = std::fabs(point.z - center.z) - hz;
    const T ax      = (qx > T(0)) ? qx : T(0);
    const T ay      = (qy > T(0)) ? qy : T(0);
    const T outside = std::sqrt(ax * ax + ay * ay);
    const T mxy     = (qx > qy) ? qx : qy;
    const T inside  = (mxy < T(0)) ? mxy : T(0);
    T sd            = outside + inside;
    return sd;
}

template <typename T>
Vector3<T>
Cylinder<T>::closest_point(const Vector3<T>& point) const {
    const T hz   = height * T(0.5);
    const T zmin = center.z - hz;
    const T zmax = center.z + hz;
    const T dx   = point.x - center.x;
    const T dy   = point.y - center.y;
    const T rho  = std::sqrt(dx * dx + dy * dy);
    T zc         = (point.z < zmin)
                ? zmin
                : (point.z > zmax)
                ? zmax
                : point.z;
    T sx, sy;
    if (rho > radius) {
        const T inv = T(1) / rho;
        sx          = center.x + dx * (radius * inv);
        sy          = center.y + dy * (radius * inv);
    } else {
        sx = point.x;
        sy = point.y;
    }
    Vector3<T> cp(sx, sy, zc);
    const bool inside_radial = (rho <= radius);
    const bool inside_z      = (point.z >= zmin) && (point.z <= zmax);
    if (inside_radial && inside_z) {
        const T d_to_side = radius - rho;
        const T d_to_bot  = point.z - zmin;
        const T d_to_top  = zmax - point.z;
        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                cp.x        = center.x + dx * (radius * inv);
                cp.y        = center.y + dy * (radius * inv);
            } else {
                cp.x = center.x + radius;
                cp.y = center.y;
            }
            cp.z = point.z;
        } else if (d_to_bot <= d_to_top) {
            cp.x = point.x;
            cp.y = point.y;
            cp.z = zmin;
        } else {
            cp.x = point.x;
            cp.y = point.y;
            cp.z = zmax;
        }
    }
    return cp;
}

template <typename T>
Vector3<T>
Cylinder<T>::closest_normal(const Vector3<T>& point) const {
    const T hz               = height * T(0.5);
    const T zmin             = center.z - hz;
    const T zmax             = center.z + hz;
    const T dx               = point.x - center.x;
    const T dy               = point.y - center.y;
    const T rho              = std::sqrt(dx * dx + dy * dy);
    const bool inside_radial = (rho <= radius);
    const bool inside_z      = (point.z >= zmin) && (point.z <= zmax);
    Vector3<T> n { T(0), T(0), T(0) };
    if (inside_radial && inside_z) {
        const T d_to_side = radius - rho;
        const T d_to_bot  = point.z - zmin;
        const T d_to_top  = zmax - point.z;
        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                n           = Vector3<T>(dx * inv, dy * inv, T(0));
            } else {
                n = Vector3<T>(T(1), T(0), T(0));
            }
        } else if (d_to_bot <= d_to_top) {
            n = Vector3<T>(T(0), T(0), -T(1));
        } else {
            n = Vector3<T>(T(0), T(0), T(1));
        }
    } else {
        const Vector3<T> cp = closest_point(point);
        const T eps         = std::numeric_limits<T>::epsilon();
        if (std::fabs(cp.z - zmin) <= eps) {
            n = Vector3<T>(T(0), T(0), -T(1));
        } else if (std::fabs(cp.z - zmax) <= eps) {
            n = Vector3<T>(T(0), T(0), T(1));
        } else {
            const T cdx = cp.x - center.x;
            const T cdy = cp.y - center.y;
            const T cr  = std::sqrt(cdx * cdx + cdy * cdy);
            if (cr > T(0)) {
                const T inv = T(1) / cr;
                n           = Vector3<T>(cdx * inv, cdy * inv, T(0));
            } else {
                n = Vector3<T>(T(1), T(0), T(0));
            }
        }
    }
    return n;
}

template <typename T>
T
Cylinder<T>::closest_distance(const Vector3<T>& point) const {
    T sd = signed_distance(point);
    return (sd >= T(0)) ? sd : -sd;
}

template <typename T>
AABB<T>
Cylinder<T>::bound() const {
    const T hz = height * T(0.5);
    const Vector3<T> lower(
        center.x - radius,
        center.y - radius,
        center.z - hz);
    const Vector3<T> upper(
        center.x + radius,
        center.y + radius,
        center.z + hz);
    return AABB<T>(lower, upper);
}

template <typename T>
bool
Cylinder<T>::intersects(const Ray<T>& ray) const {
    return trace(ray).is_intersecting;
}

template <typename T>
CylinderTraceOperator<T>
Cylinder<T>::make_trace_operator() const {
    CylinderTraceOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op;
}

template <typename T>
bool
Cylinder<T>::is_inside(const Vector3<T>& point) const {
    const T hz               = height * T(0.5);
    const T zmin             = center.z - hz;
    const T zmax             = center.z + hz;
    const T dx               = point.x - center.x;
    const T dy               = point.y - center.y;
    const T rho2             = dx * dx + dy * dy;
    const bool inside_radial = (rho2 <= radius * radius);
    const bool inside_z      = (point.z >= zmin) && (point.z <= zmax);
    return inside_radial && inside_z;
}

template <typename T>
void
Cylinder<T>::set_params(const Vector3<T>& center_, T radius_, T height_) noexcept {
    center = center_;
    radius = radius_;
    height = height_;
    ATLAS_ASSERT(is_valid());
}

template <typename T>
Vector3<T>
Cylinder<T>::extents() const noexcept {
    return Vector3<T>(T(2) * radius,
                      T(2) * radius,
                      height);
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    return radius >= T(0) && height >= T(0);
}
}