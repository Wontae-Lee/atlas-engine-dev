#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
Cylinder<T>::Cylinder() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1))
    , height(T(1))
    , open(false) {
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_)
    , open(false) {
}

template <typename T>
Cylinder<T>::Cylinder(const Cylinder& other) noexcept
    : center(other.center)
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
}

template <typename T>
Cylinder<T>::Cylinder(Cylinder&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(const Cylinder& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center = other.center;
    radius = other.radius;
    height = other.height;
    open   = other.open;

    return *this;
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(Cylinder&& other) noexcept {

    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    radius = other.radius;
    height = other.height;
    open   = other.open;

    return *this;
}

template <typename T>
CylinderGeometryOperator<T>
Cylinder<T>::make_cylinder_operator() const noexcept {
    CylinderGeometryOperator<T> op {};
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    op.open   = atlas::raw_pointer_cast(&open);
    return op;
}

template <typename T>
typename Cylinder<T>::Builder
Cylinder<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Cylinder<T>::make_device_geometry_view() const {
    return GeometryOperator<T>(make_cylinder_operator());
}

template <typename T>
atlas::Vector<T, 3>
Cylinder<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return make_cylinder_operator().closest_point(p);
}

template <typename T>
atlas::Vector<T, 3>
Cylinder<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return make_cylinder_operator().closest_normal(p);
}

template <typename T>
T
Cylinder<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return make_cylinder_operator().signed_distance(p);
}

template <typename T>
bool
Cylinder<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_cylinder_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Cylinder<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_cylinder_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
Cylinder<T>::centroid() const noexcept {
    return make_cylinder_operator().centroid();
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
Cylinder<T>::bound() const noexcept {
    return make_cylinder_operator().bound();
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    return make_cylinder_operator().is_valid();
}

template <typename T>
GeometryType
Cylinder<T>::type() const noexcept {

    return GeometryType::Cylinder;
}

template <typename T>
Cylinder<T>
Cylinder<T>::Builder::build() const {

    validate();

    Cylinder<T> c {};
    c.center = _center;
    c.radius = _radius;
    c.height = _height;
    c.open   = _open;

    return c;
}

template <typename T>
atlas::host_shared_ptr<Cylinder<T>>
Cylinder<T>::Builder::make_host_shared() const {

    auto c = build();
    return atlas::make_host_shared<Cylinder<T>>(std::move(c));
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_center(const Vector3<T>& center_) noexcept {

    _center = center_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_radius(T radius_) noexcept {

    _radius = radius_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_height(T height_) noexcept {

    _height = height_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_open(const bool open_) noexcept {

    _open = open_;
    return *this;
}

template <typename T>
void
Cylinder<T>::Builder::validate() const {
    atlas::CylinderGeometryOperator<T> op;

    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);
    op.height = atlas::raw_pointer_cast(&_height);
    op.open   = atlas::raw_pointer_cast(&_open);

    if (!op.is_valid()) {
        throw std::runtime_error("Cylinder::Builder: invalid parameters.");
    }
}

template <typename T>
T
CylinderGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !radius || !height) {
        return std::numeric_limits<T>::infinity();
    }

    const bool is_open_cylinder = open && *open;
    const T hz                  = (*height) * T(0.5);
    const atlas::Vector<T, 3> d = p - *center;

    const T rho = atlas::xy_length(d);

    const T qx = rho - *radius;
    const T qy = atlas::abs(d.z) - hz;

    if (is_open_cylinder) {

        if (qy <= T(0)) {
            return qx;
        }

        return atlas::sqrt_nonnegative(qx * qx + qy * qy);
    }

    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    const T outside = atlas::sqrt_nonnegative(ax * ax + ay * ay);

    const T mxy    = (qx > qy) ? qx : qy;
    const T inside = (mxy < T(0)) ? mxy : T(0);

    return outside + inside;
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {

    if (!center || !radius || !height) {
        return false;
    }

    const bool is_open_cylinder = open && *open;
    const T hz                  = (*height) * T(0.5);
    const atlas::Vector<T, 3> d = p - *center;

    const T rho = atlas::xy_length(d);
    const T qx  = rho - *radius;
    const T qy  = atlas::abs(d.z) - hz;

    if (is_open_cylinder) {

        return qy <= T(0) && qx <= tolerance;
    }

    if (qx <= T(0) && qy <= T(0)) {

        const T inside = (qx > qy) ? qx : qy;
        return inside <= tolerance;
    }

    if (tolerance < T(0)) {
        return false;
    }

    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    return (ax * ax + ay * ay) <= (tolerance * tolerance);
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (!center || !radius || !height
        || !(*radius > T(0)) || !(*height > T(0))
        || tolerance < T(0)) {
        return false;
    }

    if (center && radius && height && open && *open) {
        const T hz                  = (*height) * T(0.5);
        const atlas::Vector<T, 3> d = p - *center;

        const T rho  = atlas::xy_length(d);
        const T zmin = (*center).z - hz;
        const T zmax = (*center).z + hz;

        return atlas::abs(rho - *radius) <= tolerance
            && p.z >= zmin - tolerance
            && p.z <= zmax + tolerance;
    }

    const T hz                  = (*height) * T(0.5);
    const atlas::Vector<T, 3> d = p - *center;
    const T rho                 = atlas::xy_length(d);
    const T radial              = rho - *radius;
    const T axial               = atlas::abs(d.z) - hz;

    if (radial <= T(0) && axial <= T(0)) {
        const T inside_distance = (radial > axial) ? radial : axial;
        return inside_distance >= -tolerance;
    }

    const T ax = (radial > T(0)) ? radial : T(0);
    const T ay = (axial > T(0)) ? axial : T(0);

    return ax * ax + ay * ay <= tolerance * tolerance;
}

template <typename T>
atlas::Vector<T, 3>
CylinderGeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !radius || !height) {
        return p;
    }

    const T hz                  = (*height) * T(0.5);
    const T zmin                = (*center).z - hz;
    const T zmax                = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    const atlas::Vector<T, 3> d = p - *center;
    const T rho                 = atlas::xy_length(d);

    const T zc = (p.z < zmin) ? zmin
        : (p.z > zmax)        ? zmax
                              : p.z;

    T sx = p.x;
    T sy = p.y;

    if (rho > *radius) {

        const T inv = T(1) / rho;
        sx          = (*center).x + d.x * ((*radius) * inv);
        sy          = (*center).y + d.y * ((*radius) * inv);
    }

    atlas::Vector<T, 3> cp(sx, sy, zc);

    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {
        if (rho > T(0)) {

            const T inv = T(1) / rho;
            cp.x        = (*center).x + d.x * ((*radius) * inv);
            cp.y        = (*center).y + d.y * ((*radius) * inv);
        } else {

            cp.x = (*center).x + (*radius);
            cp.y = (*center).y;
        }

        cp.z = zc;
        return cp;
    }

    if (inside_radial && inside_z) {

        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            if (rho > T(0)) {

                const T inv = T(1) / rho;
                cp.x        = (*center).x + d.x * ((*radius) * inv);
                cp.y        = (*center).y + d.y * ((*radius) * inv);
            } else {

                cp.x = (*center).x + (*radius);
                cp.y = (*center).y;
            }

            cp.z = p.z;
        } else if (d_to_bot <= d_to_top) {

            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmin;
        } else {

            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmax;
        }
    }

    return cp;
}

template <typename T>
atlas::Vector<T, 3>
CylinderGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {

    if (!center || !radius || !height) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    const T hz                  = (*height) * T(0.5);
    const T zmin                = (*center).z - hz;
    const T zmax                = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    const atlas::Vector<T, 3> d = p - *center;
    const T rho                 = atlas::xy_length(d);

    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {

        const atlas::Vector<T, 3> cp = closest_point(p);
        const atlas::Vector<T, 3> cd = cp - *center;
        return atlas::xy_normalized_or(
            cd,
            atlas::Vector<T, 3>(T(1), T(0), T(0)));
    }

    if (inside_radial && inside_z) {

        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            return atlas::xy_normalized_or(
                d,
                atlas::Vector<T, 3>(T(1), T(0), T(0)));
        }

        return (d_to_bot <= d_to_top)
            ? atlas::Vector<T, 3>(T(0), T(0), -T(1))
            : atlas::Vector<T, 3>(T(0), T(0), T(1));
    }

    const atlas::Vector<T, 3> cp = closest_point(p);
    const T e                    = std::numeric_limits<T>::epsilon();

    if (atlas::abs(cp.z - zmin) <= e) {
        return atlas::Vector<T, 3>(T(0), T(0), -T(1));
    }

    if (atlas::abs(cp.z - zmax) <= e) {
        return atlas::Vector<T, 3>(T(0), T(0), T(1));
    }

    const atlas::Vector<T, 3> cd = cp - *center;
    return atlas::xy_normalized_or(
        cd,
        atlas::Vector<T, 3>(T(1), T(0), T(0)));
}

template <typename T>
atlas::Vector<T, 3>
CylinderGeometryOperator<T>::centroid() const noexcept {

    if (!center) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    return *center;
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
CylinderGeometryOperator<T>::bound() const noexcept {

    if (!center || !radius || !height) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    const T hz = (*height) * T(0.5);

    return atlas::AxisAlignedBoundingBox<T>(
        atlas::Vector<T, 3>((*center).x - *radius, (*center).y - *radius, (*center).z - hz),
        atlas::Vector<T, 3>((*center).x + *radius, (*center).y + *radius, (*center).z + hz));
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_valid() const noexcept {

    if (!radius || !height) {
        return false;
    }

    return (*radius) > T(0) && (*height) > T(0);
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::trace(const atlas::Ray<T>& ray) const noexcept {
    HitSurface<T> out {};

    if (!center || !radius || !height) {
        return out;
    }

    const atlas::Vector<T, 3> ro = ray.origin - *center;
    const atlas::Vector<T, 3> rd = ray.direction;

    const T r                   = *radius;
    const T hz                  = (*height) * T(0.5);
    const T zmin                = -hz;
    const T zmax                = hz;
    const bool is_open_cylinder = open && *open;

    T best_t = std::numeric_limits<T>::infinity();
    atlas::Vector<T, 3> best_n(T(0), T(0), T(0));

    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();

    if (rd.z == T(0)) {

        if (ro.z < zmin || ro.z > zmax) {
            z_ok = false;
        }
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

        const T A = atlas::xy_length_squared(rd);
        const T B = T(2) * atlas::xy_dot(ro, rd);
        const T C = atlas::xy_length_squared(ro) - r * r;

        if (A > T(0)) {
            const T disc = B * B - T(4) * A * C;

            if (disc >= T(0)) {
                const T sqrt_disc = atlas::sqrt_nonnegative(disc);

                const T sign_b = (B >= T(0)) ? T(1) : T(-1);
                const T q      = -T(0.5) * (B + sign_b * sqrt_disc);

                T t0 = (q == T(0)) ? std::numeric_limits<T>::infinity() : (C / q);
                T t1 = (A == T(0)) ? std::numeric_limits<T>::infinity() : (q / A);

                if (t0 > t1) {

                    const T tmp = t0;
                    t0          = t1;
                    t1          = tmp;
                }

                auto accept_side = [&](const T t) -> bool {
                    if (!(t >= T(0))) {
                        return false;
                    }

                    if (rd.z != T(0) && (t < t_z_enter || t > t_z_exit)) {
                        return false;
                    }

                    return true;
                };

                auto set_side_hit = [&](const T t) {
                    best_t = t;

                    const atlas::Vector<T, 3> ph = ro + rd * t;
                    best_n                       = atlas::xy_normalized_or(
                        ph,
                        atlas::Vector<T, 3>(T(1), T(0), T(0)));
                };

                if (accept_side(t0)) {
                    set_side_hit(t0);
                }

                if (!atlas::isfinite(best_t) && accept_side(t1)) {
                    set_side_hit(t1);
                }
            }
        }
    }

    if (!is_open_cylinder && rd.z != T(0)) {
        auto try_cap = [&](const T zplane, const T nz) {
            const T t = (zplane - ro.z) / rd.z;

            if (!(t >= T(0)) || t >= best_t) {
                return;
            }

            const atlas::Vector<T, 3> ph = ro + rd * t;

            if (atlas::xy_length_squared(ph) <= r * r) {

                best_t = t;
                best_n = atlas::Vector<T, 3>(T(0), T(0), nz);
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

    if (!atlas::isfinite(best_t)) {
        return out;
    }

    out.is_intersecting = true;
    out.distance        = best_t;
    out.point           = ray.point_at(best_t);
    out.normal          = best_n;

    return out;
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {

    return trace(ray);
}

}