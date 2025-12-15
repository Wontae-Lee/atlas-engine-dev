#pragma once
#include <atlas/math/vector/elementwise.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <cmath>
#include <cstddef>

namespace atlas::geometry {
template <typename T>
ATLAS_DEVICE HitSurface<T>

BoxTraceOperator<T>::operator()(const Ray<T>& r) const {
    HitSurface<T> result;
    if (!lower || !upper) return result;
    const Vector3<T>& lo = *lower;
    const Vector3<T>& hi = *upper;
    const Vector3<T> inv_dir{ T(1) / r.direction.x, T(1) / r.direction.y, T(1) / r.direction.z };
    const T tx1    = (lo.x - r.origin.x) * inv_dir.x;
    const T tx2    = (hi.x - r.origin.x) * inv_dir.x;
    const T ty1    = (lo.y - r.origin.y) * inv_dir.y;
    const T ty2    = (hi.y - r.origin.y) * inv_dir.y;
    const T tz1    = (lo.z - r.origin.z) * inv_dir.z;
    const T tz2    = (hi.z - r.origin.z) * inv_dir.z;
    const T tminx  = tx1 < tx2 ? tx1 : tx2;
    const T tmaxx  = tx1 > tx2 ? tx1 : tx2;
    const T tminy  = ty1 < ty2 ? ty1 : ty2;
    const T tmaxy  = ty1 > ty2 ? ty1 : ty2;
    const T tminz  = tz1 < tz2 ? tz1 : tz2;
    const T tmaxz  = tz1 > tz2 ? tz1 : tz2;
    T t_enter      = tminx;
    int enter_axis = 0;
    if (tminy > t_enter) {
        t_enter    = tminy;
        enter_axis = 1;
    }
    if (tminz > t_enter) {
        t_enter    = tminz;
        enter_axis = 2;
    }
    const T t_exit = (tmaxx < tmaxy ? (tmaxx < tmaxz ? tmaxx : tmaxz) : (tmaxy < tmaxz ? tmaxy : tmaxz));
    if (t_exit < t_enter) return result;
    if (t_exit < T(eps)) return result;
    const bool use_enter = (t_enter >= T(eps));
    const T t            = use_enter ? t_enter : t_exit;
    Vector3<T> n(T(0), T(0), T(0));
    if (use_enter) {
        if (enter_axis == 0) n = Vector3<T>(-(r.direction.x >= T(0) ? T(1) : -T(1)), T(0), T(0));
        else if (enter_axis == 1) n = Vector3<T>(T(0), -(r.direction.y >= T(0) ? T(1) : -T(1)), T(0));
        else n                      = Vector3<T>(T(0), T(0), -(r.direction.z >= T(0) ? T(1) : -T(1)));
    } else {
        int exit_axis = 0;
        T best        = tmaxx;
        if (tmaxy <= best) {
            best      = tmaxy;
            exit_axis = 1;
        }
        if (tmaxz <= best) {
            best      = tmaxz;
            exit_axis = 2;
        }
        if (exit_axis == 0) n = Vector3<T>((r.direction.x >= T(0) ? T(1) : -T(1)), T(0), T(0));
        else if (exit_axis == 1) n = Vector3<T>(T(0), (r.direction.y >= T(0) ? T(1) : -T(1)), T(0));
        else n                     = Vector3<T>(T(0), T(0), (r.direction.z >= T(0) ? T(1) : -T(1)));
    }
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
      , upper_corner(T(+1), T(+1), T(+1)) {}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_, const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
      , upper_corner(upper_corner_) {
    ATLAS_ASSERT(is_valid());
}

template <typename T>
Vector3<T>
Box<T>::closest_point(const Vector3<T>& point) const {
    T cx = (point.x < lower_corner.x)
        ? lower_corner.x
        : (point.x > upper_corner.x)
        ? upper_corner.x
        : point.x;
    T cy = (point.y < lower_corner.y)
        ? lower_corner.y
        : (point.y > upper_corner.y)
        ? upper_corner.y
        : point.y;
    T cz = (point.z < lower_corner.z)
        ? lower_corner.z
        : (point.z > upper_corner.z)
        ? upper_corner.z
        : point.z;
    Vector3<T> cp(cx, cy, cz);
    if (is_inside(point)) {
        const Vector3<T> l_to_p = point - lower_corner;
        const Vector3<T> p_to_u = upper_corner - point;
        std::size_t axis        = 0;
        bool face               = false;
        if (l_to_p.min() < p_to_u.min()) {
            axis = l_to_p.minor_axis();
            face = false;
        } else {
            axis = p_to_u.minor_axis();
            face = true;
        }
        if (axis == 0) cp.x = face ? upper_corner.x : lower_corner.x;
        else if (axis == 1) cp.y = face ? upper_corner.y : lower_corner.y;
        else cp.z                = face ? upper_corner.z : lower_corner.z;
    }
    return cp;
}

template <typename T>
Vector3<T>
Box<T>::closest_normal(const Vector3<T>& point) const {
    Vector3<T> n(T(0), T(0), T(0));
    const T cx = (point.x < lower_corner.x)
        ? lower_corner.x
        : (point.x > upper_corner.x)
        ? upper_corner.x
        : point.x;
    const T cy = (point.y < lower_corner.y)
        ? lower_corner.y
        : (point.y > upper_corner.y)
        ? upper_corner.y
        : point.y;
    const T cz = (point.z < lower_corner.z)
        ? lower_corner.z
        : (point.z > upper_corner.z)
        ? upper_corner.z
        : point.z;
    const Vector3<T> cp(cx, cy, cz);
    if (is_inside(point)) {
        const Vector3<T> l_to_p = point - lower_corner;
        const Vector3<T> p_to_u = upper_corner - point;
        std::size_t axis        = 0;
        bool face               = false;
        if (l_to_p.min() < p_to_u.min()) {
            axis = l_to_p.minor_axis();
            face = false;
        } else {
            axis = p_to_u.minor_axis();
            face = true;
        }
        if (axis == 0) {
            n = face ? Vector3<T>(T(1), T(0), T(0)) : Vector3<T>(-T(1), T(0), T(0));
        } else if (axis == 1) {
            n = face ? Vector3<T>(T(0), T(1), T(0)) : Vector3<T>(T(0), -T(1), T(0));
        } else {
            n = face ? Vector3<T>(T(0), T(0), T(1)) : Vector3<T>(T(0), T(0), -T(1));
        }
    } else {
        const Vector3<T> d = point - cp;
        const Vector3<T> a = math::abs(d);
        std::size_t axis   = a.major_axis();
        if (axis == 0) {
            n = (d.x >= T(0)) ? Vector3<T>(T(1), T(0), T(0)) : Vector3<T>(-T(1), T(0), T(0));
        } else if (axis == 1) {
            n = (d.y >= T(0)) ? Vector3<T>(T(0), T(1), T(0)) : Vector3<T>(T(0), -T(1), T(0));
        } else {
            n = (d.z >= T(0)) ? Vector3<T>(T(0), T(0), T(1)) : Vector3<T>(T(0), T(0), -T(1));
        }
    }
    return n;
}

template <typename T>
T
Box<T>::signed_distance(const Vector3<T>& point) const {
    if (is_inside(point)) {
        T min1 = math::min(point - lower_corner);
        T min2 = math::min(upper_corner - point);
        return -(min1 < min2 ? min1 : min2);
    }
    return closest_distance(point);
}

template <typename T>
T
Box<T>::closest_distance(const Vector3<T>& point) const {
    Vector3<T> distance_vector = point - closest_point(point);
    return distance_vector.length();
}

template <typename T>
AABB<T>
Box<T>::bound() const {
    return AABB<T>(lower_corner, upper_corner);
}

template <typename T>
bool
Box<T>::intersects(const Ray<T>& ray) const {
    AABB<T> box(lower_corner, upper_corner);
    return box.intersects(ray);
}

template <typename T>
BoxTraceOperator<T>
Box<T>::make_trace_operator() const {
    BoxTraceOperator<T> op;
    op.lower = atlas::raw_pointer_cast(&lower_corner);
    op.upper = atlas::raw_pointer_cast(&upper_corner);
    return op;
}

template <typename T>
bool
Box<T>::is_inside(const Vector3<T>& point) const {
    return math::all((lower_corner <= point) & (point <= upper_corner));
}

template <typename T>
void
Box<T>::set_corners(const Vector3<T>& lower_corner_, const Vector3<T>& upper_corner_) noexcept {
    lower_corner = lower_corner_;
    upper_corner = upper_corner_;
    ATLAS_ASSERT(is_valid());
}

template <typename T>
Vector3<T>
Box<T>::center() const noexcept {
    return (lower_corner + upper_corner) * T(0.5);
}

template <typename T>
Vector3<T>
Box<T>::extents() const noexcept {
    return upper_corner - lower_corner;
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    return math::all(upper_corner >= lower_corner);
}
}