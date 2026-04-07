#pragma once

#include <cmath>
#include <limits>

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    if (!lower_corner || !upper_corner) return p;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const bool hit_lower                   = (l_to_p.min() < p_to_u.min());
        const std::size_t axis                 = hit_lower ? l_to_p.minor_axis()
                                                           : p_to_u.minor_axis();
        cp[axis]                               = hit_lower ? lo[axis] : hi[axis];
    }

    return cp;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    const bool inside                   = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    atlas::math::Vector<T, 3> n(T(0));

    if (inside) {
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const bool hit_lower                   = (l_to_p.min() < p_to_u.min());
        const std::size_t axis                 = hit_lower ? l_to_p.minor_axis()
                                                           : p_to_u.minor_axis();
        n[axis]                                = hit_lower ? T(-1) : T(1);
        return n;
    }

    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    const atlas::math::Vector<T, 3> d  = p - cp;
    const std::size_t axis             = atlas::math::abs(d).major_axis();
    n[axis]                            = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    if (!lower_corner || !upper_corner) return atlas::inf;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    const bool inside                   = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const T m1                             = l_to_p.min();
        const T m2                             = p_to_u.min();
        return -((m1 < m2) ? m1 : m2);
    }

    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    return (cp - p).length();
}

template <typename T>
bool
BoxGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (!lower_corner || !upper_corner) return false;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    return (p.x >= lo.x - tolerance) && (p.x <= hi.x + tolerance)
        && (p.y >= lo.y - tolerance) && (p.y <= hi.y + tolerance)
        && (p.z >= lo.z - tolerance) && (p.z <= hi.z + tolerance);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::centroid() const noexcept {
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
    return ((*lower_corner) + (*upper_corner)) * T(0.5);
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
BoxGeometryOperator<T>::bound() const noexcept {
    if (!lower_corner || !upper_corner) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }
    return atlas::spatial::AxisAlignedBoundingBox<T>(*lower_corner, *upper_corner);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_valid() const noexcept {
    if (!lower_corner || !upper_corner) return false;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    return (hi.x >= lo.x) && (hi.y >= lo.y) && (hi.z >= lo.z);
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    HitSurface<T> result {};
    if (!lower_corner || !upper_corner) return result;

    const atlas::math::Vector<T, 3>& lo     = *lower_corner;
    const atlas::math::Vector<T, 3>& hi     = *upper_corner;
    const atlas::math::Vector<T, 3> inv_dir = T(1) / r.direction;
    const atlas::math::Vector<T, 3> t0      = (lo - r.origin) * inv_dir;
    const atlas::math::Vector<T, 3> t1      = (hi - r.origin) * inv_dir;
    const atlas::math::Vector<T, 3> tmin_v  = atlas::math::cmin(t0, t1);
    const atlas::math::Vector<T, 3> tmax_v  = atlas::math::cmax(t0, t1);
    const T t_enter                         = tmin_v.max();
    const T t_exit                          = tmax_v.min();
    if (t_exit < t_enter || t_exit < T(eps)) return result;

    const bool use_enter   = (t_enter >= T(eps));
    const T t              = use_enter ? t_enter : t_exit;
    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    atlas::math::Vector<T, 3> n(T(0));
    const T dir = r.direction[axis];
    n[axis]     = use_enter ? ((dir >= T(0)) ? T(-1) : T(1))
                            : ((dir >= T(0)) ? T(1) : T(-1));

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {
    bind_operator();
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {
    bind_operator();
}

template <typename T>
Box<T>::Box(const Box& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
    bind_operator();
}

template <typename T>
Box<T>::Box(Box&& other) noexcept
    : lower_corner(std::move(other.lower_corner))
    , upper_corner(std::move(other.upper_corner)) {
    bind_operator();
    other.bind_operator();
}

template <typename T>
Box<T>&
Box<T>::operator=(const Box& other) noexcept {
    if (this == &other) return *this;

    lower_corner = other.lower_corner;
    upper_corner = other.upper_corner;
    bind_operator();
    return *this;
}

template <typename T>
Box<T>&
Box<T>::operator=(Box&& other) noexcept {
    if (this == &other) return *this;

    lower_corner = std::move(other.lower_corner);
    upper_corner = std::move(other.upper_corner);
    bind_operator();
    other.bind_operator();
    return *this;
}

template <typename T>
void
Box<T>::bind_operator() noexcept {
    _operator.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    _operator.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    this->invalidate_validity_cache();
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
GeometryOperator<T>
Box<T>::make_geometry_operator() const {
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    return _operator.signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::centroid() const noexcept {
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {
    return _operator.bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    return this->cached_is_valid([this]() noexcept {
        return _operator.is_valid();
    });
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {

    return GeometryType::Box;
}

template <typename T>
Box<T>
Box<T>::Builder::build() const {

    validate();

    Box<T> b {};

    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {

    auto b = build();
    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {

    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {

    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {

    atlas::geometry::BoxGeometryOperator<T> op;

    op.lower_corner = atlas::raw_pointer_cast(&_lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&_upper_corner);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Box::Builder validation failed: lower_corner must be <= upper_corner.";
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

}
