#pragma once

#include <cmath>

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    // If the box corners are not bound, the operator cannot evaluate a valid
    // geometric query. Return the input point unchanged as a safe fallback.
    if (!lower_corner || !upper_corner) return p;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // For a point outside the box, component-wise clamping directly gives the
    // closest point on the box.
    atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {

        // When the query point is inside the box, the closest point is not the
        // point itself for a surface query. Instead, project the point onto the
        // nearest box face.
        //
        // l_to_p: distance from the lower faces to the point
        // p_to_u: distance from the point to the upper faces
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        // Choose whether the nearest face belongs to the lower side or the
        // upper side by comparing the smallest clearance to each side.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Select the axis of the nearest face.
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Snap the corresponding coordinate onto the chosen face.
        cp[axis] = hit_lower ? lo[axis] : hi[axis];
    }

    return cp;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {

    // If the box corners are not available, no valid surface normal can be
    // determined. Return the zero vector as a fallback.
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    atlas::math::Vector<T, 3> n(T(0));

    if (inside) {

        // For interior points, the closest surface normal is the outward normal
        // of the nearest face.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        n[axis] = hit_lower ? T(-1) : T(1);
        return n;
    }

    // For exterior points, first compute the closest point on the box and then
    // infer the dominant outward direction from the displacement.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    const atlas::math::Vector<T, 3> d = p - cp;

    // Choose the axis with the largest absolute displacement.
    const std::size_t axis = atlas::math::abs(d).major_axis();

    n[axis] = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {

    // If the box corners are not bound, the signed-distance query is undefined.
    // Return +infinity to indicate an invalid or unreachable surface.
    if (!lower_corner || !upper_corner) return atlas::inf;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {

        // For interior points, the signed distance is negative and equal to the
        // distance to the nearest face.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const T m1                             = l_to_p.min();
        const T m2                             = p_to_u.min();

        return -((m1 < m2) ? m1 : m2);
    }

    // For exterior points, the signed distance is the Euclidean distance to the
    // closest point on the box surface and is therefore non-negative.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    return (cp - p).length();
}

template <typename T>
bool
BoxGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    // If the box is not properly bound, treat the query as outside.
    if (!lower_corner || !upper_corner) return false;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Expand the valid interior region by the supplied tolerance.
    return (p.x >= lo.x - tolerance) && (p.x <= hi.x + tolerance)
        && (p.y >= lo.y - tolerance) && (p.y <= hi.y + tolerance)
        && (p.z >= lo.z - tolerance) && (p.z <= hi.z + tolerance);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    // A point is considered on the surface when its signed distance is close
    // enough to zero within the requested tolerance band.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::centroid() const noexcept {

    // If the operator is not bound to valid corners, return the origin as a
    // safe fallback centroid.
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The centroid of an axis-aligned box is the midpoint between opposite corners.
    return ((*lower_corner) + (*upper_corner)) * T(0.5);
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
BoxGeometryOperator<T>::bound() const noexcept {

    // If the operator is not bound, return a default-constructed bounding box.
    if (!lower_corner || !upper_corner) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    return atlas::spatial::AxisAlignedBoundingBox<T>(*lower_corner, *upper_corner);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_valid() const noexcept {

    // A valid box requires both corners and a non-negative extent on each axis.
    if (!lower_corner || !upper_corner) return false;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    return (hi.x >= lo.x) && (hi.y >= lo.y) && (hi.z >= lo.z);
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {

    HitSurface<T> result {};

    // Without valid corners, the ray-box intersection query cannot proceed.
    if (!lower_corner || !upper_corner) return result;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Slab-based ray-box intersection:
    // compute parametric distances to the planes on each axis.
    const atlas::math::Vector<T, 3> inv_dir = T(1) / ray.direction;

    const atlas::math::Vector<T, 3> t0 = (lo - ray.origin) * inv_dir;
    const atlas::math::Vector<T, 3> t1 = (hi - ray.origin) * inv_dir;

    // For each axis:
    // - tmin_v contains the near-plane intersection
    // - tmax_v contains the far-plane intersection
    const atlas::math::Vector<T, 3> tmin_v = atlas::math::cmin(t0, t1);
    const atlas::math::Vector<T, 3> tmax_v = atlas::math::cmax(t0, t1);

    const T t_enter = tmin_v.max();
    const T t_exit  = tmax_v.min();

    // Reject if slabs do not overlap or if the whole box lies behind the ray.
    if (t_exit < t_enter || t_exit < T(eps)) return result;

    // If the entry point is in front of the ray origin, use it.
    // Otherwise the origin is already inside the box, so use the exit point.
    const bool use_enter = (t_enter >= T(eps));
    const T t            = use_enter ? t_enter : t_exit;

    // The intersected face corresponds to:
    // - the axis that produced the maximum near time on entry
    // - the axis that produced the minimum far time on exit
    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    atlas::math::Vector<T, 3> n(T(0));
    const T dir = ray.direction[axis];

    // Determine the outward normal based on whether this is an entering or
    // exiting hit and on the ray direction sign along the hit axis.
    n[axis] = use_enter ? ((dir >= T(0)) ? T(-1) : T(1))
                        : ((dir >= T(0)) ? T(1) : T(-1));

    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {

    // Make the geometry operator directly callable as a ray intersection query.
    return trace(ray);
}

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {

    // Bind the internal geometry operator to this object's corner storage.
    bind_operator();
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {

    // Bind the internal geometry operator to this object's corner storage.
    bind_operator();
}

template <typename T>
Box<T>::Box(const Box& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {

    // After copying the corner values, rebind the operator so it points to this
    // object's members rather than the source object's members.
    bind_operator();
}

template <typename T>
Box<T>::Box(Box&& other) noexcept
    : lower_corner(std::move(other.lower_corner))
    , upper_corner(std::move(other.upper_corner)) {

    // After moving the corner values, rebind this object's operator to its new
    // member addresses.
    bind_operator();

    // Rebind the moved-from object as well so its operator still references its
    // own members consistently after the move.
    other.bind_operator();
}

template <typename T>
Box<T>&
Box<T>::operator=(const Box& other) noexcept {

    if (this == &other) return *this;

    lower_corner = other.lower_corner;
    upper_corner = other.upper_corner;

    // Rebind because the operator stores raw pointers to member data.
    bind_operator();
    return *this;
}

template <typename T>
Box<T>&
Box<T>::operator=(Box&& other) noexcept {

    if (this == &other) return *this;

    lower_corner = std::move(other.lower_corner);
    upper_corner = std::move(other.upper_corner);

    // Rebind both objects after move assignment for the same reason as in the
    // move constructor.
    bind_operator();

    other.bind_operator();
    return *this;
}

template <typename T>
void
Box<T>::bind_operator() noexcept {

    // The lightweight geometry operator does not own the box coordinates.
    // Instead, it stores raw pointers to this Box object's corners so all
    // geometric queries operate directly on the current member values.
    _operator.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    _operator.upper_corner = atlas::raw_pointer_cast(&upper_corner);
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {

    // Return a default-initialized builder for fluent box construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Box<T>::make_geometry_operator() const {

    // Return a generic geometry operator wrapper around this box-specific operator.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::centroid() const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {

    // Forward the query to the bound box geometry operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {

    // Identify this geometry object as a box.
    return GeometryType::Box;
}

template <typename T>
Box<T>
Box<T>::Builder::build() const {

    // Validate user-provided box corners before constructing the final object.
    validate();

    Box<T> b {};

    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    // The default constructor already bound the operator to b's members, so
    // assigning the member values is sufficient here.
    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {

    // Build a value object first, then move it into host-managed shared storage.
    auto b = build();

    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {

    // Set the lower corner used for subsequent construction.
    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {

    // Set the upper corner used for subsequent construction.
    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {

    // Reuse the box geometry operator's validity rule rather than duplicating
    // the corner ordering logic here.
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