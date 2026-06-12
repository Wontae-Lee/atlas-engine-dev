#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
atlas::Vector<T, 3>
BoxGeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    // Without valid box corners, there is no meaningful projection target.
    if (!lower_corner || !upper_corner) {
        return p;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    // Clamp the point to the box bounds to get the closest point for exterior queries.
    atlas::Vector<T, 3> cp = atlas::clamp(p, lo, hi);

    const bool inside = atlas::all(p >= lo)
        && atlas::all(p <= hi);

    if (inside) {
        // For interior points, project to the nearest box face instead of returning p.
        const atlas::Vector<T, 3> l_to_p = p - lo;
        const atlas::Vector<T, 3> p_to_u = hi - p;

        // Select whether the nearest face lies on the lower or upper side.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Choose the axis with the shortest distance to the selected face.
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Move the projected point onto the nearest face.
        cp[axis] = hit_lower ? lo[axis] : hi[axis];
    }

    return cp;
}

template <typename T>
atlas::Vector<T, 3>
BoxGeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    // Invalid geometry cannot provide a reliable surface normal.
    if (!lower_corner || !upper_corner) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    const bool inside = atlas::all(p >= lo)
        && atlas::all(p <= hi);

    atlas::Vector<T, 3> n(T(0));

    if (inside) {
        // Interior normals point toward the nearest box face.
        const atlas::Vector<T, 3> l_to_p = p - lo;
        const atlas::Vector<T, 3> p_to_u = hi - p;

        // Select the nearest lower or upper face.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Use the axis with the smallest distance to that face.
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Lower faces have negative outward normals; upper faces have positive normals.
        n[axis] = hit_lower ? T(-1) : T(1);
        return n;
    }

    // Exterior normals are determined from the clamped closest point on the box.
    const atlas::Vector<T, 3> cp = atlas::clamp(p, lo, hi);
    const atlas::Vector<T, 3> d  = p - cp;

    // Use the dominant separation axis to select the closest face normal.
    const std::size_t axis = atlas::abs(d).major_axis();

    n[axis] = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxGeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!lower_corner || !upper_corner) {
        return atlas::inf;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    const bool inside = atlas::all(p >= lo)
        && atlas::all(p <= hi);

    if (inside) {
        // Inside the box, distance is negative and equals the nearest face distance.
        const atlas::Vector<T, 3> l_to_p = p - lo;
        const atlas::Vector<T, 3> p_to_u = hi - p;

        const T m1 = l_to_p.min();
        const T m2 = p_to_u.min();

        return -((m1 < m2) ? m1 : m2);
    }

    // Outside the box, distance is the Euclidean distance to the clamped point.
    const atlas::Vector<T, 3> cp = atlas::clamp(p, lo, hi);
    return (cp - p).length();
}

template <typename T>
bool
BoxGeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p,
                                  const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!lower_corner || !upper_corner) {
        return false;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    // Expand the box by tolerance to make boundary checks numerically robust.
    return atlas::all(p >= lo - tolerance)
        && atlas::all(p <= hi + tolerance);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p,
                                      const T tolerance) const noexcept {
    // Invalid geometry or negative tolerances cannot accept surface points.
    if (!lower_corner || !upper_corner || tolerance < T(0)) {
        return false;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;
    const T tolerance2                  = tolerance * tolerance;

    const bool inside = atlas::all(p >= lo)
        && atlas::all(p <= hi);

    if (inside) {
        const T dx = (p.x - lo.x < hi.x - p.x) ? p.x - lo.x : hi.x - p.x;
        const T dy = (p.y - lo.y < hi.y - p.y) ? p.y - lo.y : hi.y - p.y;
        const T dz = (p.z - lo.z < hi.z - p.z) ? p.z - lo.z : hi.z - p.z;
        const T d  = (dx < dy) ? ((dx < dz) ? dx : dz)
                               : ((dy < dz) ? dy : dz);

        return d <= tolerance;
    }

    const T dx = p.x < lo.x ? lo.x - p.x
        : p.x > hi.x       ? p.x - hi.x
                            : T(0);
    const T dy = p.y < lo.y ? lo.y - p.y
        : p.y > hi.y       ? p.y - hi.y
                            : T(0);
    const T dz = p.z < lo.z ? lo.z - p.z
        : p.z > hi.z       ? p.z - hi.z
                            : T(0);

    return dx * dx + dy * dy + dz * dz <= tolerance2;
}

template <typename T>
atlas::Vector<T, 3>
BoxGeometryOperator<T>::centroid() const noexcept {
    // Invalid geometry falls back to the origin as a neutral centroid.
    if (!lower_corner || !upper_corner) {
        return atlas::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The centroid of an axis-aligned box is the midpoint of its opposite corners.
    return ((*lower_corner) + (*upper_corner)) * T(0.5);
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
BoxGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!lower_corner || !upper_corner) {
        return atlas::AxisAlignedBoundingBox<T>();
    }

    // A box is already represented by its axis-aligned lower and upper bounds.
    return atlas::AxisAlignedBoundingBox<T>(*lower_corner, *upper_corner);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_valid() const noexcept {
    // Both corners must be bound before the operator can represent a box.
    if (!lower_corner || !upper_corner) {
        return false;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    // Each upper coordinate must be greater than or equal to the matching lower coordinate.
    return atlas::isfinite(lo)
        && atlas::isfinite(hi)
        && atlas::all(hi >= lo);
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::trace(const atlas::Ray<T>& r) const noexcept {
    HitSurface<T> result {};

    // Invalid geometry produces a default non-intersecting hit result.
    if (!lower_corner || !upper_corner) {
        return result;
    }

    const atlas::Vector<T, 3>& lo = *lower_corner;
    const atlas::Vector<T, 3>& hi = *upper_corner;

    // Compute reciprocal ray directions for the slab intersection test.
    const atlas::Vector<T, 3> inv_dir = T(1) / r.direction;

    // Intersect the ray with the lower and upper slabs on each axis.
    const atlas::Vector<T, 3> t0 = (lo - r.origin) * inv_dir;
    const atlas::Vector<T, 3> t1 = (hi - r.origin) * inv_dir;

    // Sort entry and exit distances per axis.
    const atlas::Vector<T, 3> tmin_v = atlas::cmin(t0, t1);
    const atlas::Vector<T, 3> tmax_v = atlas::cmax(t0, t1);

    // The ray enters after all lower slab crossings and exits before any upper slab crossing.
    const T t_enter = tmin_v.max();
    const T t_exit  = tmax_v.min();

    // Reject misses and intersections that lie fully behind the ray origin.
    if (t_exit < t_enter || t_exit < T(eps)) {
        return result;
    }

    // Use the entry distance for outside rays, or the exit distance when the ray starts inside.
    const bool use_enter = (t_enter >= T(eps));
    const T t            = use_enter ? t_enter : t_exit;

    // Select the face axis that produced the active entry or exit distance.
    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    atlas::Vector<T, 3> n(T(0));

    // Orient the normal outward from the hit face.
    const T dir = r.direction[axis];
    n[axis]     = use_enter ? ((dir >= T(0)) ? T(-1) : T(1))
                            : ((dir >= T(0)) ? T(1) : T(-1));

    // Populate the hit record with distance, point, and surface normal.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    result.normal          = n;

    return result;
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {
}

template <typename T>
Box<T>::Box(const Box& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
}

template <typename T>
Box<T>::Box(Box&& other) noexcept
    : lower_corner(std::move(other.lower_corner))
    , upper_corner(std::move(other.upper_corner)) {
}

template <typename T>
Box<T>&
Box<T>::operator=(const Box& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    lower_corner = other.lower_corner;
    upper_corner = other.upper_corner;

    return *this;
}

template <typename T>
Box<T>&
Box<T>::operator=(Box&& other) noexcept {
    // Avoid unnecessary rebinding on self move-assignment.
    if (this == &other) {
        return *this;
    }

    lower_corner = std::move(other.lower_corner);
    upper_corner = std::move(other.upper_corner);

    return *this;
}

template <typename T>
BoxGeometryOperator<T>
Box<T>::make_box_operator() const noexcept {
    BoxGeometryOperator<T> op {};
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op;
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the box fluently.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Box<T>::make_device_geometry_view() const {
    return GeometryOperator<T>(make_box_operator());
}

template <typename T>
atlas::Vector<T, 3>
Box<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return make_box_operator().closest_point(p);
}

template <typename T>
atlas::Vector<T, 3>
Box<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return make_box_operator().closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return make_box_operator().signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_box_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return make_box_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::Vector<T, 3>
Box<T>::centroid() const noexcept {
    return make_box_operator().centroid();
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {
    return make_box_operator().bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    return make_box_operator().is_valid();
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {
    // Identify this geometry as an axis-aligned box.
    return GeometryType::Box;
}

template <typename T>
Box<T>
Box<T>::Builder::build() const {
    // Validate corner ordering before constructing the box.
    validate();

    Box<T> b {};
    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {
    // Build a validated box and store it in host-managed shared ownership.
    auto b = build();
    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {
    // Store the lower corner used to define the box extent.
    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {
    // Store the upper corner used to define the box extent.
    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {
    atlas::BoxGeometryOperator<T> op;

    // Validate using the same operator logic used by constructed Box instances.
    op.lower_corner = atlas::raw_pointer_cast(&_lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&_upper_corner);

    if (!op.is_valid()) {
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

} // namespace atlas
