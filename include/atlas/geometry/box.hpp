#pragma once

#include <cmath>
#include <limits>

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* BoxGeometryOperator<T>                                                     */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Return the closest point on (or inside) an axis-aligned box to point p.
    //
    // Inputs:
    // - p : query point in the same coordinate space as lower/upper corners.
    //
    // Output:
    // - closest point on the *surface* if p is inside,
    // - closest point on the box volume boundary if p is outside.
    //
    // Important design choice:
    // - For points inside the box, simply clamping would return p itself.
    //   That is a valid "closest point in the volume", but not a surface point.
    //   Here we intentionally return the closest *surface* point by pushing p
    //   to the nearest face.
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
        const bool hit_lower = (l_to_p.min() < p_to_u.min());
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();
        cp[axis] = hit_lower ? lo[axis] : hi[axis];
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
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    atlas::math::Vector<T, 3> n(T(0));

    if (inside) {
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const bool hit_lower = (l_to_p.min() < p_to_u.min());
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();
        n[axis] = hit_lower ? T(-1) : T(1);
        return n;
    }

    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    const atlas::math::Vector<T, 3> d = p - cp;
    const std::size_t axis = atlas::math::abs(d).major_axis();
    n[axis] = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    if (!lower_corner || !upper_corner) return atlas::inf;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const T m1 = l_to_p.min();
        const T m2 = p_to_u.min();
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

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    const atlas::math::Vector<T, 3> inv_dir = T(1) / r.direction;
    const atlas::math::Vector<T, 3> t0 = (lo - r.origin) * inv_dir;
    const atlas::math::Vector<T, 3> t1 = (hi - r.origin) * inv_dir;
    const atlas::math::Vector<T, 3> tmin_v = atlas::math::cmin(t0, t1);
    const atlas::math::Vector<T, 3> tmax_v = atlas::math::cmax(t0, t1);
    const T t_enter = tmin_v.max();
    const T t_exit = tmax_v.min();
    if (t_exit < t_enter || t_exit < T(eps)) return result;

    const bool use_enter = (t_enter >= T(eps));
    const T t = use_enter ? t_enter : t_exit;
    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    atlas::math::Vector<T, 3> n(T(0));
    const T dir = r.direction[axis];
    n[axis] = use_enter ? ((dir >= T(0)) ? T(-1) : T(1))
                        : ((dir >= T(0)) ? T(1) : T(-1));

    result.is_intersecting = true;
    result.distance = t;
    result.point = r.point_at(t);
    result.normal = n;
    return result;
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

/* ====================================================================== */
/* Box<T>                                                                  */
/* ====================================================================== */

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {
    // Default constructor creates a canonical axis-aligned box.
    //
    // Current policy:
    // - lower = (-1, -1, -1)
    // - upper = (+1, +1, +1)
    //
    // Rationale:
    // - Provides a sensible non-degenerate default volume.
    // - Keeps Box<T> immediately usable without additional setup.
    //
    // Notes:
    // - Being axis-aligned, "lower" must be component-wise <= "upper" to be valid.
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {
    // Construct a box directly from bounds.
    //
    // Caller responsibility:
    // - This constructor does NOT validate bounds.
    // - If lower_corner_ has components greater than upper_corner_,
    //   the box becomes invalid for most geometric queries.
    //
    // If you want enforced validity:
    // - Use Box<T>::builder().with_params(...).build()
    //   which runs validate().
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {
    // Entry point for fluent Builder construction.
    //
    // Builder is useful because:
    // - It can validate bounds and enforce invariants before producing Box<T>.
    // - It keeps construction readable in call sites with multiple parameters.
    return Builder {};
}


template <typename T>
GeometryOperator<T>
Box<T>::make_geometry_operator() const {
    // Create a GeometryOperator suitable for closest-point/normal/distance tests.
    //
    // Same lifetime and pointer considerations as make_geometry_operator().
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the closest point on (or in) the box to a query point p.
    //
    // Delegation approach:
    // - Build a temporary BoxGeometryOperator wired to this box's bounds.
    // - Reuse the operator's tested implementation.
    //
    // Performance note:
    // - Creating the operator is very cheap (two pointers).
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the outward normal of the closest surface feature to p.
    //
    // Behavior depends on operator definition:
    // - If p is outside: normal points outward from the nearest face/edge/corner.
    // - If p is inside : normal is typically the normal of the nearest face (tie-breaking
    //   rules may apply on edges/corners).
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to the box.
    //
    // Common convention (typical SDF for an AABB):
    // - Negative inside, zero on surface, positive outside.
    //
    // Delegates to the query operator for consistent behavior across the codebase.
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward host-side inside classification through the query operator so
    // Box<T> and BoxGeometryOperator<T> keep identical tolerance semantics.
    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Surface-band classification is delegated to the query operator for
    // consistency with all other box query entry points.
    return make_geometry_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::centroid() const noexcept {
    // Return the box centroid:
    //   c = (lower + upper) / 2
    //
    // Delegation ensures identical behavior between Box<T> and BoxGeometryOperator<T>.
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {
    // Return the axis-aligned bounding box of this box.
    //
    // For an axis-aligned box geometry, the bound is the box itself.
    // Still delegated for API uniformity with other geometry types.
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    // Validate bounds:
    // - A box is valid if lower_corner <= upper_corner component-wise.
    //
    // Delegation keeps the definition of validity centralized in BoxGeometryOperator<T>.
    atlas::geometry::BoxGeometryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.is_valid();
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {
    // Return the geometry type tag for this box.
    return GeometryType::Box;
}

/* ====================================================================== */
/* Box<T>::Builder                                                         */
/* ====================================================================== */

template <typename T>
Box<T>
Box<T>::Builder::build() const {
    // Build a Box<T> after validation.
    //
    // Strong exception guarantee:
    // - If validate() throws, no Box is returned.
    validate();

    Box<T> b {};

    // Copy builder state into the final object.
    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build the Box<T> by value
    // - Move it into a shared, heap-allocated object
    auto b = build();
    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {
    // Set lower corner (min corner) of the box.
    //
    // No validation here:
    // - Builder collects inputs; validation happens in validate()/build().
    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {
    // Set upper corner (max corner) of the box.
    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {
    // Validate builder state before constructing Box<T>.
    //
    // Rule:
    // - lower_corner must be component-wise <= upper_corner.
    //
    // We reuse BoxGeometryOperator<T>::is_valid() so "validity" is defined once.
    atlas::geometry::BoxGeometryOperator<T> op;

    // Here we point at builder-owned temporary storage (_lower_corner/_upper_corner).
    // That is safe because validate() uses them immediately.
    op.lower_corner = atlas::raw_pointer_cast(&_lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&_upper_corner);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Box::Builder validation failed: lower_corner must be <= upper_corner.";
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

} // namespace atlas::geometry
