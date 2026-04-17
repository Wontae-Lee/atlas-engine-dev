#pragma once

#include <cmath>

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not bound to valid box corner data,
    // return the query point unchanged as a safe fallback.
    if (!lower_corner || !upper_corner) return p;

    // Read the lower/upper corners through the bound pointers.
    // `lo` is the component-wise minimum corner.
    // `hi` is the component-wise maximum corner.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Start with the clamped point.
    // For points outside the box, this already gives the closest point
    // on the box volume.
    atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    // Determine whether the query point lies inside the box or on its boundary.
    // Inclusive comparisons are used so points on faces/edges/corners are
    // treated as inside for the purpose of the projection rule below.
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        // For an interior point, plain clamping would return `p` itself.
        // This implementation instead returns the closest *surface* point.
        //
        // `l_to_p` stores distances from the lower faces.
        // `p_to_u` stores distances to the upper faces.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        // Decide whether the nearest face belongs to the lower side or upper side.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Select the axis of the nearest face:
        // - lower-side nearest axis if `hit_lower` is true
        // - upper-side nearest axis otherwise
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Snap the chosen coordinate to the corresponding face plane.
        // The other coordinates remain unchanged.
        cp[axis] = hit_lower ? lo[axis] : hi[axis];
    }

    // Return the closest point on the box surface.
    return cp;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not bound, no meaningful normal can be computed.
    // Return the zero vector as a neutral fallback.
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // Access the box bounds.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Check whether the point lies inside or on the boundary.
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    // Initialize the normal as zero and later assign exactly one axis component.
    atlas::math::Vector<T, 3> n(T(0));

    if (inside) {
        // For interior points, choose the outward normal of the nearest face,
        // using the same face-selection rule as `closest_point()`.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        // Determine whether the nearest face is on the lower side or upper side.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Select the axis of the nearest face.
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Assign the outward unit normal:
        // - lower face -> negative axis direction
        // - upper face -> positive axis direction
        n[axis] = hit_lower ? T(-1) : T(1);
        return n;
    }

    // For exterior points, first find the closest point on the box volume.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    // The vector from the closest point on the box to the query point
    // indicates the outward direction.
    const atlas::math::Vector<T, 3> d = p - cp;

    // Choose the dominant axis of separation.
    // This yields an axis-aligned normal that best represents the outward face direction.
    const std::size_t axis = atlas::math::abs(d).major_axis();

    // Set the normal sign according to which side of the box the query lies on.
    n[axis] = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not bound, report an infinite distance to indicate
    // that no valid geometric query can be performed.
    if (!lower_corner || !upper_corner) return atlas::inf;

    // Access the box corners.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Classify the query point.
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        // For interior points, signed distance is negative.
        // Its magnitude is the shortest distance to any face.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;
        const T m1                             = l_to_p.min();
        const T m2                             = p_to_u.min();

        // Return the negative of the nearest-face distance.
        return -((m1 < m2) ? m1 : m2);
    }

    // For exterior points, clamp to the box and measure Euclidean distance
    // from the query point to that closest point.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    return (cp - p).length();
}

template <typename T>
bool
BoxGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Without valid box bounds, containment cannot be established.
    if (!lower_corner || !upper_corner) return false;

    // Access the box corners.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Test whether `p` lies within the box expanded by `tolerance` on every side.
    // Positive tolerance loosens the containment test.
    // Zero tolerance performs an exact inclusive bound test.
    return (p.x >= lo.x - tolerance) && (p.x <= hi.x + tolerance)
        && (p.y >= lo.y - tolerance) && (p.y <= hi.y + tolerance)
        && (p.z >= lo.z - tolerance) && (p.z <= hi.z + tolerance);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the surface if its signed distance
    // is within the given absolute tolerance band around zero.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxGeometryOperator<T>::centroid() const noexcept {
    // If the operator is unbound, return the origin as a safe default.
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // The centroid of an axis-aligned box is the midpoint
    // between the lower and upper corners.
    return ((*lower_corner) + (*upper_corner)) * T(0.5);
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
BoxGeometryOperator<T>::bound() const noexcept {
    // If the operator is unbound, return a default-constructed bounding box.
    if (!lower_corner || !upper_corner) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    // Since this geometry is already an axis-aligned box,
    // its bound is simply the box defined by its own corners.
    return atlas::spatial::AxisAlignedBoundingBox<T>(*lower_corner, *upper_corner);
}

template <typename T>
bool
BoxGeometryOperator<T>::is_valid() const noexcept {
    // A valid box requires both corner pointers to be bound.
    if (!lower_corner || !upper_corner) return false;

    // Read the corners and verify component-wise ordering.
    // Degenerate boxes are still valid as long as `hi >= lo` for every axis.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;
    return (hi.x >= lo.x) && (hi.y >= lo.y) && (hi.z >= lo.z);
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    // Initialize the return record to the default "no hit" state.
    HitSurface<T> result {};

    // If the operator is not bound to valid corners, no intersection test can be done.
    if (!lower_corner || !upper_corner) return result;

    // Read the box bounds.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Precompute inverse ray direction for slab intersection.
    // This converts division into multiplication in later expressions.
    const atlas::math::Vector<T, 3> inv_dir = T(1) / r.direction;

    // Compute parametric intersection distances to the lower and upper slab planes
    // on each axis.
    const atlas::math::Vector<T, 3> t0 = (lo - r.origin) * inv_dir;
    const atlas::math::Vector<T, 3> t1 = (hi - r.origin) * inv_dir;

    // For each axis:
    // - `tmin_v` stores the earlier intersection
    // - `tmax_v` stores the later intersection
    //
    // This automatically handles both positive and negative ray directions.
    const atlas::math::Vector<T, 3> tmin_v = atlas::math::cmin(t0, t1);
    const atlas::math::Vector<T, 3> tmax_v = atlas::math::cmax(t0, t1);

    // The ray enters the box at the maximum of the three entry times
    // and exits at the minimum of the three exit times.
    const T t_enter = tmin_v.max();
    const T t_exit  = tmax_v.min();

    // Reject if:
    // - the entry occurs after the exit, meaning slabs do not overlap, or
    // - the whole intersection interval is behind the ray start / below epsilon.
    if (t_exit < t_enter || t_exit < T(eps)) return result;

    // If the entry point is in front of the origin, use it.
    // Otherwise the ray started inside the box, so use the exit point.
    const bool use_enter = (t_enter >= T(eps));
    const T t            = use_enter ? t_enter : t_exit;

    // Determine which axis produced the relevant hit:
    // - entering hit -> axis of largest entry time
    // - exiting hit  -> axis of smallest exit time
    const std::size_t axis = use_enter ? tmin_v.major_axis() : tmax_v.minor_axis();

    // Build the outward surface normal for the hit face.
    atlas::math::Vector<T, 3> n(T(0));
    const T dir = r.direction[axis];

    // For an entering hit, the normal opposes the ray direction along that axis.
    // For an exiting hit, the normal points with the ray direction along that axis.
    n[axis] = use_enter ? ((dir >= T(0)) ? T(-1) : T(1))
                        : ((dir >= T(0)) ? T(1) : T(-1));

    // Fill the hit record.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = r.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
BoxGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Forward call-operator syntax to the explicit ray trace routine.
    return trace(ray);
}

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {
    // Bind the cached operator so it points to this instance's corners.
    bind_operator();
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {
    // Bind the cached operator after direct corner initialization.
    bind_operator();
}

template <typename T>
Box<T>::Box(const Box& other) noexcept
    : lower_corner(other.lower_corner)
    , upper_corner(other.upper_corner) {
    // Rebind the operator so it references this object's copied members,
    // not the source object's members.
    bind_operator();
}

template <typename T>
Box<T>::Box(Box&& other) noexcept
    : lower_corner(std::move(other.lower_corner))
    , upper_corner(std::move(other.upper_corner)) {
    // After moving the corner data into this object, bind this operator
    // to the moved-in members.
    bind_operator();

    // Rebind the moved-from object's operator as well so its internal pointers
    // remain consistent with its own member storage after the move.
    other.bind_operator();
}

template <typename T>
Box<T>&
Box<T>::operator=(const Box& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy the corner values from the source.
    lower_corner = other.lower_corner;
    upper_corner = other.upper_corner;

    // Rebind because the cached operator must always reference this object's members.
    bind_operator();
    return *this;
}

template <typename T>
Box<T>&
Box<T>::operator=(Box&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move the corner data from the source object.
    lower_corner = std::move(other.lower_corner);
    upper_corner = std::move(other.upper_corner);

    // Rebind this object's cached operator to its current members.
    bind_operator();

    // Rebind the moved-from object's operator so its internal pointers
    // still point to its own member storage.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Box<T>::bind_operator() noexcept {
    // Make the cached operator reference this box's actual corner members.
    // This must be called whenever object construction, copy, or move may have
    // changed the addresses or ownership context of those members.
    _operator.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    _operator.upper_corner = atlas::raw_pointer_cast(&upper_corner);
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {
    // Return a fresh builder object for staged box construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Box<T>::make_geometry_operator() const {
    // Wrap the cached concrete box operator into the generic geometry operator interface.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::centroid() const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    // Forward the query to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {
    // Return the runtime geometry type tag for this class.
    return GeometryType::Box;
}

template <typename T>
Box<T>
Box<T>::Builder::build() const {
    // Validate the staged builder parameters before constructing the final object.
    validate();

    // Start from the default box instance.
    Box<T> b {};

    // Overwrite the default bounds with the validated builder values.
    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    // `b` already has its operator bound to its own members from the default
    // constructor, so no extra binding step is required here.
    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {
    // Build the validated box value first.
    auto b = build();

    // Move the built value into shared host-managed storage.
    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {
    // Store the lower corner in the builder.
    // Validation is intentionally deferred until `validate()` / `build()`.
    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {
    // Store the upper corner in the builder.
    // Validation is intentionally deferred until `validate()` / `build()`.
    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {
    // Use the same validity logic as the runtime geometry operator
    // to keep the definition of a valid box centralized.
    atlas::geometry::BoxGeometryOperator<T> op;

    // Bind the temporary operator to the builder-staged corners.
    op.lower_corner = atlas::raw_pointer_cast(&_lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&_upper_corner);

    // Reject invalid corner ordering with both a log entry and an exception.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Box::Builder validation failed: lower_corner must be <= upper_corner.";
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

} // namespace atlas::geometry