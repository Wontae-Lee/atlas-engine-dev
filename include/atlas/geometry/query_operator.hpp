#pragma once

#include <cmath>   // std::sqrt, std::fabs
#include <cstddef> // std::size_t
#include <limits>  // std::numeric_limits

namespace atlas::geometry {

/* ====================================================================== */
/* BoxQueryOperator<T>                                                     */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
BoxQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
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

    // Step 1: Clamp p to box extents.
    //
    // - If p is outside, clamp returns the nearest point on the boundary.
    // - If p is inside, clamp returns p unchanged (still inside).
    atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    // Determine if p lies inside (including boundary).
    const bool inside = (p.x >= lo.x && p.x <= hi.x)
        && (p.y >= lo.y && p.y <= hi.y)
        && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        // Step 2 (inside case): push to the nearest face.
        //
        // Distances from p to each pair of faces:
        // - distance to lower faces: p - lo
        // - distance to upper faces: hi - p
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        // Choose which side (lower vs upper) is closer by comparing the minimum
        // distance among components.
        //
        // Example:
        // - If min(p-lo) is smaller, nearest face is on a lower side.
        // - Otherwise nearest face is on an upper side.
        const bool hit_lower = (l_to_p.min() < p_to_u.min());

        // Choose axis of nearest face:
        // - minor_axis() returns the index of the smallest component.
        // - For hit_lower: the closest lower face axis = argmin(p - lo)
        // - For hit_upper: the closest upper face axis = argmin(hi - p)
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Snap that axis coordinate to the face plane.
        // Other coordinates remain unchanged.
        cp[axis] = hit_lower ? lo[axis] : hi[axis];
    }

    return cp;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Return the outward normal of the closest surface feature to p.
    //
    // Outside case:
    // - Compute cp = clamp(p, lo, hi)
    // - d = p - cp points from surface to p
    // - Dominant component of d indicates which face is closest.
    //
    // Inside case:
    // - Determine nearest face using distances to lo/hi
    // - Normal is ± axis unit vector.
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    const bool inside = (p.x >= lo.x && p.x <= hi.x) && (p.y >= lo.y && p.y <= hi.y) && (p.z >= lo.z && p.z <= hi.z);

    atlas::math::Vector<T, 3> n(T(0));

    if (inside) {
        // Same nearest-face logic as closest_point(inside).
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        const bool hit_lower   = (l_to_p.min() < p_to_u.min());
        const std::size_t axis = hit_lower ? l_to_p.minor_axis()
                                           : p_to_u.minor_axis();

        // Outward normal:
        // - lower face -> -axis direction
        // - upper face -> +axis direction
        n[axis] = hit_lower ? T(-1) : T(1);
        return n;
    }

    // Outside: closest point on the box (in the volume sense) is the clamped point.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);

    // Vector from surface point to query.
    // - For a clean "outside" point, at least one component of d is nonzero.
    const atlas::math::Vector<T, 3> d = p - cp;

    // Pick the face corresponding to the largest magnitude displacement.
    // This avoids ambiguous normals when p is diagonally outside near an edge/corner.
    const std::size_t axis = atlas::math::abs(d).major_axis();

    // Normal sign is based on whether p is on the + or - side of the clamped point.
    n[axis] = (d[axis] >= T(0)) ? T(1) : T(-1);
    return n;
}

template <typename T>
T
BoxQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to an axis-aligned box.
    //
    // Conventions used here:
    // - Outside: return positive Euclidean distance to the clamped point.
    // - Inside : return negative distance to the nearest face
    //           (i.e., penetration depth with negative sign).
    if (!lower_corner || !upper_corner) return atlas::inf;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    const bool inside = (p.x >= lo.x && p.x <= hi.x) && (p.y >= lo.y && p.y <= hi.y) && (p.z >= lo.z && p.z <= hi.z);

    if (inside) {
        // For inside points, distance to surface is the minimum distance to any face.
        const atlas::math::Vector<T, 3> l_to_p = p - lo;
        const atlas::math::Vector<T, 3> p_to_u = hi - p;

        // Nearest face distance is min of the mins.
        const T m1   = l_to_p.min();
        const T m2   = p_to_u.min();
        const T dmin = (m1 < m2) ? m1 : m2;

        // Negative inside.
        return -dmin;
    }

    // Outside:
    // - Clamp and return distance to the clamped point.
    const atlas::math::Vector<T, 3> cp = atlas::math::clamp(p, lo, hi);
    return (cp - p).length();
}

template <typename T>
bool
BoxQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p is inside the box, allowing a configurable tolerance.
    //
    // Geometric meaning:
    // - Without tolerance, the test is the standard inclusive AABB containment test:
    //     lo.x <= p.x <= hi.x, etc.
    //
    // - With tolerance > 0, the box is effectively expanded outward by `tolerance`
    //   in every direction before testing containment.
    //   This is useful for:
    //   * numerical robustness near boundaries,
    //   * treating near-surface points as still "inside",
    //   * avoiding false negatives caused by floating-point roundoff.
    //
    // - With tolerance < 0, the valid interior region shrinks, making the test stricter.
    //   In practice, callers usually pass a non-negative tolerance.
    //
    // Fallback policy:
    // - If the operator is not bound to valid box corners, there is no meaningful
    //   containment test, so return false.
    if (!lower_corner || !upper_corner) return false;

    // Access the stored box corners by reference for readability and to avoid repeated dereference.
    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    // Inclusive per-axis containment test with tolerance padding:
    //
    // X axis:
    //   p.x must lie in [lo.x - tolerance, hi.x + tolerance]
    //
    // Y axis:
    //   p.y must lie in [lo.y - tolerance, hi.y + tolerance]
    //
    // Z axis:
    //   p.z must lie in [lo.z - tolerance, hi.z + tolerance]
    //
    // Because the checks are combined with logical AND, p is considered inside
    // only if it lies within the tolerated interval on all three axes.
    return (p.x >= lo.x - tolerance) && (p.x <= hi.x + tolerance)
        && (p.y >= lo.y - tolerance) && (p.y <= hi.y + tolerance)
        && (p.z >= lo.z - tolerance) && (p.z <= hi.z + tolerance);
}

template <typename T>
bool
BoxQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the box surface within the given tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which already encodes the distance to the box surface:
    //     * negative inside
    //     * zero on the surface
    //     * positive outside
    //
    // - A point is considered "on the surface" if the magnitude of that signed distance
    //   is no larger than `tolerance`.
    //
    // In other words:
    //   |signed_distance(p)| <= tolerance
    //
    // Interpretation:
    // - Exact arithmetic / tolerance = 0:
    //     only points exactly on the surface return true.
    //
    // - Positive tolerance:
    //     points slightly inside or slightly outside the surface are also accepted,
    //     which is often desirable for robust geometric classification.
    //
    // Design benefit:
    // - This keeps the surface test consistent with the operator's signed-distance definition,
    //   rather than duplicating face-by-face boundary logic here.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
BoxQueryOperator<T>::centroid() const noexcept {
    // Centroid of an axis-aligned box:
    //   (lo + hi) / 2
    if (!lower_corner || !upper_corner) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
    return ((*lower_corner) + (*upper_corner)) * T(0.5);
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
BoxQueryOperator<T>::bound() const noexcept {
    // Bounding box of a box is itself.
    if (!lower_corner || !upper_corner) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }
    return atlas::spatial::AxisAlignedBoundingBox<T>(*lower_corner, *upper_corner);
}

template <typename T>
bool
BoxQueryOperator<T>::is_valid() const noexcept {
    // Validity rule:
    // - Both pointers must exist
    // - Each component must satisfy lo <= hi
    if (!lower_corner || !upper_corner) return false;

    const atlas::math::Vector<T, 3>& lo = *lower_corner;
    const atlas::math::Vector<T, 3>& hi = *upper_corner;

    return (hi.x >= lo.x) && (hi.y >= lo.y) && (hi.z >= lo.z);
}

/* ====================================================================== */
/* SphereQueryOperator<T>                                                  */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
SphereQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on sphere surface to p.
    //
    // Sphere definition:
    // - center c
    // - radius r
    //
    // For p != c:
    // - direction v = (p - c)
    // - closest point = c + r * normalize(v)
    //
    // Degenerate case:
    // - If p is extremely close to center, direction is undefined.
    //   We return a point on +X axis of the sphere.
    if (!center || !radius) return p;

    const atlas::math::Vector<T, 3> v = p - *center;
    const T len2                      = v.length_squared();
    const T e                         = std::numeric_limits<T>::epsilon();

    if (len2 <= e) {
        // p is (numerically) at center; choose an arbitrary point on the sphere.
        return atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y, (*center).z);
    }

    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return (*center) + v * ((*radius) * inv_len);
}

template <typename T>
atlas::math::Vector<T, 3>
SphereQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Outward normal of sphere at the closest point.
    //
    // For p not at center:
    // - normal = normalize(p - c)
    //
    // Degenerate case:
    // - If p ~ center, return +X as arbitrary normal.
    if (!center || !radius) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    const atlas::math::Vector<T, 3> v = p - *center;
    const T len2                      = v.length_squared();
    const T e                         = std::numeric_limits<T>::epsilon();

    if (len2 <= e) return atlas::math::Vector<T, 3>(T(1), T(0), T(0));

    const T inv_len = T(1) / static_cast<T>(std::sqrt(len2));
    return v * inv_len;
}

template <typename T>
T
SphereQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to sphere:
    //   |p - c| - r
    //
    // Negative inside, positive outside (common SDF convention).
    if (!center || !radius) return std::numeric_limits<T>::infinity();
    return (p - *center).length() - *radius;
}

template <typename T>
bool
SphereQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies inside the sphere, allowing a configurable tolerance.
    //
    // Sphere model:
    // - center : c
    // - radius : r
    //
    // Exact containment test (no tolerance):
    //   |p - c| <= r
    //
    // Tolerance handling:
    // - We interpret tolerance by expanding the radius:
    //     expanded_radius = r + tolerance
    //
    //   Then the test becomes:
    //     |p - c| <= expanded_radius
    //
    // Meaning:
    // - tolerance > 0:
    //     sphere grows outward, so near-outside points can still count as inside
    //
    // - tolerance = 0:
    //     exact inclusive sphere containment
    //
    // - tolerance < 0:
    //     sphere shrinks inward, making the test stricter
    //
    // This is useful for:
    // - robust geometric classification near the boundary
    // - compensating for floating-point error
    // - intentionally shrinking/expanding the accepted region
    //
    // Fallback policy:
    // - If the operator is not bound to a valid center/radius, return false.
    if (!center || !radius) return false;

    // Adjust radius by tolerance.
    const T expanded_radius = *radius + tolerance;

    // If the effective radius becomes negative, the accepted interior is empty.
    // In that case no point can be considered inside.
    if (expanded_radius < T(0)) return false;

    // Use squared-distance comparison to avoid an unnecessary sqrt:
    //
    //   |p - c| <= expanded_radius
    // is equivalent to
    //   |p - c|^2 <= expanded_radius^2
    //
    // This is both cheaper and numerically common for containment tests.
    return (p - *center).length_squared() <= expanded_radius * expanded_radius;
}

template <typename T>
bool
SphereQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the sphere surface within the given tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for a sphere is:
    //
    //     sd = |p - center| - radius
    //
    //   so:
    //   - sd < 0 : inside
    //   - sd = 0 : exactly on the surface
    //   - sd > 0 : outside
    //
    // - A point is treated as "on the surface" if the magnitude of that signed
    //   distance is at most `tolerance`:
    //
    //     |sd| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     exact surface membership
    //
    // - tolerance > 0:
    //     accept a thin shell around the sphere surface, both slightly inside
    //     and slightly outside
    //
    // Design benefit:
    // - This keeps surface classification fully consistent with the signed-distance
    //   convention already used elsewhere in the operator.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
SphereQueryOperator<T>::centroid() const noexcept {
    // Centroid of sphere is its center.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
SphereQueryOperator<T>::bound() const noexcept {
    // Axis-aligned bounding box of sphere:
    // - min = c - (r,r,r)
    // - max = c + (r,r,r)
    if (!center || !radius) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> dr(*radius, *radius, *radius);
    return atlas::spatial::AxisAlignedBoundingBox<T>((*center) - dr, (*center) + dr);
}

template <typename T>
bool
SphereQueryOperator<T>::is_valid() const noexcept {
    // Valid if radius pointer exists and radius is non-negative.
    // Note:
    if (!radius) return false;
    return (*radius) > T(0);
}

/* ====================================================================== */
/* PlaneQueryOperator<T>                                                   */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
PlaneQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on plane to p using projection.
    //
    // Plane implicit form:
    //   n · x + d = 0
    //
    // Signed distance (scaled by |n| if n not unit):
    //   s = n · p + d
    //
    // Projection (assuming n is unit):
    //   cp = p - s * n
    //
    // If n is not unit, the correct projection would be:
    //   cp = p - (s / |n|^2) * n
    //
    // This operator assumes the provided normal is already unit-length
    // (or that the caller accepts the scaled behavior).
    if (!normal || !offset) return p;

    const T sdev = ((*normal).dot(p) + (*offset));
    return p - sdev * (*normal);
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Plane normal is constant everywhere.
    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *normal;
}

template <typename T>
T
PlaneQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance (not necessarily normalized):
    //   sd = n · p + d
    if (!normal || !offset) return std::numeric_limits<T>::infinity();
    return (*normal).dot(p) + (*offset);
}

template <typename T>
bool
PlaneQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the "inside" side of the plane, allowing tolerance.
    //
    // Plane implicit form:
    //   n · x + d = 0
    //
    // The scalar
    //   s = n · p + d
    // classifies the point relative to the plane:
    //
    // - s < 0 : point is on the negative side of the plane
    // - s = 0 : point lies exactly on the plane
    // - s > 0 : point is on the positive side of the plane
    //
    // This operator defines "inside" as:
    //   s <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     inside includes the plane itself and the entire negative half-space
    //
    // - tolerance > 0:
    //     the accepted half-space is expanded slightly into the positive side,
    //     which is useful for numerical robustness near the plane
    //
    // - tolerance < 0:
    //     the accepted region shrinks and becomes stricter
    //
    // Important note:
    // - This is a half-space test, not a bounded-volume test.
    // - For a plane, "inside" always means one side of the infinite plane.
    //
    // Fallback policy:
    // - If either normal or offset is missing, there is no valid plane equation,
    //   so return false.
    if (!normal || !offset) return false;

    // Evaluate the plane equation at p and compare against tolerance.
    return (*normal).dot(p) + (*offset) <= tolerance;
}

template <typename T>
bool
PlaneQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the plane surface within the given tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for this operator is:
    //
    //     sd = n · p + d
    //
    //   Note:
    //   - This is the signed plane equation value.
    //   - It is the true geometric signed distance only if the normal is unit length.
    //   - If the normal is not normalized, this is still a consistent signed
    //     classification value, just scaled by |n|.
    //
    // Surface test:
    //   |sd| <= tolerance
    //
    // Meaning:
    // - tolerance = 0:
    //     only points exactly satisfying the plane equation count as on-surface
    //
    // - tolerance > 0:
    //     accept a thin slab around the plane, which helps with floating-point
    //     robustness and near-surface classification
    //
    // Design benefit:
    // - This keeps the surface test fully consistent with the same signed-distance
    //   convention used elsewhere by the operator.
    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
PlaneQueryOperator<T>::centroid() const noexcept {
    // Infinite plane has no unique centroid.
    // This implementation returns the origin as a stable placeholder.
    return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
PlaneQueryOperator<T>::bound() const noexcept {
    // Infinite plane has an infinite AABB.
    //
    // Implementation uses numeric extremes as a sentinel "infinite" box.
    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>(lo, lo, lo),
        atlas::math::Vector<T, 3>(hi, hi, hi));
}

template <typename T>
bool
PlaneQueryOperator<T>::is_valid() const noexcept {
    // Validity rules:
    // - both pointers must be present
    // - normal must be non-zero length
    // - offset must be finite
    if (!normal || !offset) return false;

    const T n2 = (*normal).length_squared();
    return (n2 > T(0)) && std::isfinite(static_cast<double>(*offset));
}

/* ====================================================================== */
/* CylinderQueryOperator<T>                                                */
/* ====================================================================== */

template <typename T>
T
CylinderQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to a finite axis-aligned cylinder (Z axis).
    //
    // Cylinder definition:
    // - center c
    // - radius r in XY
    // - height h along Z
    //
    // Typical SDF construction:
    // - qx = radial_distance - r
    // - qy = abs(z) - h/2
    // - outside = length(max(q,0))
    // - inside  = min(max(qx,qy), 0)
    //
    // This produces:
    // - positive outside
    // - negative inside
    if (!center || !radius || !height) return std::numeric_limits<T>::infinity();

    const T hz = (*height) * T(0.5);

    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    const T qx = rho - *radius;
    const T qy = static_cast<T>(std::fabs(d.z)) - hz;

    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    const T outside = static_cast<T>(std::sqrt(ax * ax + ay * ay));
    const T mxy     = (qx > qy) ? qx : qy;
    const T inside  = (mxy < T(0)) ? mxy : T(0);

    return outside + inside;
}

template <typename T>
bool
CylinderQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies inside or on a finite cylinder,
    // allowing a tolerance.
    //
    // Cylinder definition:
    // - center : cylinder center
    // - radius : radius in the x-y plane
    // - height : total extent along the z-axis
    //
    // Geometry:
    // - The cylinder is centered at *center
    // - Its axis is aligned with the z-axis
    // - Its axial half-height is:
    //
    //     hz = height / 2
    //
    // - A point is inside the exact cylinder if both hold:
    //
    //     rho <= radius
    //     |dz| <= hz
    //
    //   where:
    //
    //     d   = p - center
    //     rho = sqrt(dx^2 + dy^2)
    //     dz  = d.z
    //
    // Signed-distance-style reduction:
    // - Define the radial and axial offsets from the ideal cylinder bounds:
    //
    //     qx = rho - radius
    //     qy = |dz| - hz
    //
    // Interpretation:
    // - qx <= 0 : point is within the radial bound
    // - qx >  0 : point is outside radially
    // - qy <= 0 : point is within the axial bound
    // - qy >  0 : point is outside axially
    //
    // Cases:
    //
    // 1) Point projects inside both bounds:
    //
    //      qx <= 0 and qy <= 0
    //
    //    Then the point is inside the cylinder.
    //    In this case, the signed-distance-style inside value is:
    //
    //      inside = max(qx, qy)
    //
    //    which is:
    //    - negative inside the volume
    //    - zero on the side wall or top/bottom caps
    //
    //    The acceptance test is:
    //
    //      inside <= tolerance
    //
    // 2) Point is outside in at least one direction:
    //
    //      qx > 0 or qy > 0
    //
    //    Then the shortest outside distance to the cylinder is obtained from
    //    the positive excesses only:
    //
    //      ax = max(qx, 0)
    //      ay = max(qy, 0)
    //
    //    and the outside-distance test becomes:
    //
    //      ax^2 + ay^2 <= tolerance^2
    //
    // Tolerance interpretation:
    // - tolerance = 0:
    //     accept only points exactly inside or on the cylinder
    //
    // - tolerance > 0:
    //     expand the accepted region slightly outside the cylinder,
    //     which improves robustness near edges and corners
    //
    // - tolerance < 0:
    //     for outside points, this operator rejects immediately
    //     because a negative outside allowance is not meaningful here
    //     in the distance-squared test
    //
    // Important note:
    // - This is a bounded-volume test for a finite cylinder,
    //   not an infinite-cylinder test.
    //
    // Fallback policy:
    // - If center, radius, or height is missing, the cylinder is not
    //   properly defined, so return false.
    if (!center || !radius || !height) return false;

    const T hz                        = (*height) * T(0.5);
    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));
    const T qx                        = rho - *radius;
    const T qy                        = static_cast<T>(std::fabs(d.z)) - hz;

    if (qx <= T(0) && qy <= T(0)) {
        const T inside = (qx > qy) ? qx : qy;
        return inside <= tolerance;
    }

    if (tolerance < T(0)) return false;

    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);
    return (ax * ax + ay * ay) <= (tolerance * tolerance);
}

template <typename T>
bool
CylinderQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the cylinder surface within tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for this operator is interpreted as a
    //   signed-distance-style value for a finite z-aligned cylinder.
    //
    // Sign convention:
    // - signed_distance(p) < 0 : point is inside
    // - signed_distance(p) = 0 : point is on the surface
    // - signed_distance(p) > 0 : point is outside
    //
    // Surface test:
    //   |signed_distance(p)| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     only points exactly on the cylinder surface are accepted
    //
    // - tolerance > 0:
    //     accept a thin shell around the surface, which helps with
    //     floating-point robustness near the side wall, top cap,
    //     bottom cap, and edge rim
    //
    // Important note:
    // - This surface includes all boundary parts of the finite cylinder:
    //   the curved side wall, the top and bottom caps, and their circular rims.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on a finite axis-aligned cylinder.
    //
    // Outline:
    // 1) Compute z clamp to [zmin,zmax]
    // 2) Compute radial clamp to radius (project onto circle if outside)
    // 3) If point is inside both radial and z extents, push it to nearest surface:
    //    - side wall OR bottom cap OR top cap
    if (!center || !radius || !height) return p;

    const T hz   = (*height) * T(0.5);
    const T zmin = (*center).z - hz;
    const T zmax = (*center).z + hz;

    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Clamp z into finite cylinder span.
    const T zc = (p.z < zmin) ? zmin
        : (p.z > zmax)        ? zmax
                              : p.z;

    // Start with x,y = p.x,p.y (will be adjusted if outside radial).
    T sx = p.x;
    T sy = p.y;

    // If outside radial extent, project onto side circle at radius r.
    if (rho > *radius) {
        const T inv = T(1) / rho;
        sx          = (*center).x + d.x * ((*radius) * inv);
        sy          = (*center).y + d.y * ((*radius) * inv);
    }

    atlas::math::Vector<T, 3> cp(sx, sy, zc);

    // Determine whether p is fully inside the finite cylinder volume.
    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (inside_radial && inside_z) {
        // Inside volume: the closest surface is whichever is nearest:
        // - side wall (distance = r - rho)
        // - bottom cap (distance = p.z - zmin)
        // - top cap (distance = zmax - p.z)
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            // Nearest is side wall: project to radius at same z.
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                cp.x        = (*center).x + d.x * ((*radius) * inv);
                cp.y        = (*center).y + d.y * ((*radius) * inv);
            } else {
                // On axis: choose arbitrary radial direction (+X).
                cp.x = (*center).x + (*radius);
                cp.y = (*center).y;
            }
            cp.z = p.z;
        } else if (d_to_bot <= d_to_top) {
            // Nearest is bottom cap: keep x,y and snap z to zmin.
            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmin;
        } else {
            // Nearest is top cap: keep x,y and snap z to zmax.
            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmax;
        }
    }

    return cp;
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest surface normal of a finite cylinder.
    //
    // Inside volume:
    // - Choose normal of the nearest surface (side/bottom/top) similarly to closest_point().
    //
    // Outside volume:
    // - Find closest point (cp)
    // - Determine whether cp lies on a cap or on side wall
    // - Return corresponding normal.
    if (!center || !radius || !height) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    const T hz   = (*height) * T(0.5);
    const T zmin = (*center).z - hz;
    const T zmax = (*center).z + hz;

    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (inside_radial && inside_z) {
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            // Side wall normal is radial (x,y,0) normalized.
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                return atlas::math::Vector<T, 3>(d.x * inv, d.y * inv, T(0));
            }
            // On axis: arbitrary radial normal.
            return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
        }

        // Cap normals:
        // - bottom cap points -Z
        // - top cap points +Z
        return (d_to_bot <= d_to_top)
            ? atlas::math::Vector<T, 3>(T(0), T(0), -T(1))
            : atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    // Outside: infer which feature the closest point lies on.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T e                          = std::numeric_limits<T>::epsilon();

    // If cp lies on a cap plane (within epsilon), return cap normal.
    if (static_cast<T>(std::fabs(cp.z - zmin)) <= e) return atlas::math::Vector<T, 3>(T(0), T(0), -T(1));
    if (static_cast<T>(std::fabs(cp.z - zmax)) <= e) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    // Otherwise, treat as side wall: normal is radial from center to cp.
    const atlas::math::Vector<T, 3> cd = cp - *center;
    const T rr2                        = cd.x * cd.x + cd.y * cd.y;

    if (rr2 > T(0)) {
        const T inv = T(1) / static_cast<T>(std::sqrt(rr2));
        return atlas::math::Vector<T, 3>(cd.x * inv, cd.y * inv, T(0));
    }

    // Degenerate: choose arbitrary normal.
    return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderQueryOperator<T>::centroid() const noexcept {
    // Centroid of cylinder is its center.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CylinderQueryOperator<T>::bound() const noexcept {
    // AABB of axis-aligned cylinder:
    // - x,y extents: center ± radius
    // - z extent   : center.z ± height/2
    if (!center || !radius || !height) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const T hz = (*height) * T(0.5);
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>((*center).x - *radius, (*center).y - *radius, (*center).z - hz),
        atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y + *radius, (*center).z + hz));
}

template <typename T>
bool
CylinderQueryOperator<T>::is_valid() const noexcept {
    // Valid if:
    // - radius/height pointers exist
    // - parameters are non-negative
    //
    // Note:
    // - Some systems require strictly positive radius/height;
    if (!radius || !height) return false;
    return (*radius) > T(0) && (*height) > T(0);
}

/* ====================================================================== */
/* TriangleQueryOperator<T>                                                */
/* ====================================================================== */

template <typename T>
atlas::math::Vector<T, 3>
TriangleQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Closest point on a triangle to point p.
    //
    // This is the classic region-based test (Christer Ericson, RTCD):
    // - Check vertex regions outside A, B, C
    // - Check edge regions AB, AC, BC
    // - Otherwise inside face region (use barycentric coordinates)
    //
    // Requires triangle vertices a,b,c.
    if (!a || !b || !c) return p;

    const atlas::math::Vector<T, 3> v0 = *a;
    const atlas::math::Vector<T, 3> v1 = *b;
    const atlas::math::Vector<T, 3> v2 = *c;

    const atlas::math::Vector<T, 3> ab = v1 - v0;
    const atlas::math::Vector<T, 3> ac = v2 - v0;
    const atlas::math::Vector<T, 3> ap = p - v0;

    const T d1 = ab.dot(ap);
    const T d2 = ac.dot(ap);
    if (d1 <= T(0) && d2 <= T(0)) return v0; // Vertex region A

    const atlas::math::Vector<T, 3> bp = p - v1;
    const T d3                         = ab.dot(bp);
    const T d4                         = ac.dot(bp);
    if (d3 >= T(0) && d4 <= d3) return v1; // Vertex region B

    const T vc = d1 * d4 - d3 * d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        // Edge region AB
        const T vv = d1 / (d1 - d3);
        return v0 + ab * vv;
    }

    const atlas::math::Vector<T, 3> cpv = p - v2;
    const T d5                          = ab.dot(cpv);
    const T d6                          = ac.dot(cpv);
    if (d6 >= T(0) && d5 <= d6) return v2; // Vertex region C

    const T vb = d5 * d2 - d1 * d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        // Edge region AC
        const T ww = d2 / (d2 - d6);
        return v0 + ac * ww;
    }

    const T va = d3 * d6 - d5 * d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        // Edge region BC
        const T ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * ww;
    }

    // Inside face region: compute barycentric coordinates.
    const T denom = T(1) / (va + vb + vc);
    const T vv    = vb * denom;
    const T ww    = vc * denom;
    return v0 + ab * vv + ac * ww;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Return triangle normal.
    //
    // Priority:
    // 1) If explicit normal pointer `n` is provided, return it.
    // 2) Otherwise compute geometric normal cross(b-a, c-a) and normalize.
    if (n) return *n;

    if (!a || !b || !c) {
        // Fallback: arbitrary up normal if geometry not set.
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T len2                 = nn.length_squared();

    if (len2 > T(0)) {
        nn *= (T(1) / static_cast<T>(std::sqrt(len2)));
    } else {
        // Degenerate triangle: choose a fallback normal.
        nn = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    return nn;
}

template <typename T>
T
TriangleQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to a triangle (not a plane).
    //
    // Approach:
    // - Compute closest point on triangle.
    // - Use plane-side sign from triangle normal:
    //     sd_plane = dot(p - a, n)
    // - Distance magnitude is |p - cp|.
    //
    // This yields a signed distance that is consistent with the triangle's
    // orientation (normal direction), but note:
    // - For a thin open surface, "inside/outside" is not globally defined.
    if (!a || !b || !c) return std::numeric_limits<T>::infinity();

    const atlas::math::Vector<T, 3> nn = closest_normal(p);
    const T sd_plane                   = (p - (*a)).dot(nn);

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d                          = (p - cp).length();

    return (sd_plane >= T(0)) ? d : -d;
}

template <typename T>
bool
TriangleQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p is considered "inside" relative to the triangle,
    // allowing a tolerance.
    //
    // Important note:
    // - A triangle is a 2D surface element in 3D space, not a volumetric region.
    // - Therefore, "inside" here does not mean inside a bounded 3D volume.
    // - Instead, this operator uses:
    //   1) the oriented supporting plane of the triangle, and
    //   2) the closest-point distance to the triangle itself.
    //
    // Triangle definition:
    // - Vertices: a, b, c
    // - Optional stored normal: n
    //
    // If no normal is provided, use the geometric triangle normal:
    //
    //   nn = (b - a) × (c - a)
    //
    // This normal defines the oriented supporting plane of the triangle.
    //
    // Degeneracy:
    // - If the triangle normal has zero length, the triangle is degenerate
    //   (zero area), so classification is undefined and this function returns false.
    //
    // Closest-point strategy:
    // - Compute the closest point cp on the triangle to p.
    //
    //     cp = closest_point(p)
    //
    // - Then compute the squared Euclidean distance:
    //
    //     d2 = |p - cp|^2
    //
    // This gives the shortest distance from p to the finite triangle
    // (including its interior, edges, and vertices).
    //
    // Oriented side test:
    // - Compute:
    //
    //     side = (p - a) · nn
    //
    // Interpretation:
    // - side < 0 : p lies on the negative side of the triangle plane
    // - side = 0 : p lies on the supporting plane
    // - side > 0 : p lies on the positive side of the triangle plane
    //
    // Inside convention:
    // - This operator treats the negative side of the oriented triangle plane
    //   as the "inside" side.
    //
    // Cases:
    //
    // 1) side <= 0
    //
    //    The point is on the plane or on the negative side.
    //
    //    - If tolerance >= 0:
    //        accept immediately
    //
    //      This means the entire negative half-space is considered inside,
    //      independent of the finite-triangle distance.
    //
    //    - If tolerance < 0:
    //        require:
    //
    //            d2 >= tolerance^2
    //
    //      Since tolerance^2 is positive, this excludes points that are too
    //      close to the triangle surface while still remaining on the inside side.
    //
    // 2) side > 0
    //
    //    The point is on the positive side of the plane.
    //
    //    - If tolerance < 0:
    //        reject immediately
    //
    //    - If tolerance >= 0:
    //        accept only if the point lies within tolerance distance of the
    //        finite triangle:
    //
    //            d2 <= tolerance^2
    //
    // Tolerance interpretation:
    // - tolerance = 0:
    //     accept the full negative half-space and the triangle surface itself;
    //     on the positive side, only exact surface points are accepted
    //
    // - tolerance > 0:
    //     expand acceptance slightly into the positive side, but only near the
    //     finite triangle surface
    //
    // - tolerance < 0:
    //     shrink acceptance on the inside side by excluding points too close
    //     to the triangle surface, and reject all points on the positive side
    //
    // Important note:
    // - This is not a pure point-to-triangle inclusion test.
    // - It is an oriented half-space classification combined with a finite-triangle
    //   proximity test near the positive side of the surface.
    //
    // Fallback policy:
    // - If any vertex is missing, or if the triangle is degenerate,
    //   return false.
    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T d2                         = (p - cp).length_squared();

    atlas::math::Vector<T, 3> nn = n ? *n : atlas::math::cross((*b) - (*a), (*c) - (*a));
    const T nn_len2              = nn.length_squared();
    if (nn_len2 <= T(0)) return false;

    const T side = (p - (*a)).dot(nn);

    if (side <= T(0)) {
        if (tolerance >= T(0)) return true;
        return d2 >= (tolerance * tolerance);
    }

    if (tolerance < T(0)) return false;
    return d2 <= (tolerance * tolerance);
}

template <typename T>
bool
TriangleQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the triangle surface within tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which for this operator is expected to provide
    //   a signed-distance-style value consistent with the triangle surface.
    //
    // Surface test:
    //   |signed_distance(p)| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     only points exactly on the triangle surface are accepted
    //
    // - tolerance > 0:
    //     accept a thin neighborhood around the triangle, including regions
    //     near its interior, edges, and vertices
    //
    // Important note:
    // - For full consistency, signed_distance(p) should follow the same surface
    //   convention as closest_point(p) and the finite triangle geometry.
    return std::abs(signed_distance(p)) <= tolerance;
}
template <typename T>
atlas::math::Vector<T, 3>
TriangleQueryOperator<T>::centroid() const noexcept {
    // Centroid of triangle:
    //   (a + b + c) / 3
    if (!a || !b || !c) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return ((*a) + (*b) + (*c)) * (T(1) / T(3));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleQueryOperator<T>::bound() const noexcept {
    // AABB of triangle = component-wise min/max over vertices.
    if (!a || !b || !c) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const atlas::math::Vector<T, 3> mn = atlas::math::cmin(*a, atlas::math::cmin(*b, *c));
    const atlas::math::Vector<T, 3> mx = atlas::math::cmax(*a, atlas::math::cmax(*b, *c));
    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleQueryOperator<T>::is_valid() const noexcept {
    // Valid if:
    // - all three vertices exist
    // - triangle has non-zero area (cross product magnitude > 0)
    if (!a || !b || !c) return false;

    const atlas::math::Vector<T, 3> nn = atlas::math::cross((*b) - (*a), (*c) - (*a));
    return nn.length_squared() > T(0);
}

/* ====================================================================== */
/* TriangleMeshQueryOperator<T>                                            */
/* ====================================================================== */

template <typename T>
T
TriangleMeshQueryOperator<T>::solid_angle(const atlas::math::Vector<T, 3>& p,
                                          const atlas::math::Vector<T, 3>& a,
                                          const atlas::math::Vector<T, 3>& b,
                                          const atlas::math::Vector<T, 3>& c) const noexcept {
    const atlas::math::Vector<T, 3> va = a - p;
    const atlas::math::Vector<T, 3> vb = b - p;
    const atlas::math::Vector<T, 3> vc = c - p;

    const T la = va.length();
    const T lb = vb.length();
    const T lc = vc.length();

    if (la <= std::numeric_limits<T>::epsilon() || lb <= std::numeric_limits<T>::epsilon()
        || lc <= std::numeric_limits<T>::epsilon()) {
        return T(0);
    }

    const T numerator   = va.dot(atlas::math::cross(vb, vc));
    const T denominator = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;

    return T(2) * std::atan2(numerator, denominator);
}

template <typename T>
T
TriangleMeshQueryOperator<T>::winding_number(const atlas::math::Vector<T, 3>& p) const noexcept {
    T solid_angle_sum = T(0);

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        solid_angle_sum += solid_angle(p, a, b, c);
    }

    const T four_pi = T(4) * std::acos(T(-1));
    return solid_angle_sum / four_pi;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshQueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Brute-force closest point on a triangle mesh.
    //
    // Validity:
    // - vertices and indices must be non-null
    // - triangle_count > 0
    //
    // Complexity:
    // - O(triangle_count) per query. This is simple but can be expensive.
    // - For acceleration, typical next step is BVH / spatial hashing / AABB tree.
    if (!is_valid()) return p;

    T best_d2                         = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_cp = p;

    TriangleQueryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        // Each triangle uses 3 indices.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        // Fetch triangle vertices.
        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Compute and normalize triangle normal (for signed distance / normal queries).
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        // Wire operator pointers.
        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        // Compute closest point and track best (smallest squared distance).
        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
            best_cp = cp;
        }
    }

    return best_cp;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshQueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Brute-force closest normal:
    // - Find the triangle whose closest point is nearest to p
    // - Return that triangle's normal (via TriangleQueryOperator)
    if (!is_valid()) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    T best_d2 = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(1));

    TriangleQueryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
            best_n  = tri.closest_normal(p);
        }
    }

    return best_n;
}

template <typename T>
T
TriangleMeshQueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Brute-force mesh distance magnitude with winding-number sign.
    //
    // Sign policy:
    // - Use the generalized winding number induced by the triangle winding.
    // - Closed consistently wound meshes classify points with |w| > 0.5 as inside.
    // - Open meshes typically produce winding numbers near zero, so the distance remains positive.
    if (!is_valid()) return std::numeric_limits<T>::infinity();

    T best_d2 = std::numeric_limits<T>::max();

    TriangleQueryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist = static_cast<T>(std::sqrt(best_d2));
    if (dist <= std::numeric_limits<T>::epsilon()) return T(0);

    const T winding = winding_number(p);
    return (std::abs(winding) > T(0.5)) ? -dist : dist;
}

template <typename T>
bool
TriangleMeshQueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p is considered inside the triangle mesh,
    // allowing a tolerance around the surface.
    //
    // Classification strategy:
    // - Use the mesh winding number at p:
    //
    //     winding = winding_number(p)
    //
    // - For a closed, consistently wound mesh:
    //     |winding| > 0.5  => inside
    //     |winding| <= 0.5 => outside
    //
    // Interpretation:
    // - The winding number provides a global inside/outside classification
    //   for properly oriented closed triangle meshes.
    // - In practice, values near ±1 indicate interior points and values near 0
    //   indicate exterior points.
    //
    // Surface-distance handling:
    // - In addition to winding classification, compute the shortest distance
    //   from p to the mesh surface by scanning all triangles and taking the
    //   minimum point-to-triangle distance.
    //
    //     dist = min distance from p to the mesh
    //
    // - This distance is then used only to apply tolerance around the surface.
    //
    // Tolerance interpretation:
    // - tolerance = 0:
    //     accept exactly the winding-defined interior and the surface itself
    //
    // - tolerance > 0:
    //     expand acceptance slightly outside the mesh by 'tolerance'
    //
    // - tolerance < 0:
    //     shrink the accepted interior by excluding points whose distance
    //     to the mesh surface is smaller than |tolerance|
    //
    // Cases:
    // - If p is winding-classified as inside:
    //     * tolerance >= 0 : accept
    //     * tolerance <  0 : accept only if dist >= -tolerance
    //
    // - If p is winding-classified as outside:
    //     * tolerance <  0 : reject
    //     * tolerance >= 0 : accept only if dist <= tolerance
    //
    // Important limitations:
    // - The winding-number test is meaningful primarily for closed meshes
    //   with consistent triangle orientation.
    // - For open meshes, self-intersecting meshes, or inconsistent winding,
    //   the result may not match a well-defined volumetric interior.
    //
    // Fallback policy:
    // - If the mesh is invalid, return false.
    if (!is_valid()) return false;

    T best_d2 = std::numeric_limits<T>::max();

    TriangleQueryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist      = static_cast<T>(std::sqrt(best_d2));
    const T winding   = winding_number(p);
    const bool inside = std::abs(winding) > T(0.5);

    if (inside) {
        if (tolerance >= T(0)) return true;
        return dist >= -tolerance;
    }

    if (tolerance < T(0)) return false;
    return dist <= tolerance;
}

template <typename T>
bool
TriangleMeshQueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Return whether point p lies on the mesh surface within tolerance.
    //
    // Strategy:
    // - Reuse signed_distance(p), which is expected to provide a signed-distance-style
    //   value consistent with the mesh surface.
    //
    // Surface test:
    //   |signed_distance(p)| <= tolerance
    //
    // Interpretation:
    // - tolerance = 0:
    //     only points exactly on the mesh surface are accepted
    //
    // - tolerance > 0:
    //     accept a thin neighborhood around the mesh surface, including regions
    //     near triangle interiors, edges, and vertices
    //
    // Important note:
    // - For full consistency, signed_distance(p) should use the same surface-distance
    //   convention as the nearest-point distance used in is_inside().
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshQueryOperator<T>::centroid() const noexcept {
    // Return a simple centroid estimate for the triangle mesh.
    //
    // Strategy:
    // - Compute the centroid of each triangle:
    //
    //     (a + b + c) / 3
    //
    // - Then average those triangle centroids uniformly over all triangles.
    //
    // Interpretation:
    // - Each triangle contributes equally, regardless of area.
    // - Therefore, this is a simple per-triangle average, not an area-weighted
    //   surface centroid and not a volume centroid.
    //
    // Consequence:
    // - For meshes with strongly non-uniform triangle areas, this estimate may
    //   differ noticeably from the geometric surface centroid.
    //
    // Fallback policy:
    // - If the mesh is invalid, return the zero vector.
    if (!is_valid()) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    atlas::math::Vector<T, 3> sum(T(0), T(0), T(0));

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        sum += (a + b + c) * (T(1) / T(3));
    }

    return sum * (T(1) / static_cast<T>(triangle_count));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMeshQueryOperator<T>::bound() const noexcept {
    // Compute mesh AABB by scanning all vertices referenced by indices.
    //
    // Complexity:
    // - O(triangle_count) index reads.
    //
    // Assumes:
    // - indices are valid indices into vertices.
    if (!is_valid()) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const int i0                 = indices[0];
    atlas::math::Vector<T, 3> mn = vertices[i0];
    atlas::math::Vector<T, 3> mx = vertices[i0];

    for (int t = 0; t < triangle_count; ++t) {
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        mn = atlas::math::cmin(mn, vertices[j0]);
        mn = atlas::math::cmin(mn, vertices[j1]);
        mn = atlas::math::cmin(mn, vertices[j2]);

        mx = atlas::math::cmax(mx, vertices[j0]);
        mx = atlas::math::cmax(mx, vertices[j1]);
        mx = atlas::math::cmax(mx, vertices[j2]);
    }

    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleMeshQueryOperator<T>::is_valid() const noexcept {
    // Basic structural validity:
    // - vertex/ index buffers exist
    // - triangle_count is positive
    //
    // Note:
    // - Does not validate index ranges (that would be more expensive).
    if (!vertices || !indices) return false;
    if (triangle_count <= 0) return false;
    return true;
}

/* ====================================================================== */
/* QueryOperator<T> (tagged union dispatcher)                              */
/* ====================================================================== */

template <typename T>

QueryOperator<T>::QueryOperator() noexcept
    : type(GeometryType::Sphere)
    , sphere() {
    // Default query operator is a sphere operator.
    //
    // Rationale:
    // - Safe default shape.
    // - Ensures QueryOperator<T> is always usable after default construction.
    //
    // Note:
    // - sphere() here is default-constructed SphereQueryOperator<T>,
    //   which is expected to be a lightweight POD-like object.
}

template <typename T>

QueryOperator<T>::QueryOperator(const QueryOperator& other) noexcept
    : type(other.type) {
    // Copy constructor for tagged union.
    //
    // Rule:
    // - Copy ONLY the active member selected by `type`.
    // - For an invalid `type`, fall back to sphere for safety.
    switch (type) {
    case GeometryType::Box:
        box = other.box;
        return;
    case GeometryType::Cylinder:
        cylinder = other.cylinder;
        return;
    case GeometryType::Plane:
        plane = other.plane;
        return;
    case GeometryType::Sphere:
        sphere = other.sphere;
        return;
    case GeometryType::Triangle:
        triangle = other.triangle;
        return;
    case GeometryType::TriangleMesh:
        triangle_mesh = other.triangle_mesh;
        return;
    default:
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return;
    }
}

template <typename T>
QueryOperator<T>&
QueryOperator<T>::operator=(const QueryOperator& other) noexcept {
    // Copy assignment for tagged union.
    if (this == &other) return *this;

    type = other.type;

    switch (type) {
    case GeometryType::Box:
        box = other.box;
        return *this;
    case GeometryType::Cylinder:
        cylinder = other.cylinder;
        return *this;
    case GeometryType::Plane:
        plane = other.plane;
        return *this;
    case GeometryType::Sphere:
        sphere = other.sphere;
        return *this;
    case GeometryType::Triangle:
        triangle = other.triangle;
        return *this;
    case GeometryType::TriangleMesh:
        triangle_mesh = other.triangle_mesh;
        return *this;
    default:
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return *this;
    }
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const BoxQueryOperator<T>& op)
    : type(GeometryType::Box)
    , box(op) {
    // Construct QueryOperator as "Box" variant.
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const CylinderQueryOperator<T>& op)
    : type(GeometryType::Cylinder)
    , cylinder(op) {
    // Construct QueryOperator as "Cylinder" variant.
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const PlaneQueryOperator<T>& op)
    : type(GeometryType::Plane)
    , plane(op) {
    // Construct QueryOperator as "Plane" variant.
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const SphereQueryOperator<T>& op)
    : type(GeometryType::Sphere)
    , sphere(op) {
    // Construct QueryOperator as "Sphere" variant.
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const TriangleQueryOperator<T>& op)
    : type(GeometryType::Triangle)
    , triangle(op) {
    // Construct QueryOperator as "Triangle" variant.
}

template <typename T>
ATLAS_HOST
QueryOperator<T>::QueryOperator(const TriangleMeshQueryOperator<T>& op)
    : type(GeometryType::TriangleMesh)
    , triangle_mesh(op) {
    // Construct QueryOperator as "TriangleMesh" variant.
}

template <typename T>
atlas::math::Vector<T, 3>
QueryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch to the active operator.
    switch (type) {
    case GeometryType::Box:
        return box.closest_point(p);
    case GeometryType::Cylinder:
        return cylinder.closest_point(p);
    case GeometryType::Plane:
        return plane.closest_point(p);
    case GeometryType::Sphere:
        return sphere.closest_point(p);
    case GeometryType::Triangle:
        return triangle.closest_point(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.closest_point(p);
    default:
        // Defensive fallback: return input.
        return p;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
QueryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch to the active operator.
    switch (type) {
    case GeometryType::Box:
        return box.closest_normal(p);
    case GeometryType::Cylinder:
        return cylinder.closest_normal(p);
    case GeometryType::Plane:
        return plane.closest_normal(p);
    case GeometryType::Sphere:
        return sphere.closest_normal(p);
    case GeometryType::Triangle:
        return triangle.closest_normal(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.closest_normal(p);
    default:
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
T
QueryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch signed-distance query.
    switch (type) {
    case GeometryType::Box:
        return box.signed_distance(p);
    case GeometryType::Cylinder:
        return cylinder.signed_distance(p);
    case GeometryType::Plane:
        return plane.signed_distance(p);
    case GeometryType::Sphere:
        return sphere.signed_distance(p);
    case GeometryType::Triangle:
        return triangle.signed_distance(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.signed_distance(p);
    default:
        return std::numeric_limits<T>::infinity();
    }
}

template <typename T>
bool
QueryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    switch (type) {
    case GeometryType::Box:
        return box.is_inside(p, tolerance);
    case GeometryType::Cylinder:
        return cylinder.is_inside(p, tolerance);
    case GeometryType::Plane:
        return plane.is_inside(p, tolerance);
    case GeometryType::Sphere:
        return sphere.is_inside(p, tolerance);
    case GeometryType::Triangle:
        return triangle.is_inside(p, tolerance);
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_inside(p, tolerance);
    default:
        return false;
    }
}

template <typename T>
bool
QueryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    switch (type) {
    case GeometryType::Box:
        return box.is_on_surface(p, tolerance);
    case GeometryType::Cylinder:
        return cylinder.is_on_surface(p, tolerance);
    case GeometryType::Plane:
        return plane.is_on_surface(p, tolerance);
    case GeometryType::Sphere:
        return sphere.is_on_surface(p, tolerance);
    case GeometryType::Triangle:
        return triangle.is_on_surface(p, tolerance);
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_on_surface(p, tolerance);
    default:
        return false;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
QueryOperator<T>::centroid() const noexcept {
    // Dispatch centroid query.
    switch (type) {
    case GeometryType::Box:
        return box.centroid();
    case GeometryType::Cylinder:
        return cylinder.centroid();
    case GeometryType::Plane:
        return plane.centroid();
    case GeometryType::Sphere:
        return sphere.centroid();
    case GeometryType::Triangle:
        return triangle.centroid();
    case GeometryType::TriangleMesh:
        return triangle_mesh.centroid();
    default:
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
QueryOperator<T>::bound() const noexcept {
    // Dispatch bounding-box query.
    switch (type) {
    case GeometryType::Box:
        return box.bound();
    case GeometryType::Cylinder:
        return cylinder.bound();
    case GeometryType::Plane:
        return plane.bound();
    case GeometryType::Sphere:
        return sphere.bound();
    case GeometryType::Triangle:
        return triangle.bound();
    case GeometryType::TriangleMesh:
        return triangle_mesh.bound();
    default:
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }
}

template <typename T>
bool
QueryOperator<T>::is_valid() const noexcept {
    // Dispatch validity query.
    switch (type) {
    case GeometryType::Box:
        return box.is_valid();
    case GeometryType::Cylinder:
        return cylinder.is_valid();
    case GeometryType::Plane:
        return plane.is_valid();
    case GeometryType::Sphere:
        return sphere.is_valid();
    case GeometryType::Triangle:
        return triangle.is_valid();
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_valid();
    default:
        return false;
    }
}

} // namespace atlas::geometry
