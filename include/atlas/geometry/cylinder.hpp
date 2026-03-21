#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* Cylinder<T>                                                             */
/* ====================================================================== */

template <typename T>
Cylinder<T>::Cylinder() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1))
    , height(T(1)) {
    // Default constructor creates a canonical cylinder:
    // - center at origin
    // - radius = 1
    // - height = 1
    //
    // Coordinate convention (as used by the trace/query operators):
    // - Cylinder is axis-aligned along Z in its local space.
    // - Its caps lie at z = center.z ± height/2.
    //
    // Rationale:
    // - Provides a non-degenerate, immediately usable primitive.
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_) {
    // Construct cylinder directly from parameters.
    //
    // Caller responsibility:
    // - This constructor does NOT validate radius/height.
    // - If radius <= 0 or height <= 0, the cylinder becomes degenerate/invalid.
    //
    // If you want enforced validity, use:
    //   Cylinder<T>::builder().with_center(...).with_radius(...).with_height(...).build()
}

template <typename T>
typename Cylinder<T>::Builder
Cylinder<T>::builder() noexcept {
    // Builder entry point.
    //
    // Why a builder?
    // - Enables validation (radius/height > 0, finite values, etc.) before constructing.
    // - Keeps call sites readable when parameters are set in multiple steps.
    return Builder {};
}


template <typename T>
GeometryOperator<T>
Cylinder<T>::make_geometry_operator() const {
    // Construct a GeometryOperator for closest-point/normal/distance queries.
    //
    // Same pointer/lifetime considerations as make_geometry_operator().
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return GeometryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute closest point on (or inside) the cylinder to query point p.
    //
    // Delegation pattern:
    // - Create a temporary query operator wired to this cylinder's parameters.
    // - Reuse the operator's implementation to keep behavior consistent across APIs.
    //
    // Cost:
    // - The operator is tiny (just pointers), so this is cheap.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute outward normal of the closest surface feature to p.
    //
    // Expected behavior (typical):
    // - Outside near side wall: normal points radially outward (x,y,0) normalized.
    // - Outside near cap: normal is (0,0,±1).
    // - Inside: usually normal of nearest surface (tie-breaks on edges).
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.closest_normal(p);
}

template <typename T>
T
Cylinder<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to the cylinder (SDF).
    //
    // Typical convention:
    // - negative inside
    // - zero on the surface
    // - positive outside
    //
    // Delegates to CylinderGeometryOperator<T> for a single source of truth.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.signed_distance(p);
}

template <typename T>
bool
Cylinder<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Keep finite-cylinder interior classification centralized in the query
    // operator so tolerance handling is shared across all call paths.
    return make_geometry_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Cylinder<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-band checks to the query operator to avoid duplicating
    // cap/side-wall boundary logic in the geometry wrapper.
    return make_geometry_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::centroid() const noexcept {
    // Cylinder centroid in this representation is its center.
    //
    // Delegation ensures consistent definition with query operator.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Cylinder<T>::bound() const noexcept {
    // Axis-aligned bounding box of this cylinder.
    //
    // Since the cylinder is axis-aligned along Z:
    // - x,y extents are center ± radius
    // - z extent is center.z ± height/2
    //
    // Delegation keeps bounding behavior consistent with other primitives.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.bound();
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    // Validate cylinder parameters.
    //
    // Typical validity rules:
    // - radius > 0
    // - height > 0
    // - values are finite
    //
    // Exact rules are defined in CylinderGeometryOperator<T>::is_valid().
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.is_valid();
}

template <typename T>
GeometryType
Cylinder<T>::type() const noexcept {
    // Return the geometry type tag for this class.
    return GeometryType::Cylinder;
}

/* ====================================================================== */
/* Cylinder<T>::Builder                                                    */
/* ====================================================================== */

template <typename T>
Cylinder<T>
Cylinder<T>::Builder::build() const {
    // Build a Cylinder<T> after validation.
    //
    // Strong exception guarantee:
    // - If validate() throws, no Cylinder is produced.
    validate();

    Cylinder<T> c {};

    // Copy validated parameters into the final cylinder object.
    c.center = _center;
    c.radius = _radius;
    c.height = _height;

    return c;
}

template <typename T>
atlas::host_shared_ptr<Cylinder<T>>
Cylinder<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build by value
    // - Move into a shared, heap-allocated cylinder object
    auto c = build();
    return atlas::make_host_shared<Cylinder<T>>(std::move(c));
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Set center of the cylinder.
    //
    // The cylinder axis is assumed Z-aligned; center defines mid-point along Z.
    _center = center_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_radius(T radius_) noexcept {
    // Set cylinder radius in XY plane.
    //
    // Valid cylinders require radius > 0 (enforced in validate()).
    _radius = radius_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_height(T height_) noexcept {
    // Set cylinder height along Z.
    //
    // Valid cylinders require height > 0 (enforced in validate()).
    _height = height_;
    return *this;
}

template <typename T>
void
Cylinder<T>::Builder::validate() const {
    // Validate builder parameters prior to building Cylinder<T>.
    //
    // Rule set is delegated to CylinderGeometryOperator<T>::is_valid():
    // - This keeps validity rules consistent across the codebase.
    //
    // Implementation:
    // - Create a temporary query operator pointing to builder-owned fields.
    // - Safe because validate() uses them immediately.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);
    op.height = atlas::raw_pointer_cast(&_height);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Cylinder::Builder validation failed: radius and height must be > 0, and values must be finite.";
        throw std::runtime_error("Cylinder::Builder: invalid parameters.");
    }
}

/* CylinderGeometryOperator<T>                                                */
/* ====================================================================== */

template <typename T>
T
CylinderGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
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
CylinderGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
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
CylinderGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
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
CylinderGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
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
CylinderGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
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
CylinderGeometryOperator<T>::centroid() const noexcept {
    // Centroid of cylinder is its center.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CylinderGeometryOperator<T>::bound() const noexcept {
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
CylinderGeometryOperator<T>::is_valid() const noexcept {
    // Valid if:
    // - radius/height pointers exist
    // - parameters are non-negative
    //
    // Note:
    // - Some systems require strictly positive radius/height;
    if (!radius || !height) return false;
    return (*radius) > T(0) && (*height) > T(0);
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> out {};
    if (!center || !radius || !height) return out;

    const atlas::math::Vector<T, 3> ro = ray.origin - *center;
    const atlas::math::Vector<T, 3> rd = ray.direction;
    const T r    = *radius;
    const T hz   = (*height) * T(0.5);
    const T zmin = -hz;
    const T zmax = hz;

    T best_t = std::numeric_limits<T>::infinity();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(0));

    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();

    if (rd.z == T(0)) {
        if (ro.z < zmin || ro.z > zmax) z_ok = false;
    } else {
        const T inv_dz = T(1) / rd.z;
        T a = (zmin - ro.z) * inv_dz;
        T b = (zmax - ro.z) * inv_dz;
        if (a > b) {
            const T tmp = a;
            a = b;
            b = tmp;
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
                const T sign_b    = (B >= T(0)) ? T(1) : T(-1);
                const T q         = -T(0.5) * (B + sign_b * sqrt_disc);

                T t0 = (q == T(0)) ? std::numeric_limits<T>::infinity() : (C / q);
                T t1 = (A == T(0)) ? std::numeric_limits<T>::infinity() : (q / A);
                if (t0 > t1) {
                    const T tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }

                auto accept_side = [&](const T t) -> bool {
                    if (!(t >= T(0))) return false;
                    if (rd.z != T(0) && (t < t_z_enter || t > t_z_exit)) return false;
                    return true;
                };

                auto set_side_hit = [&](const T t) {
                    best_t = t;
                    const atlas::math::Vector<T, 3> ph = ro + rd * t;
                    const T rr2 = ph.x * ph.x + ph.y * ph.y;
                    if (rr2 > T(0)) {
                        const T inv_rr = T(1) / static_cast<T>(std::sqrt(rr2));
                        best_n         = atlas::math::Vector<T, 3>(ph.x * inv_rr, ph.y * inv_rr, T(0));
                    } else {
                        best_n = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
                    }
                };

                if (accept_side(t0)) set_side_hit(t0);
                if (!std::isfinite(best_t) && accept_side(t1)) set_side_hit(t1);
            }
        }
    }

    if (rd.z != T(0)) {
        auto try_cap = [&](const T zplane, const T nz) {
            const T t = (zplane - ro.z) / rd.z;
            if (!(t >= T(0)) || t >= best_t) return;
            const atlas::math::Vector<T, 3> ph = ro + rd * t;
            if (ph.x * ph.x + ph.y * ph.y <= r * r) {
                best_t = t;
                best_n = atlas::math::Vector<T, 3>(T(0), T(0), nz);
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

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

} // namespace atlas::geometry
