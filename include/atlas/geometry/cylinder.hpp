#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Cylinder<T>::Cylinder() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1))
    , height(T(1))
    , open(false) {
    // Initialize a default cylinder centered at the origin.
    //
    // Default geometric convention:
    // - axis   : aligned with the z-axis
    // - center : midpoint of the cylinder
    // - radius : 1
    // - height : 1
    //
    // After member initialization, bind the cached operator so all delegated
    // geometric queries reference this instance's storage.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_)
    , open(false) {
    // Construct the cylinder directly from caller-provided parameters.
    //
    // Geometric interpretation:
    // - `center_` is the midpoint of the finite cylinder
    // - `radius_` is the radial extent in the x-y plane
    // - `height_` is the full extent along the z-axis
    //
    // Bind the cached operator so it points at this object's member storage.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(const Cylinder& other) noexcept
    : center(other.center)
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
    // Copy geometric state from another cylinder.
    //
    // The cached operator must be rebound because it needs to reference this
    // object's members, not the source object's members.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(Cylinder&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
    // Move or copy the source cylinder's geometric state into this object.
    //
    // Rebind this object's operator so it points at the moved-in members.
    bind_operator();

    // Rebind the moved-from object's operator as well so its internal pointers
    // remain self-consistent after the move.
    other.bind_operator();
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(const Cylinder& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy all geometric parameters from the source cylinder.
    center = other.center;
    radius = other.radius;
    height = other.height;
    open   = other.open;

    // Rebind the cached operator because it must always reference this object's
    // current member storage.
    bind_operator();
    return *this;
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(Cylinder&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move or copy the source cylinder's data into this object.
    center = std::move(other.center);
    radius = other.radius;
    height = other.height;
    open   = other.open;

    // Rebind this object's cached operator.
    bind_operator();

    // Rebind the moved-from object's cached operator to keep its internal
    // pointers aligned with its own member storage.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Cylinder<T>::bind_operator() noexcept {
    // Bind the cached geometry operator to the actual member storage of this object.
    //
    // This keeps delegated queries cheap while ensuring the operator always sees
    // the current cylinder parameters.
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.radius = atlas::raw_pointer_cast(&radius);
    _operator.height = atlas::raw_pointer_cast(&height);
    _operator.open   = atlas::raw_pointer_cast(&open);
}

template <typename T>
typename Cylinder<T>::Builder
Cylinder<T>::builder() noexcept {
    // Return a fresh builder object for staged cylinder construction.
    //
    // The builder path is useful when center, radius, and height are supplied
    // incrementally before final validation and materialization.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Cylinder<T>::make_geometry_operator() const {
    // Wrap the cached cylinder-specific operator in the generic geometry-operator
    // interface expected by the rest of the geometry system.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-point query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-normal query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Cylinder<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the signed-distance query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Cylinder<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward inside classification to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Cylinder<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward surface-band classification to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::centroid() const noexcept {
    // Forward centroid computation to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Cylinder<T>::bound() const noexcept {
    // Forward bounding-box computation to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    // Forward validity testing to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Cylinder<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete geometry.
    return GeometryType::Cylinder;
}

template <typename T>
Cylinder<T>
Cylinder<T>::Builder::build() const {
    // Validate the staged builder parameters before constructing the cylinder.
    validate();

    // Start from a default-constructed cylinder so its cached operator is already
    // bound to its own member storage.
    Cylinder<T> c {};

    // Overwrite the default geometry with the validated builder state.
    c.center = _center;
    c.radius = _radius;
    c.height = _height;
    c.open   = _open;

    // No explicit rebind is required here because the cached operator already
    // points to `c`'s own members and only the stored values changed.
    return c;
}

template <typename T>
atlas::host_shared_ptr<Cylinder<T>>
Cylinder<T>::Builder::make_host_shared() const {
    // Build the cylinder by value first.
    auto c = build();

    // Move the built cylinder into host-shared managed storage.
    return atlas::make_host_shared<Cylinder<T>>(std::move(c));
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Store the cylinder center in the builder's staged state.
    _center = center_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_radius(T radius_) noexcept {
    // Store the cylinder radius in the builder's staged state.
    _radius = radius_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_height(T height_) noexcept {
    // Store the cylinder height in the builder's staged state.
    _height = height_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_open(const bool open_) noexcept {
    // Store whether the final cylinder should exclude top and bottom caps.
    _open = open_;
    return *this;
}

template <typename T>
void
Cylinder<T>::Builder::validate() const {
    // Reuse the runtime geometry-operator validity logic so the definition of
    // a valid cylinder stays centralized in one place.
    atlas::geometry::CylinderGeometryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);
    op.height = atlas::raw_pointer_cast(&_height);
    op.open   = atlas::raw_pointer_cast(&_open);

    // Reject invalid staged parameters with both a log message and an exception.
    if (!op.is_valid()) {
        throw std::runtime_error("Cylinder::Builder: invalid parameters.");
    }
}

template <typename T>
T
CylinderGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Return an infinite distance if the operator is not fully bound.
    if (!center || !radius || !height) return std::numeric_limits<T>::infinity();

    const bool is_open_cylinder = open && *open;

    // Half-height is used because the cylinder is centered at `*center`
    // and extends symmetrically along the z-axis.
    const T hz = (*height) * T(0.5);

    // Express the query point in cylinder-centered coordinates.
    const atlas::math::Vector<T, 3> d = p - *center;

    // Radial distance from the cylinder axis in the x-y plane.
    const T rho = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Signed offset from the side wall.
    // - negative or zero : radially inside
    // - positive         : radially outside
    const T qx = rho - *radius;

    // Signed offset from the top/bottom slab.
    const T qy = static_cast<T>(std::fabs(d.z)) - hz;

    if (is_open_cylinder) {
        // Open-ended cylinders use only the lateral wall as a signed boundary
        // while keeping the axial interval finite for the lateral surface.
        if (qy <= T(0)) {
            return qx;
        }

        return static_cast<T>(std::sqrt(qx * qx + qy * qy));
    }

    // Positive overflow outside the side wall.
    const T ax = (qx > T(0)) ? qx : T(0);

    // Positive overflow outside the top/bottom caps.
    const T ay = (qy > T(0)) ? qy : T(0);

    // Euclidean distance to the exterior region when the point is outside
    // in one or both dimensions.
    const T outside = static_cast<T>(std::sqrt(ax * ax + ay * ay));

    // Maximum of the signed side/cap offsets.
    // When both are negative, this gives the distance to the nearest boundary
    // from the interior.
    const T mxy = (qx > qy) ? qx : qy;

    // Interior contribution:
    // - negative when inside
    // - zero when outside
    const T inside = (mxy < T(0)) ? mxy : T(0);

    // Standard finite-cylinder SDF composition:
    // - outside distance when outside
    // - negative nearest-boundary distance when inside
    return outside + inside;
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // If the operator is unbound, containment cannot be established.
    if (!center || !radius || !height) return false;

    const bool is_open_cylinder = open && *open;

    // Precompute half-height and local point coordinates.
    const T hz                        = (*height) * T(0.5);
    const atlas::math::Vector<T, 3> d = p - *center;

    // Radial distance from cylinder axis.
    const T rho = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Signed radial and axial offsets as in the SDF formulation.
    const T qx = rho - *radius;
    const T qy = static_cast<T>(std::fabs(d.z)) - hz;

    if (is_open_cylinder) {
        // Open cylinders keep the finite z interval but ignore cap-distance
        // tolerance. Only the lateral wall participates in the boundary test.
        return qy <= T(0) && qx <= tolerance;
    }

    if (qx <= T(0) && qy <= T(0)) {
        // The point lies inside the finite cylinder bounds.
        //
        // `inside` is the negative distance to the nearest boundary.
        const T inside = (qx > qy) ? qx : qy;
        return inside <= tolerance;
    }

    // For exterior points, a negative tolerance cannot admit outside points.
    if (tolerance < T(0)) return false;

    // Measure squared exterior overflow relative to side/cap bounds.
    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    // Exterior points are accepted only if their distance to the cylinder
    // does not exceed the tolerance.
    return (ax * ax + ay * ay) <= (tolerance * tolerance);
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (center && radius && height && open && *open) {
        const T hz                        = (*height) * T(0.5);
        const atlas::math::Vector<T, 3> d = p - *center;
        const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));
        const T zmin                      = (*center).z - hz;
        const T zmax                      = (*center).z + hz;

        return std::abs(rho - *radius) <= tolerance
            && p.z >= zmin - tolerance
            && p.z <= zmax + tolerance;
    }

    // Classify a point as on the surface when its absolute signed distance
    // falls within the specified tolerance band around zero.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is unbound, return the input point unchanged.
    if (!center || !radius || !height) return p;

    // Precompute top/bottom z coordinates from the center and half-height.
    const T hz   = (*height) * T(0.5);
    const T zmin = (*center).z - hz;
    const T zmax = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    // Local coordinates relative to the cylinder center.
    const atlas::math::Vector<T, 3> d = p - *center;

    // Radial distance in the x-y plane.
    const T rho = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Clamp the z coordinate into the cylinder's axial interval.
    const T zc = (p.z < zmin) ? zmin
        : (p.z > zmax)        ? zmax
                              : p.z;

    // Start from the original x-y position and project radially only if needed.
    T sx = p.x;
    T sy = p.y;

    if (rho > *radius) {
        // The point lies outside the side wall in the radial direction.
        // Project its x-y position onto the circular side boundary.
        const T inv = T(1) / rho;
        sx          = (*center).x + d.x * ((*radius) * inv);
        sy          = (*center).y + d.y * ((*radius) * inv);
    }

    // Candidate closest point formed by radial clamp + axial clamp.
    atlas::math::Vector<T, 3> cp(sx, sy, zc);

    // Determine whether the point lies within the cylinder volume.
    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {
        // The nearest point on an open-ended cylinder always lies on the
        // lateral wall, with z clamped into the finite axial interval.
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
        // Interior point handling:
        // choose the nearest surface feature among
        // - side wall
        // - bottom cap
        // - top cap
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            // Nearest feature is the curved side wall.
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                cp.x        = (*center).x + d.x * ((*radius) * inv);
                cp.y        = (*center).y + d.y * ((*radius) * inv);
            } else {
                // At the exact axis center, the radial direction is undefined.
                // Choose a deterministic point on the side wall along +x.
                cp.x = (*center).x + (*radius);
                cp.y = (*center).y;
            }

            // Preserve z because the closest side-wall point shares the same z.
            cp.z = p.z;
        } else if (d_to_bot <= d_to_top) {
            // Nearest feature is the bottom cap.
            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmin;
        } else {
            // Nearest feature is the top cap.
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
    // If the operator is unbound, return the zero vector.
    if (!center || !radius || !height) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // Precompute top/bottom z coordinates.
    const T hz   = (*height) * T(0.5);
    const T zmin = (*center).z - hz;
    const T zmax = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    // Local coordinates and radial distance.
    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Determine whether the point lies inside the finite cylinder volume.
    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {
        const atlas::math::Vector<T, 3> cp = closest_point(p);
        const atlas::math::Vector<T, 3> cd = cp - *center;
        const T rr2                        = cd.x * cd.x + cd.y * cd.y;

        if (rr2 > T(0)) {
            const T inv = T(1) / static_cast<T>(std::sqrt(rr2));
            return atlas::math::Vector<T, 3>(cd.x * inv, cd.y * inv, T(0));
        }

        return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    }

    if (inside_radial && inside_z) {
        // Interior-point normal:
        // choose the outward normal of the nearest surface feature.
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            // Nearest feature is the side wall.
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                return atlas::math::Vector<T, 3>(d.x * inv, d.y * inv, T(0));
            }

            // At the axis center, radial direction is undefined.
            // Choose a deterministic outward normal along +x.
            return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
        }

        // Otherwise the nearest feature is one of the caps.
        return (d_to_bot <= d_to_top)
            ? atlas::math::Vector<T, 3>(T(0), T(0), -T(1))
            : atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    // Exterior-point normal:
    // first find the closest point on the cylinder.
    const atlas::math::Vector<T, 3> cp = closest_point(p);

    // Use epsilon to robustly detect whether the closest point lies on a cap.
    const T e = std::numeric_limits<T>::epsilon();

    if (static_cast<T>(std::fabs(cp.z - zmin)) <= e) return atlas::math::Vector<T, 3>(T(0), T(0), -T(1));
    if (static_cast<T>(std::fabs(cp.z - zmax)) <= e) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    // Otherwise treat the closest point as lying on the side wall.
    const atlas::math::Vector<T, 3> cd = cp - *center;
    const T rr2                        = cd.x * cd.x + cd.y * cd.y;

    if (rr2 > T(0)) {
        // Normalize the radial component to obtain the outward side-wall normal.
        const T inv = T(1) / static_cast<T>(std::sqrt(rr2));
        return atlas::math::Vector<T, 3>(cd.x * inv, cd.y * inv, T(0));
    }

    // Degenerate fallback if radial direction is numerically undefined.
    return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderGeometryOperator<T>::centroid() const noexcept {
    // If the center pointer is missing, return the origin as a safe fallback.
    if (!center) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // The centroid of a uniform finite cylinder is its center point.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CylinderGeometryOperator<T>::bound() const noexcept {
    // If the operator is not fully bound, return a default-constructed AABB.
    if (!center || !radius || !height) return atlas::spatial::AxisAlignedBoundingBox<T>();

    // The cylinder is assumed to be axis-aligned along z.
    // Therefore its AABB is simply the radial extent in x/y and half-height in z.
    const T hz = (*height) * T(0.5);
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>((*center).x - *radius, (*center).y - *radius, (*center).z - hz),
        atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y + *radius, (*center).z + hz));
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_valid() const noexcept {
    // Radius and height pointers must both be present.
    if (!radius || !height) return false;

    // A valid finite cylinder requires strictly positive radius and height.
    return (*radius) > T(0) && (*height) > T(0);
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Initialize the result to the default "no hit" state.
    HitSurface<T> out {};
    if (!center || !radius || !height) return out;

    // Transform the ray origin into cylinder-local coordinates where the cylinder
    // is centered at the origin and aligned with the z-axis.
    const atlas::math::Vector<T, 3> ro = ray.origin - *center;

    // Ray direction in the same coordinate frame.
    const atlas::math::Vector<T, 3> rd = ray.direction;

    // Local cylinder parameters.
    const T r    = *radius;
    const T hz   = (*height) * T(0.5);
    const T zmin = -hz;
    const T zmax = hz;
    const bool is_open_cylinder = open && *open;

    // Track the nearest valid hit found so far.
    T best_t = std::numeric_limits<T>::infinity();

    // Store the corresponding outward surface normal in local/world-aligned coordinates.
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(0));

    // Track whether the ray can overlap the z slab of the finite cylinder.
    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();

    if (rd.z == T(0)) {
        // The ray is parallel to the caps.
        // It can only intersect the finite cylinder if its z coordinate already
        // lies within the cylinder's axial interval.
        if (ro.z < zmin || ro.z > zmax) z_ok = false;
    } else {
        // Intersect the ray with the z slab [zmin, zmax].
        // This yields the parametric interval where the ray lies between the caps.
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
        // Solve ray vs. infinite cylinder side wall in x-y:
        //   x^2 + y^2 = r^2
        //
        // This produces a quadratic in ray parameter t.
        const T A = rd.x * rd.x + rd.y * rd.y;
        const T B = T(2) * (ro.x * rd.x + ro.y * rd.y);
        const T C = ro.x * ro.x + ro.y * ro.y - r * r;

        if (A > T(0)) {
            // Non-degenerate radial direction: solve the quadratic.
            const T disc = B * B - T(4) * A * C;
            if (disc >= T(0)) {
                const T sqrt_disc = static_cast<T>(std::sqrt(disc));

                // Numerically stable quadratic-root formulation.
                const T sign_b = (B >= T(0)) ? T(1) : T(-1);
                const T q      = -T(0.5) * (B + sign_b * sqrt_disc);

                T t0 = (q == T(0)) ? std::numeric_limits<T>::infinity() : (C / q);
                T t1 = (A == T(0)) ? std::numeric_limits<T>::infinity() : (q / A);
                if (t0 > t1) {
                    const T tmp = t0;
                    t0          = t1;
                    t1          = tmp;
                }

                // Accept a side hit only if:
                // - it is in front of the ray origin, and
                // - its z coordinate lies within the finite-cylinder slab interval.
                auto accept_side = [&](const T t) -> bool {
                    if (!(t >= T(0))) return false;
                    if (rd.z != T(0) && (t < t_z_enter || t > t_z_exit)) return false;
                    return true;
                };

                // Record a valid side-wall hit and compute its outward normal.
                auto set_side_hit = [&](const T t) {
                    best_t                             = t;
                    const atlas::math::Vector<T, 3> ph = ro + rd * t;
                    const T rr2                        = ph.x * ph.x + ph.y * ph.y;
                    if (rr2 > T(0)) {
                        const T inv_rr = T(1) / static_cast<T>(std::sqrt(rr2));
                        best_n         = atlas::math::Vector<T, 3>(ph.x * inv_rr, ph.y * inv_rr, T(0));
                    } else {
                        // Degenerate fallback if the side-wall normal cannot be
                        // resolved radially.
                        best_n = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
                    }
                };

                // Prefer the nearer valid side intersection.
                if (accept_side(t0)) set_side_hit(t0);
                if (!std::isfinite(best_t) && accept_side(t1)) set_side_hit(t1);
            }
        }
    }

    if (!is_open_cylinder && rd.z != T(0)) {
        // Also test intersections against the bottom and top caps.
        auto try_cap = [&](const T zplane, const T nz) {
            const T t = (zplane - ro.z) / rd.z;

            // Ignore hits behind the origin or farther than a previously found hit.
            if (!(t >= T(0)) || t >= best_t) return;

            // Compute hit point in local cylinder coordinates.
            const atlas::math::Vector<T, 3> ph = ro + rd * t;

            // Accept only if the hit lies inside the cap disk.
            if (ph.x * ph.x + ph.y * ph.y <= r * r) {
                best_t = t;
                best_n = atlas::math::Vector<T, 3>(T(0), T(0), nz);
            }
        };

        // Evaluate caps in near-to-far order for slightly cleaner early selection.
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

    // If no finite hit was found, return the default "no hit" result.
    if (!std::isfinite(best_t)) return out;

    // Populate the final hit record.
    out.is_intersecting = true;
    out.distance        = best_t;
    out.point           = ray.point_at(best_t);
    out.normal          = best_n;
    return out;
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Function-call convenience wrapper around the explicit ray-trace routine.
    return trace(ray);
}

} // namespace atlas::geometry
