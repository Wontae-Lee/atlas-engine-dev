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
    // Bind the operator to this object's default cylinder parameter storage.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_)
    , open(false) {
    // Bind the operator after storing the user-provided cylinder parameters.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(const Cylinder& other) noexcept
    : center(other.center)
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
    // Rebind the operator because copied raw pointers must refer to this object.
    bind_operator();
}

template <typename T>
Cylinder<T>::Cylinder(Cylinder&& other) noexcept
    : center(std::move(other.center))
    , radius(other.radius)
    , height(other.height)
    , open(other.open) {
    // Rebind this object after moving member storage.
    bind_operator();

    // Keep the moved-from object internally consistent.
    other.bind_operator();
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(const Cylinder& other) noexcept {
    // Avoid unnecessary work and preserve pointer bindings on self-assignment.
    if (this == &other) {
        return *this;
    }

    center = other.center;
    radius = other.radius;
    height = other.height;
    open   = other.open;

    // Rebind after assignment because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
Cylinder<T>&
Cylinder<T>::operator=(Cylinder&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    center = std::move(other.center);
    radius = other.radius;
    height = other.height;
    open   = other.open;

    // Rebind both objects so each operator points to its own member storage.
    bind_operator();
    other.bind_operator();

    return *this;
}

template <typename T>
void
Cylinder<T>::bind_operator() noexcept {
    // Store non-owning raw pointers to the cylinder parameters used by the operator.
    _operator.center = atlas::raw_pointer_cast(&center);
    _operator.radius = atlas::raw_pointer_cast(&radius);
    _operator.height = atlas::raw_pointer_cast(&height);
    _operator.open   = atlas::raw_pointer_cast(&open);
}

template <typename T>
typename Cylinder<T>::Builder
Cylinder<T>::builder() noexcept {
    // Return a fresh builder for fluent cylinder construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Cylinder<T>::make_geometry_operator() const {
    // Wrap the concrete cylinder operator in the generic geometry operator type.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound cylinder operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound cylinder operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Cylinder<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound cylinder operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Cylinder<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate containment checks to the bound cylinder operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Cylinder<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-membership checks to the bound cylinder operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::centroid() const noexcept {
    // Delegate centroid computation to the bound cylinder operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Cylinder<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound cylinder operator.
    return _operator.bound();
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound cylinder operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Cylinder<T>::type() const noexcept {
    // Identify this geometry as a cylinder.
    return GeometryType::Cylinder;
}

template <typename T>
Cylinder<T>
Cylinder<T>::Builder::build() const {
    // Validate all builder parameters before constructing the final cylinder.
    validate();

    Cylinder<T> c {};
    c.center = _center;
    c.radius = _radius;
    c.height = _height;
    c.open   = _open;

    // Rebind because parameters are assigned after default construction.
    c.bind_operator();

    return c;
}

template <typename T>
atlas::host_shared_ptr<Cylinder<T>>
Cylinder<T>::Builder::make_host_shared() const {
    // Build a validated cylinder before placing it into host-shared ownership.
    auto c = build();
    return atlas::make_host_shared<Cylinder<T>>(std::move(c));
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Store the requested center for the later build() call.
    _center = center_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_radius(T radius_) noexcept {
    // Store the requested radius; validate() enforces positivity later.
    _radius = radius_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_height(T height_) noexcept {
    // Store the requested height; validate() enforces positivity later.
    _height = height_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_open(const bool open_) noexcept {
    // Store whether the cylinder should omit its end caps.
    _open = open_;
    return *this;
}

template <typename T>
void
Cylinder<T>::Builder::validate() const {
    atlas::geometry::CylinderGeometryOperator<T> op;

    // Validate through the same operator logic used by constructed cylinders.
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
CylinderGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!center || !radius || !height) {
        return std::numeric_limits<T>::infinity();
    }

    const bool is_open_cylinder       = open && *open;
    const T hz                        = (*height) * T(0.5);
    const atlas::math::Vector<T, 3> d = p - *center;

    // Compute radial distance from the cylinder axis.
    const T rho = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // qx is radial signed excess; qy is axial signed excess.
    const T qx = rho - *radius;
    const T qy = static_cast<T>(std::fabs(d.z)) - hz;

    if (is_open_cylinder) {
        // Inside the finite axial range, an open cylinder only measures distance to the side wall.
        if (qy <= T(0)) {
            return qx;
        }

        // Outside the axial range, measure distance to the nearest rim curve.
        return static_cast<T>(std::sqrt(qx * qx + qy * qy));
    }

    // Positive components represent the outside distance from the capped cylinder.
    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    // Euclidean distance to the closest exterior feature.
    const T outside = static_cast<T>(std::sqrt(ax * ax + ay * ay));

    // Interior distance is controlled by the larger signed constraint value.
    const T mxy    = (qx > qy) ? qx : qy;
    const T inside = (mxy < T(0)) ? mxy : T(0);

    return outside + inside;
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!center || !radius || !height) {
        return false;
    }

    const bool is_open_cylinder       = open && *open;
    const T hz                        = (*height) * T(0.5);
    const atlas::math::Vector<T, 3> d = p - *center;

    // Convert the query point to radial and axial signed offsets.
    const T rho = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));
    const T qx  = rho - *radius;
    const T qy  = static_cast<T>(std::fabs(d.z)) - hz;

    if (is_open_cylinder) {
        // Open cylinders ignore cap interiors and only require finite height plus radial tolerance.
        return qy <= T(0) && qx <= tolerance;
    }

    if (qx <= T(0) && qy <= T(0)) {
        // For interior points, the closest boundary determines tolerance acceptance.
        const T inside = (qx > qy) ? qx : qy;
        return inside <= tolerance;
    }

    // Negative tolerance cannot expand the outside acceptance region.
    if (tolerance < T(0)) {
        return false;
    }

    // Compare squared outside distance against squared tolerance.
    const T ax = (qx > T(0)) ? qx : T(0);
    const T ay = (qy > T(0)) ? qy : T(0);

    return (ax * ax + ay * ay) <= (tolerance * tolerance);
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (center && radius && height && open && *open) {
        const T hz                        = (*height) * T(0.5);
        const atlas::math::Vector<T, 3> d = p - *center;

        // Open cylinders expose only the lateral surface.
        const T rho  = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));
        const T zmin = (*center).z - hz;
        const T zmax = (*center).z + hz;

        return std::abs(rho - *radius) <= tolerance
            && p.z >= zmin - tolerance
            && p.z <= zmax + tolerance;
    }

    // Closed cylinders can use the full signed-distance surface test.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without valid geometry data, return the input unchanged.
    if (!center || !radius || !height) {
        return p;
    }

    const T hz                  = (*height) * T(0.5);
    const T zmin                = (*center).z - hz;
    const T zmax                = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    // Clamp z to the finite cylinder height.
    const T zc = (p.z < zmin) ? zmin
        : (p.z > zmax)        ? zmax
                              : p.z;

    T sx = p.x;
    T sy = p.y;

    if (rho > *radius) {
        // Project exterior radial points onto the cylindrical side wall.
        const T inv = T(1) / rho;
        sx          = (*center).x + d.x * ((*radius) * inv);
        sy          = (*center).y + d.y * ((*radius) * inv);
    }

    atlas::math::Vector<T, 3> cp(sx, sy, zc);

    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {
        if (rho > T(0)) {
            // Project to the lateral surface using the radial direction.
            const T inv = T(1) / rho;
            cp.x        = (*center).x + d.x * ((*radius) * inv);
            cp.y        = (*center).y + d.y * ((*radius) * inv);
        } else {
            // Choose a deterministic side-wall point when the query lies on the cylinder axis.
            cp.x = (*center).x + (*radius);
            cp.y = (*center).y;
        }

        // Open cylinders keep the closest point on the finite lateral sheet.
        cp.z = zc;
        return cp;
    }

    if (inside_radial && inside_z) {
        // For interior points, project to the nearest cylinder boundary feature.
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            if (rho > T(0)) {
                // Project radially onto the side wall.
                const T inv = T(1) / rho;
                cp.x        = (*center).x + d.x * ((*radius) * inv);
                cp.y        = (*center).y + d.y * ((*radius) * inv);
            } else {
                // Choose a deterministic side-wall point for axis-aligned interior queries.
                cp.x = (*center).x + (*radius);
                cp.y = (*center).y;
            }

            cp.z = p.z;
        } else if (d_to_bot <= d_to_top) {
            // Project to the lower cap.
            cp.x = p.x;
            cp.y = p.y;
            cp.z = zmin;
        } else {
            // Project to the upper cap.
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
    // Invalid geometry cannot provide a meaningful surface normal.
    if (!center || !radius || !height) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    const T hz                  = (*height) * T(0.5);
    const T zmin                = (*center).z - hz;
    const T zmax                = (*center).z + hz;
    const bool is_open_cylinder = open && *open;

    const atlas::math::Vector<T, 3> d = p - *center;
    const T rho                       = static_cast<T>(std::sqrt(d.x * d.x + d.y * d.y));

    const bool inside_radial = (rho <= *radius);
    const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

    if (is_open_cylinder) {
        // Open cylinders always return a lateral-surface normal.
        const atlas::math::Vector<T, 3> cp = closest_point(p);
        const atlas::math::Vector<T, 3> cd = cp - *center;
        const T rr2                        = cd.x * cd.x + cd.y * cd.y;

        if (rr2 > T(0)) {
            const T inv = T(1) / static_cast<T>(std::sqrt(rr2));
            return atlas::math::Vector<T, 3>(cd.x * inv, cd.y * inv, T(0));
        }

        // Deterministic fallback for degenerate radial direction.
        return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
    }

    if (inside_radial && inside_z) {
        // Interior normals point toward the nearest cylinder boundary feature.
        const T d_to_side = (*radius) - rho;
        const T d_to_bot  = p.z - zmin;
        const T d_to_top  = zmax - p.z;

        if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
            if (rho > T(0)) {
                const T inv = T(1) / rho;
                return atlas::math::Vector<T, 3>(d.x * inv, d.y * inv, T(0));
            }

            // Deterministic fallback normal on the cylinder axis.
            return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
        }

        // Choose the nearest cap normal.
        return (d_to_bot <= d_to_top)
            ? atlas::math::Vector<T, 3>(T(0), T(0), -T(1))
            : atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    // Exterior normals are determined by the closest surface point.
    const atlas::math::Vector<T, 3> cp = closest_point(p);
    const T e                          = std::numeric_limits<T>::epsilon();

    if (static_cast<T>(std::fabs(cp.z - zmin)) <= e) {
        return atlas::math::Vector<T, 3>(T(0), T(0), -T(1));
    }

    if (static_cast<T>(std::fabs(cp.z - zmax)) <= e) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    const atlas::math::Vector<T, 3> cd = cp - *center;
    const T rr2                        = cd.x * cd.x + cd.y * cd.y;

    if (rr2 > T(0)) {
        const T inv = T(1) / static_cast<T>(std::sqrt(rr2));
        return atlas::math::Vector<T, 3>(cd.x * inv, cd.y * inv, T(0));
    }

    // Deterministic fallback when the closest radial direction is undefined.
    return atlas::math::Vector<T, 3>(T(1), T(0), T(0));
}

template <typename T>
atlas::math::Vector<T, 3>
CylinderGeometryOperator<T>::centroid() const noexcept {
    // Missing center falls back to the origin as a neutral centroid.
    if (!center) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // A cylinder's centroid coincides with its center.
    return *center;
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
CylinderGeometryOperator<T>::bound() const noexcept {
    // Invalid geometry returns an empty/default bounding box.
    if (!center || !radius || !height) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    const T hz = (*height) * T(0.5);

    // The axis-aligned bound expands by radius in x/y and half-height in z.
    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>((*center).x - *radius, (*center).y - *radius, (*center).z - hz),
        atlas::math::Vector<T, 3>((*center).x + *radius, (*center).y + *radius, (*center).z + hz));
}

template <typename T>
bool
CylinderGeometryOperator<T>::is_valid() const noexcept {
    // Radius and height pointers must exist before validation can succeed.
    if (!radius || !height) {
        return false;
    }

    // A valid cylinder requires strictly positive radius and height.
    return (*radius) > T(0) && (*height) > T(0);
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> out {};

    // Invalid geometry produces a default non-intersecting hit result.
    if (!center || !radius || !height) {
        return out;
    }

    // Transform the ray origin into the cylinder-centered local frame.
    const atlas::math::Vector<T, 3> ro = ray.origin - *center;
    const atlas::math::Vector<T, 3> rd = ray.direction;

    const T r                   = *radius;
    const T hz                  = (*height) * T(0.5);
    const T zmin                = -hz;
    const T zmax                = hz;
    const bool is_open_cylinder = open && *open;

    // Track the closest accepted hit candidate.
    T best_t = std::numeric_limits<T>::infinity();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(0));

    // Compute the ray interval that overlaps the finite cylinder height.
    bool z_ok   = true;
    T t_z_enter = -std::numeric_limits<T>::infinity();
    T t_z_exit  = std::numeric_limits<T>::infinity();

    if (rd.z == T(0)) {
        // A ray parallel to the cap planes can only hit the side if already inside the z slab.
        if (ro.z < zmin || ro.z > zmax) {
            z_ok = false;
        }
    } else {
        const T inv_dz = T(1) / rd.z;
        T a            = (zmin - ro.z) * inv_dz;
        T b            = (zmax - ro.z) * inv_dz;

        if (a > b) {
            // Sort slab interval endpoints.
            const T tmp = a;
            a           = b;
            b           = tmp;
        }

        t_z_enter = a;
        t_z_exit  = b;
    }

    if (z_ok) {
        // Solve the quadratic equation for intersection with the infinite cylinder side wall.
        const T A = rd.x * rd.x + rd.y * rd.y;
        const T B = T(2) * (ro.x * rd.x + ro.y * rd.y);
        const T C = ro.x * ro.x + ro.y * ro.y - r * r;

        if (A > T(0)) {
            const T disc = B * B - T(4) * A * C;

            if (disc >= T(0)) {
                const T sqrt_disc = static_cast<T>(std::sqrt(disc));

                // Use the numerically stable quadratic form.
                const T sign_b = (B >= T(0)) ? T(1) : T(-1);
                const T q      = -T(0.5) * (B + sign_b * sqrt_disc);

                T t0 = (q == T(0)) ? std::numeric_limits<T>::infinity() : (C / q);
                T t1 = (A == T(0)) ? std::numeric_limits<T>::infinity() : (q / A);

                if (t0 > t1) {
                    // Sort roots so the nearer candidate is tested first.
                    const T tmp = t0;
                    t0          = t1;
                    t1          = tmp;
                }

                auto accept_side = [&](const T t) -> bool {
                    // Ignore intersections behind the ray origin.
                    if (!(t >= T(0))) {
                        return false;
                    }

                    // The side hit must also lie within the finite cylinder height.
                    if (rd.z != T(0) && (t < t_z_enter || t > t_z_exit)) {
                        return false;
                    }

                    return true;
                };

                auto set_side_hit = [&](const T t) {
                    best_t = t;

                    // Compute the local hit point to derive the outward radial normal.
                    const atlas::math::Vector<T, 3> ph = ro + rd * t;
                    const T rr2                        = ph.x * ph.x + ph.y * ph.y;

                    if (rr2 > T(0)) {
                        const T inv_rr = T(1) / static_cast<T>(std::sqrt(rr2));
                        best_n         = atlas::math::Vector<T, 3>(ph.x * inv_rr, ph.y * inv_rr, T(0));
                    } else {
                        // Deterministic fallback for undefined radial normal.
                        best_n = atlas::math::Vector<T, 3>(T(1), T(0), T(0));
                    }
                };

                if (accept_side(t0)) {
                    set_side_hit(t0);
                }

                if (!std::isfinite(best_t) && accept_side(t1)) {
                    set_side_hit(t1);
                }
            }
        }
    }

    if (!is_open_cylinder && rd.z != T(0)) {
        auto try_cap = [&](const T zplane, const T nz) {
            // Intersect the ray with one cap plane.
            const T t = (zplane - ro.z) / rd.z;

            // Ignore behind-ray hits and cap hits farther than the current best hit.
            if (!(t >= T(0)) || t >= best_t) {
                return;
            }

            const atlas::math::Vector<T, 3> ph = ro + rd * t;

            if (ph.x * ph.x + ph.y * ph.y <= r * r) {
                // Accept the cap hit only when the plane hit lies inside the cap disk.
                best_t = t;
                best_n = atlas::math::Vector<T, 3>(T(0), T(0), nz);
            }
        };

        const T t_min = (zmin - ro.z) / rd.z;
        const T t_max = (zmax - ro.z) / rd.z;

        if (t_min < t_max) {
            // Test the nearer cap first.
            try_cap(zmin, -T(1));
            try_cap(zmax, T(1));
        } else {
            // Reverse order when the ray approaches from the opposite z direction.
            try_cap(zmax, T(1));
            try_cap(zmin, -T(1));
        }
    }

    // Return a miss when no valid side or cap intersection was found.
    if (!std::isfinite(best_t)) {
        return out;
    }

    // Populate the hit record with the closest valid intersection.
    out.is_intersecting = true;
    out.distance        = best_t;
    out.point           = ray.point_at(best_t);
    out.normal          = best_n;

    return out;
}

template <typename T>
HitSurface<T>
CylinderGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

} // namespace atlas::geometry