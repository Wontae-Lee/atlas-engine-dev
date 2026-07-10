#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

/**
 * @brief Right circular cylinder primitive aligned with the world z-axis.
 *
 * The cylinder is centered at @ref center with the given @ref radius and total
 * @ref height, so it spans `z in [center.z - height/2, center.z + height/2]`. It
 * is always axis-aligned (no arbitrary orientation). When @ref open is `false`
 * the solid is capped on both ends; when `true` it is a bare lateral tube with no
 * caps, which changes the inside/surface classification and the ray hit set.
 *
 * One leaf of the `Geometry` tagged union, trivially copyable, every query is
 * `ATLAS_ALL_DEVICE`.
 */
class Cylinder final {
public:
    /// Host-side fluent builder that validates parameters before constructing a Cylinder.
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f); ///< Cylinder center (midpoint of the axis).
    float radius  = 1.0f;                     ///< Cross-section radius; must be positive.
    float height  = 1.0f;                     ///< Total axial extent along z; must be positive.
    bool open     = false;                    ///< `true` = lateral tube only; `false` = capped solid.

    /**
     * @brief Constructs a unit capped cylinder centered on the origin.
     */
    Cylinder() noexcept = default;

    /**
     * @brief Constructs a capped cylinder from center, radius, and height.
     *
     * @param center_ Axis midpoint.
     * @param radius_ Cross-section radius; should be positive.
     * @param height_ Total axial extent along z; should be positive.
     * @note Always constructs a closed (capped) cylinder; use the builder's
     *       `with_open` to make it open. Callable on host and device.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Cylinder(const Float3& center_, float radius_, float height_) noexcept
        : center(center_)
        , radius(radius_)
        , height(height_)
        , open(false) { }

    Cylinder(const Cylinder& other) noexcept = default; ///< Trivial copy.
    Cylinder(Cylinder&& other) noexcept      = default; ///< Trivial move.
    /// Trivial copy assignment.
    Cylinder&
    operator=(const Cylinder& other) noexcept = default;
    /// Trivial move assignment.
    Cylinder&
    operator=(Cylinder&& other) noexcept = default;

    ~Cylinder() noexcept = default; ///< Trivial destructor.

    /**
     * @brief Returns a fresh host-side builder for constructing a validated Cylinder.
     * @return A default-initialized `Cylinder::Builder`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Returns the point on the cylinder surface nearest to @p p.
     *
     * For an open cylinder the result is always on the lateral wall (the radial
     * projection of @p p onto the tube, clamped in z to the height band). For a
     * capped cylinder an interior point is pushed to whichever of the side wall or
     * the two caps is nearest, and an exterior point is clamped to the nearest
     * surface feature.
     *
     * @param p Query point in world space.
     * @return The nearest surface point. On the cylinder axis (`rho == 0`) an
     *         arbitrary side direction (`+x`) is chosen for the wall projection.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const float hz              = height * 0.5f;
        const float zmin            = center.z - hz;
        const float zmax            = center.z + hz;
        const bool is_open_cylinder = open;

        const Float3 d  = p - center;
        const float rho = atlas::xy_length(d);

        const float zc = (p.z < zmin) ? zmin
            : (p.z > zmax)            ? zmax
                                      : p.z;

        float sx = p.x;
        float sy = p.y;

        if (rho > radius) {
            // Outside radially: pull the xy position onto the tube of the radius.
            const float inv = 1.0f / rho;
            sx              = center.x + d.x * (radius * inv);
            sy              = center.y + d.y * (radius * inv);
        }

        Float3 cp(sx, sy, zc);

        const bool inside_radial = (rho <= radius);
        const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

        if (is_open_cylinder) {
            // Open tube: always project onto the lateral wall, keeping z clamped.
            if (rho > 0.0f) {
                const float inv = 1.0f / rho;
                cp.x            = center.x + d.x * (radius * inv);
                cp.y            = center.y + d.y * (radius * inv);
            } else {
                cp.x = center.x + radius;
                cp.y = center.y;
            }

            cp.z = zc;
            return cp;
        }

        if (inside_radial && inside_z) {
            // Interior of a capped cylinder: exit through the nearest of the wall,
            // the bottom cap, or the top cap.
            const float d_to_side = radius - rho;
            const float d_to_bot  = p.z - zmin;
            const float d_to_top  = zmax - p.z;

            if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
                if (rho > 0.0f) {
                    const float inv = 1.0f / rho;
                    cp.x            = center.x + d.x * (radius * inv);
                    cp.y            = center.y + d.y * (radius * inv);
                } else {
                    cp.x = center.x + radius;
                    cp.y = center.y;
                }

                cp.z = p.z;
            } else if (d_to_bot <= d_to_top) {
                cp.x = p.x;
                cp.y = p.y;
                cp.z = zmin;
            } else {
                cp.x = p.x;
                cp.y = p.y;
                cp.z = zmax;
            }
        }

        return cp;
    }

    /**
     * @brief Returns the outward unit surface normal nearest to @p p.
     *
     * For an open cylinder the normal is radial (derived from the closest wall
     * point). For a capped cylinder an interior point takes the normal of the
     * nearest feature (radial wall or `±z` cap); an exterior point is classified by
     * where its closest surface point lands (a cap plane within epsilon, otherwise
     * the wall).
     *
     * @param p Query point in world space.
     * @return An outward unit normal; on the axis the radial fallback `+x` is used.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const float hz              = height * 0.5f;
        const float zmin            = center.z - hz;
        const float zmax            = center.z + hz;
        const bool is_open_cylinder = open;

        const Float3 d  = p - center;
        const float rho = atlas::xy_length(d);

        const bool inside_radial = (rho <= radius);
        const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

        if (is_open_cylinder) {
            const Float3 cp = closest_point(p);
            const Float3 cd = cp - center;
            return atlas::xy_normalized_or(cd, Float3(1.0f, 0.0f, 0.0f));
        }

        if (inside_radial && inside_z) {
            const float d_to_side = radius - rho;
            const float d_to_bot  = p.z - zmin;
            const float d_to_top  = zmax - p.z;

            if (d_to_side <= d_to_bot && d_to_side <= d_to_top) {
                return atlas::xy_normalized_or(d, Float3(1.0f, 0.0f, 0.0f));
            }

            return (d_to_bot <= d_to_top)
                ? Float3(0.0f, 0.0f, -1.0f)
                : Float3(0.0f, 0.0f, 1.0f);
        }

        // Exterior: infer the feature from where the closest point landed in z.
        const Float3 cp = closest_point(p);
        const float e   = std::numeric_limits<float>::epsilon();

        if (std::abs(cp.z - zmin) <= e) {
            return Float3(0.0f, 0.0f, -1.0f);
        }

        if (std::abs(cp.z - zmax) <= e) {
            return Float3(0.0f, 0.0f, 1.0f);
        }

        const Float3 cd = cp - center;
        return atlas::xy_normalized_or(cd, Float3(1.0f, 0.0f, 0.0f));
    }

    /**
     * @brief Signed distance from @p p to the cylinder surface.
     *
     * Built from the radial excess `qx = rho - radius` and axial excess
     * `qy = |dz| - height/2`. For a capped cylinder this is the exact signed
     * distance (negative inside, positive outside). For an open tube it is the
     * signed radial distance while within the height band and the distance to the
     * nearest rim edge otherwise.
     *
     * @param p Query point in world space.
     * @return The signed distance; sign convention as described above.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const bool is_open_cylinder = open;

        float qx;
        float qy;
        radial_axial(p, qx, qy);

        if (is_open_cylinder) {
            if (qy <= 0.0f) {
                // Within the height band: signed distance to the tube wall.
                return qx;
            }

            // Above/below the band: distance to the nearest rim circle.
            return atlas::sqrt_nonnegative(qx * qx + qy * qy);
        }

        // Capped: standard box-in-(radius, axis) signed distance.
        const float ax = (qx > 0.0f) ? qx : 0.0f;
        const float ay = (qy > 0.0f) ? qy : 0.0f;

        const float outside = atlas::sqrt_nonnegative(ax * ax + ay * ay);

        const float mxy    = (qx > qy) ? qx : qy;
        const float inside = (mxy < 0.0f) ? mxy : 0.0f;

        return outside + inside;
    }

    /**
     * @brief Tests whether @p p lies inside the cylinder, optionally dilated.
     *
     * @param p Query point in world space.
     * @param tolerance Signed shell half-thickness; positive grows the solid,
     *        negative shrinks it.
     * @return `true` when @p p is inside the tolerance-adjusted cylinder. For an
     *         open tube "inside" means within the height band and no farther than
     *         @p tolerance outside the wall.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        const bool is_open_cylinder = open;

        float qx;
        float qy;
        radial_axial(p, qx, qy);

        if (is_open_cylinder) {
            return qy <= 0.0f && qx <= tolerance;
        }

        if (qx <= 0.0f && qy <= 0.0f) {
            // Strictly interior: the deeper of the two penetrations governs.
            const float inside = (qx > qy) ? qx : qy;
            return inside <= tolerance;
        }

        if (tolerance < 0.0f) {
            return false;
        }

        // Outside on at least one axis: accept only within the exterior tolerance shell.
        const float ax = (qx > 0.0f) ? qx : 0.0f;
        const float ay = (qy > 0.0f) ? qy : 0.0f;

        return (ax * ax + ay * ay) <= (tolerance * tolerance);
    }

    /**
     * @brief Tests whether @p p lies within @p tolerance of the cylinder surface.
     *
     * For an open tube the point must be within @p tolerance of the wall radius and
     * inside the (tolerance-extended) height band. For a capped cylinder the point
     * must be within @p tolerance of the solid's boundary.
     *
     * @param p Query point in world space.
     * @param tolerance Non-negative shell half-thickness; a negative value returns
     *        `false`.
     * @return `true` when @p p is on the (thickened) surface. Always `false` for a
     *         non-positive radius or height.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f) || !(height > 0.0f) || tolerance < 0.0f) {
            return false;
        }

        if (open) {
            const float hz = height * 0.5f;
            const Float3 d = p - center;

            const float rho  = atlas::xy_length(d);
            const float zmin = center.z - hz;
            const float zmax = center.z + hz;

            return std::abs(rho - radius) <= tolerance
                && p.z >= zmin - tolerance
                && p.z <= zmax + tolerance;
        }

        float radial;
        float axial;
        radial_axial(p, radial, axial);

        if (radial <= 0.0f && axial <= 0.0f) {
            // Interior: on the surface only if the shallowest boundary is within
            // tolerance (i.e. the least-negative excess is >= -tolerance).
            const float inside_distance = (radial > axial) ? radial : axial;
            return inside_distance >= -tolerance;
        }

        const float ax = (radial > 0.0f) ? radial : 0.0f;
        const float ay = (axial > 0.0f) ? axial : 0.0f;

        return ax * ax + ay * ay <= tolerance * tolerance;
    }

    /**
     * @brief Geometric center of the cylinder.
     * @return @ref center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    /**
     * @brief Axis-aligned bounding box of the cylinder.
     * @return An `AABB` extending `radius` in x and y and `height/2` in z about
     *         the center. (Not tightened for the `open` case, which shares the same
     *         extent.)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float hz = height * 0.5f;

        return AABB(
            Float3(center.x - radius, center.y - radius, center.z - hz),
            Float3(center.x + radius, center.y + radius, center.z + hz));
    }

    /**
     * @brief Reports whether the cylinder parameters are well-formed.
     * @return `true` when both @ref radius and @ref height are positive.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return radius > 0.0f && height > 0.0f;
    }

    /**
     * @brief Intersects a ray with the cylinder.
     *
     * Works in cylinder-local coordinates (origin shifted to @ref center). The
     * lateral wall is solved as a 2D quadratic in the xy-plane, intersected with
     * the z slab defined by the height. When the cylinder is capped and the ray is
     * not axis-parallel, the two cap disks are also tested, and the nearest forward
     * hit among wall and caps wins.
     *
     * @param ray Ray with an already-normalized direction.
     * @return A `HitSurface`; `is_intersecting` is `false` when no forward
     *         intersection exists.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface out {};

        const Float3 ro = ray.origin - center;
        const Float3 rd = ray.direction;

        const float r               = radius;
        const float hz              = height * 0.5f;
        const float zmin            = -hz;
        const float zmax            = hz;
        const bool is_open_cylinder = open;

        float best_t = std::numeric_limits<float>::infinity();
        Float3 best_n(0.0f, 0.0f, 0.0f);

        // z-slab interval the wall hit must fall within.
        bool z_ok       = true;
        float t_z_enter = -std::numeric_limits<float>::infinity();
        float t_z_exit  = std::numeric_limits<float>::infinity();

        if (rd.z == 0.0f) {
            // Ray runs perpendicular to the axis: it can only hit the wall if it
            // already lies between the caps.
            if (ro.z < zmin || ro.z > zmax) {
                z_ok = false;
            }
        } else {
            const float inv_dz = 1.0f / rd.z;
            float a            = (zmin - ro.z) * inv_dz;
            float b            = (zmax - ro.z) * inv_dz;

            if (a > b) {
                const float tmp = a;
                a               = b;
                b               = tmp;
            }

            t_z_enter = a;
            t_z_exit  = b;
        }

        if (z_ok) {
            // Lateral wall: |ro_xy + t rd_xy|^2 = r^2.
            const float A = atlas::xy_length_squared(rd);
            const float B = 2.0f * atlas::xy_dot(ro, rd);
            const float C = atlas::xy_length_squared(ro) - r * r;

            if (A > 0.0f) {
                float t0;
                float t1;

                if (atlas::solve_quadratic(A, B, C, t0, t1)) {
                    auto accept_side = [&](const float t) -> bool {
                        if (!(t >= 0.0f)) {
                            return false;
                        }

                        // Wall hit must lie inside the z-slab (unless z-invariant).
                        if (rd.z != 0.0f && (t < t_z_enter || t > t_z_exit)) {
                            return false;
                        }

                        return true;
                    };

                    auto set_side_hit = [&](const float t) {
                        best_t = t;

                        const Float3 ph = ro + rd * t;
                        best_n          = atlas::xy_normalized_or(
                            ph,
                            Float3(1.0f, 0.0f, 0.0f));
                    };

                    // t0 <= t1; prefer the nearer root, else fall back to the far one.
                    if (accept_side(t0)) {
                        set_side_hit(t0);
                    }

                    if (!atlas::isfinite(best_t) && accept_side(t1)) {
                        set_side_hit(t1);
                    }
                }
            }
        }

        if (!is_open_cylinder && rd.z != 0.0f) {
            // Cap disks: keep whichever forward hit is closer than the wall hit.
            auto try_cap = [&](const float zplane, const float nz) {
                const float t = (zplane - ro.z) / rd.z;

                if (!(t >= 0.0f) || t >= best_t) {
                    return;
                }

                const Float3 ph = ro + rd * t;

                if (atlas::xy_length_squared(ph) <= r * r) {
                    best_t = t;
                    best_n = Float3(0.0f, 0.0f, nz);
                }
            };

            // Test the nearer cap first so the early-out on `t >= best_t` is tight.
            const float t_min = (zmin - ro.z) / rd.z;
            const float t_max = (zmax - ro.z) / rd.z;

            if (t_min < t_max) {
                try_cap(zmin, -1.0f);
                try_cap(zmax, 1.0f);
            } else {
                try_cap(zmax, 1.0f);
                try_cap(zmin, -1.0f);
            }
        }

        if (!atlas::isfinite(best_t)) {
            return out;
        }

        out.is_intersecting = true;
        out.distance        = best_t;
        out.point           = ray.point_at(best_t);
        out.normal          = best_n;

        return out;
    }

private:
    /**
     * @brief Computes the radial and axial excess distances of @p p.
     *
     * @param p Query point in world space.
     * @param[out] qx Radial excess `xy_length(p - center) - radius` (negative
     *        inside the tube radius, positive outside).
     * @param[out] qy Axial excess `|p.z - center.z| - height/2` (negative between
     *        the caps, positive beyond them).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    radial_axial(const Float3& p, float& qx, float& qy) const noexcept {
        const float hz = height * 0.5f;
        const Float3 d = p - center;

        qx = atlas::xy_length(d) - radius;
        qy = std::abs(d.z) - hz;
    }
};

/**
 * @brief Host-side builder for `Cylinder` with validation.
 *
 * Accumulates parameters through `with_*` setters (including the `open` flag) and,
 * on `build()`, throws if the resulting cylinder is invalid. Runs on the host only.
 */
class Cylinder::Builder final {
public:
    Builder() = default; ///< Starts from a unit capped cylinder on the origin.

    /**
     * @brief Validates the accumulated parameters and constructs a `Cylinder`.
     * @return The constructed cylinder, with the `open` flag applied.
     * @throws std::runtime_error when radius or height is non-positive.
     */
    ATLAS_NODISCARD ATLAS_HOST Cylinder
    build() const;

    /**
     * @brief Builds the cylinder and wraps it in a host `shared_ptr`.
     * @return A `host_shared_ptr<Cylinder>` owning the constructed cylinder.
     * @throws std::runtime_error when radius or height is non-positive.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Cylinder>
    make_host_shared() const;

    /**
     * @brief Sets the axis midpoint.
     * @param center_ Cylinder center.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_center(const Float3& center_) noexcept;

    /**
     * @brief Sets the cross-section radius.
     * @param radius_ Positive radius.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_radius(float radius_) noexcept;

    /**
     * @brief Sets the total axial height.
     * @param height_ Positive height.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_height(float height_) noexcept;

    /**
     * @brief Selects a capped (`false`) or open lateral-tube (`true`) cylinder.
     * @param open_ `true` to omit the end caps.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_open(bool open_) noexcept;

private:
    /**
     * @brief Throws if the accumulated radius/height are not both positive.
     * @throws std::runtime_error on invalid parameters.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f); ///< Pending center.
    float _radius  = 1.0f;                     ///< Pending radius.
    float _height  = 1.0f;                     ///< Pending height.
    bool _open     = false;                    ///< Pending open/capped flag.
};

/// Owning host handle to a `Cylinder`.
using CylinderHostPtr = atlas::host_shared_ptr<Cylinder>;

/// Owning device handle to a `Cylinder`.
using CylinderDevicePtr = atlas::device_shared_ptr<Cylinder>;

}
