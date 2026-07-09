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

class Cylinder final {
public:
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f);
    float radius  = 1.0f;
    float height  = 1.0f;
    bool open     = false;

    Cylinder() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Cylinder(const Float3& center_, float radius_, float height_) noexcept
        : center(center_)
        , radius(radius_)
        , height(height_)
        , open(false) { }

    Cylinder(const Cylinder& other) noexcept = default;
    Cylinder(Cylinder&& other) noexcept      = default;
    Cylinder&
    operator=(const Cylinder& other) noexcept = default;
    Cylinder&
    operator=(Cylinder&& other) noexcept = default;

    ~Cylinder() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

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
            const float inv = 1.0f / rho;
            sx              = center.x + d.x * (radius * inv);
            sy              = center.y + d.y * (radius * inv);
        }

        Float3 cp(sx, sy, zc);

        const bool inside_radial = (rho <= radius);
        const bool inside_z      = (p.z >= zmin) && (p.z <= zmax);

        if (is_open_cylinder) {
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

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const bool is_open_cylinder = open;

        float qx;
        float qy;
        radial_axial(p, qx, qy);

        if (is_open_cylinder) {
            if (qy <= 0.0f) {
                return qx;
            }

            return atlas::sqrt_nonnegative(qx * qx + qy * qy);
        }

        const float ax = (qx > 0.0f) ? qx : 0.0f;
        const float ay = (qy > 0.0f) ? qy : 0.0f;

        const float outside = atlas::sqrt_nonnegative(ax * ax + ay * ay);

        const float mxy    = (qx > qy) ? qx : qy;
        const float inside = (mxy < 0.0f) ? mxy : 0.0f;

        return outside + inside;
    }

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
            const float inside = (qx > qy) ? qx : qy;
            return inside <= tolerance;
        }

        if (tolerance < 0.0f) {
            return false;
        }

        const float ax = (qx > 0.0f) ? qx : 0.0f;
        const float ay = (qy > 0.0f) ? qy : 0.0f;

        return (ax * ax + ay * ay) <= (tolerance * tolerance);
    }

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
            const float inside_distance = (radial > axial) ? radial : axial;
            return inside_distance >= -tolerance;
        }

        const float ax = (radial > 0.0f) ? radial : 0.0f;
        const float ay = (axial > 0.0f) ? axial : 0.0f;

        return ax * ax + ay * ay <= tolerance * tolerance;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float hz = height * 0.5f;

        return AABB(
            Float3(center.x - radius, center.y - radius, center.z - hz),
            Float3(center.x + radius, center.y + radius, center.z + hz));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return radius > 0.0f && height > 0.0f;
    }

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

        bool z_ok       = true;
        float t_z_enter = -std::numeric_limits<float>::infinity();
        float t_z_exit  = std::numeric_limits<float>::infinity();

        if (rd.z == 0.0f) {
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

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    radial_axial(const Float3& p, float& qx, float& qy) const noexcept {
        const float hz = height * 0.5f;
        const Float3 d = p - center;

        qx = atlas::xy_length(d) - radius;
        qy = std::abs(d.z) - hz;
    }
};

class Cylinder::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Cylinder
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Cylinder>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_center(const Float3& center_) noexcept;

    ATLAS_HOST Builder&
    with_radius(float radius_) noexcept;

    ATLAS_HOST Builder&
    with_height(float height_) noexcept;

    ATLAS_HOST Builder&
    with_open(bool open_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f);
    float _radius  = 1.0f;
    float _height  = 1.0f;
    bool _open     = false;
};

using CylinderHostPtr = atlas::host_shared_ptr<Cylinder>;

using CylinderDevicePtr = atlas::device_shared_ptr<Cylinder>;

}