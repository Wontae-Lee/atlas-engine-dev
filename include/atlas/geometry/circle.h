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

class Circle final {
public:
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f);
    Float3 normal = Float3(0.0f, 0.0f, 1.0f);
    float radius  = 1.0f;

    Circle() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Circle(const Float3& center_, const Float3& normal_, const float radius_) noexcept
        : center(center_)
        , normal(normal_)
        , radius(radius_) { }

    Circle(const Circle& other) noexcept = default;
    Circle(Circle&& other) noexcept      = default;
    Circle&
    operator=(const Circle& other) noexcept = default;
    Circle&
    operator=(Circle&& other) noexcept = default;

    ~Circle() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const float n2 = normal.length_squared();

        if (n2 <= 0.0f || radius <= 0.0f) {
            return p;
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 offset        = p - center;
        const float plane_distance = offset.dot(n);

        const Float3 planar     = offset - n * plane_distance;
        const float planar_len2 = planar.length_squared();
        const float rr          = radius * radius;

        if (planar_len2 <= rr) {
            return p - n * plane_distance;
        }

        if (planar_len2 <= std::numeric_limits<float>::epsilon()) {
            return center + Float3(radius, 0.0f, 0.0f);
        }

        const float planar_len = atlas::sqrt_nonnegative(planar_len2);
        return center + planar * (radius / planar_len);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 cp       = closest_point(p);
        const Float3 nn       = closest_normal(p);
        const float magnitude = (p - cp).length();

        const float side = (p - center).dot(nn);

        return (side >= 0.0f) ? magnitude : -magnitude;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f)) {
            return false;
        }

        float plane_distance;
        float distance2;

        if (!plane_and_radial(p, plane_distance, distance2)) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;

        if (!(plane_distance < 0.0f)) {
            return tolerance >= 0.0f && distance2 <= tolerance2;
        }

        return tolerance >= 0.0f || distance2 >= tolerance2;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f) || tolerance < 0.0f) {
            return false;
        }

        float plane_distance;
        float distance2;

        if (!plane_and_radial(p, plane_distance, distance2)) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;

        return distance2 <= tolerance2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float n2 = normal.length_squared();

        if (n2 <= 0.0f || radius <= 0.0f) {
            return AABB(center, center);
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 extent(
            radius * atlas::sqrt_nonnegative(1.0f - n.x * n.x),
            radius * atlas::sqrt_nonnegative(1.0f - n.y * n.y),
            radius * atlas::sqrt_nonnegative(1.0f - n.z * n.z));

        return AABB(center - extent, center + extent);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(center)
            && atlas::isfinite(normal)
            && normal.length_squared() > 0.0f
            && atlas::isfinite(radius)
            && radius > 0.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        if (!is_valid()) {
            return result;
        }

        const Float3 nn   = closest_normal(ray.origin);
        const float denom = nn.dot(ray.direction);

        if (std::abs(denom) <= eps) {
            const float plane_distance = (ray.origin - center).dot(nn);
            if (std::abs(plane_distance) > eps) {
                return result;
            }

            if ((ray.origin - center).length_squared() > radius * radius) {
                return result;
            }

            result.is_intersecting = true;
            result.distance        = 0.0f;
            result.point           = ray.origin;
            result.normal          = nn;

            return result;
        }

        float t;

        if (!atlas::ray_plane_distance(center, nn, ray, t)) {
            return result;
        }

        const Float3 hit_point = ray.point_at(t);

        if ((hit_point - center).length_squared() > radius * radius) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = hit_point;
        result.normal          = nn;

        return result;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    plane_and_radial(const Float3& p,
                     float& plane_distance,
                     float& distance2) const noexcept {
        const float n2 = normal.length_squared();

        if (!(n2 > 0.0f)) {
            return false;
        }

        const Float3 n = atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));

        const Float3 offset     = p - center;
        const Float3 planar     = offset - n * offset.dot(n);
        const float planar_len2 = planar.length_squared();
        const float rr          = radius * radius;

        plane_distance = offset.dot(n);
        distance2      = plane_distance * plane_distance;

        if (planar_len2 > rr) {
            const float radial_distance = atlas::sqrt_nonnegative(planar_len2) - radius;
            distance2 += radial_distance * radial_distance;
        }

        return true;
    }
};

class Circle::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Circle
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Circle>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_center(const Float3& center_) noexcept;

    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

    ATLAS_HOST Builder&
    with_radius(float radius_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f);
    Float3 _normal = Float3(0.0f, 0.0f, 1.0f);
    float _radius  = 1.0f;
};

using CircleHostPtr = atlas::host_shared_ptr<Circle>;

using CircleDevicePtr = atlas::device_shared_ptr<Circle>;

}