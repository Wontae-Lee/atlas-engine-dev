#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cstddef>
#include <limits>

namespace atlas {

class Sphere final {
public:
    class Builder;

public:
    Float3 center = Float3(0.0f, 0.0f, 0.0f);

    float radius = 1.0f;

    Sphere() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sphere(const Float3& center_, const float radius_) noexcept
        : center(center_)
        , radius(radius_) { }

    Sphere(const Sphere& other) noexcept = default;
    Sphere(Sphere&& other) noexcept      = default;
    Sphere&
    operator=(const Sphere& other) noexcept = default;
    Sphere&
    operator=(Sphere&& other) noexcept = default;

    ~Sphere() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const Float3 v = p - center;
        const float e  = std::numeric_limits<float>::epsilon();

        const Float3 direction = atlas::normalized_or(
            v,
            Float3(1.0f, 0.0f, 0.0f),
            e);
        return center + direction * radius;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        const Float3 v = p - center;
        const float e  = std::numeric_limits<float>::epsilon();

        return atlas::normalized_or(
            v,
            Float3(1.0f, 0.0f, 0.0f),
            e);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        return (p - center).length() - radius;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        const float expanded_radius = radius + tolerance;

        if (expanded_radius < 0.0f) {
            return false;
        }

        return (p - center).length_squared() <= expanded_radius * expanded_radius;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(radius > 0.0f) || tolerance < 0.0f) {
            return false;
        }

        const float outer_radius = radius + tolerance;
        const float inner_radius = (radius > tolerance) ? radius - tolerance : 0.0f;
        const float d2           = (p - center).length_squared();

        return d2 >= inner_radius * inner_radius
            && d2 <= outer_radius * outer_radius;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const Float3 dr(radius, radius, radius);

        return AABB(center - dr, center + dr);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return radius > 0.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        const Float3 oc = ray.origin - center;

        const float a  = ray.direction.length_squared();
        const float b  = 2.0f * oc.dot(ray.direction);
        const float cc = oc.length_squared() - radius * radius;

        float t0;
        float t1;

        if (!atlas::solve_quadratic(a, b, cc, t0, t1)) {
            return result;
        }

        if (t0 < 0.0f && t1 < 0.0f) {
            return result;
        }

        float t = std::numeric_limits<float>::infinity();

        if (t0 >= 0.0f) {
            t = t0;
        }

        if (t1 >= 0.0f && t1 < t) {
            t = t1;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = ray.point_at(t);

        result.normal = atlas::normalized_or(
            result.point - center,
            Float3(1.0f, 0.0f, 0.0f));

        return result;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }
};

class Sphere::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Sphere
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Sphere>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_center(const Float3& c) noexcept;

    ATLAS_HOST Builder&
    with_radius(float r) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center = Float3(0.0f, 0.0f, 0.0f);

    float _radius = 1.0f;
};

using SphereHostPtr = atlas::host_shared_ptr<Sphere>;

using SphereDevicePtr = atlas::device_shared_ptr<Sphere>;

}
