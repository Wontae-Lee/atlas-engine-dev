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

class Plane final {
public:
    class Builder;

public:
    Float3 normal = Float3(0.0f, 0.0f, 1.0f);

    float offset = 0.0f;

    Plane() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Plane(const Float3& normal_, float offset_) noexcept
        : normal(normal_)
        , offset(offset_) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Plane(const Float3& point, const Float3& normal_) noexcept
        : normal(normal_)
        , offset(-(normal_.dot(point))) { }

    Plane(const Plane& other) noexcept = default;
    Plane(Plane&& other) noexcept      = default;
    Plane&
    operator=(const Plane& other) noexcept = default;
    Plane&
    operator=(Plane&& other) noexcept = default;

    ~Plane() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        const float sdev = normal.dot(p) + offset;

        return p - sdev * normal;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return normal;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        return normal.dot(p) + offset;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return normal.dot(p) + offset <= tolerance;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const float distance = normal.dot(p) + offset;
        return distance >= -tolerance && distance <= tolerance;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return Float3(0.0f, 0.0f, 0.0f);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const float lo = std::numeric_limits<float>::lowest();
        const float hi = std::numeric_limits<float>::max();

        return AABB(
            Float3(lo, lo, lo),
            Float3(hi, hi, hi));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        const float n2 = normal.length_squared();
        return atlas::isfinite(normal)
            && (n2 > 0.0f)
            && atlas::isfinite(offset);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface result {};

        const float denom = normal.dot(ray.direction);

        const float numer = -(normal.dot(ray.origin) + offset);

        if (std::abs(denom) <= eps) {

            if (numer != 0.0f) {
                return result;
            }

            result.is_intersecting = true;
            result.distance        = 0.0f;
            result.point           = ray.origin;
            result.normal          = normal;

            return result;
        }

        const float t = numer / denom;

        if (t < 0.0f) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = ray.point_at(t);
        result.normal          = normal;

        return result;
    }
};

class Plane::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Plane
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Plane>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

    ATLAS_HOST Builder&
    with_offset(float offset_) noexcept;

    ATLAS_HOST Builder&
    with_normal_offset(const Float3& normal_, float offset_) noexcept;

    ATLAS_HOST Builder&
    with_point_normal(const Float3& point, const Float3& normal_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _normal = Float3(0.0f, 0.0f, 1.0f);

    float _offset = 0.0f;
};

using PlaneHostPtr = atlas::host_shared_ptr<Plane>;

using PlaneDevicePtr = atlas::device_shared_ptr<Plane>;

}