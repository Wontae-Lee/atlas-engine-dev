#pragma once

#include <atlas/math/math.h>

#include <cmath>
#include <limits>

namespace atlas {

struct HitSurface {

    bool is_intersecting = false;

    float distance = std::numeric_limits<float>::max();

    Float3 point = Float3(0.0f, 0.0f, 0.0f);

    Float3 normal = Float3(0.0f, 0.0f, 1.0f);
};

class Ray final {
public:
    Float3 origin;

    Float3 direction;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept
        : origin(0.0f, 0.0f, 0.0f)
        , direction(1.0f, 0.0f, 0.0f) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Float3& origin_, const Float3& direction_) noexcept
        : origin(origin_)
        , direction(atlas::normalized_or(
              direction_,
              Float3(0.0f, 0.0f, 0.0f))) { }

    Ray(const Ray& other) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    point_at(const float t) const noexcept {
        return origin + t * direction;
    }
};

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
ray_plane_distance(const Float3& plane_point,
                   const Float3& plane_normal,
                   const Ray& ray,
                   float& distance) noexcept {
    const float denominator = plane_normal.dot(ray.direction);

    if (std::abs(denominator) <= eps) {
        return false;
    }

    distance = (plane_point - ray.origin).dot(plane_normal) / denominator;

    return distance >= 0.0f;
}

using RayF = Ray;

}