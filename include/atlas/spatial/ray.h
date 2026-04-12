#pragma once

/**
 * @file ray.h
 * @brief Declares a simple geometric ray and surface-hit record.
 */
#include <atlas/math/math.h>
#include <limits>
#include <type_traits>

namespace atlas::spatial {

/**
 * @brief Surface ray hit record.
 */
template <typename T>
struct SurfaceRayIntersection {
    bool is_intersecting = false; ///< Whether an intersection was found.
    T distance = std::numeric_limits<T>::max(); ///< Ray parameter at the hit.
    Vector3<T> point { T(0), T(0), T(0) }; ///< World/local hit point, depending on caller context.
    Vector3<T> normal { T(0), T(0), T(1) }; ///< Surface normal at the hit.
};

/**
 * @brief Lightweight ray with origin and direction.
 */
template <typename T>
class Ray final {
    static_assert(std::is_floating_point_v<T>,
                  "Ray only can be instantiated with floating point types");

public:
    Vector3<T> origin; ///< Ray origin.
    Vector3<T> direction; ///< Ray direction, not necessarily normalized.

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept;

    Ray(const Ray& other) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    point_at(T t) const noexcept;
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::spatial::SurfaceRayIntersection.
 */
template <typename T>
using HitSurface = atlas::spatial::SurfaceRayIntersection<T>;

template <typename T>
using Ray = atlas::spatial::Ray<T>;

using RayF = Ray<float>;
using RayD = Ray<double>;

}

#include <atlas/spatial/ray.hpp>
