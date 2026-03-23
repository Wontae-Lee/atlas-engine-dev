#pragma once
#include <atlas/math/math.h>
#include <limits>
#include <type_traits>

namespace atlas::spatial {

template <typename T>
struct SurfaceRayIntersection {

    bool is_intersecting = false;

    T distance = std::numeric_limits<T>::max();

    Vector3<T> point { T(0), T(0), T(0) };

    Vector3<T> normal { T(0), T(0), T(1) };
};

template <typename T>
class Ray final {
    static_assert(std::is_floating_point_v<T>,
                  "Ray only can be instantiated with floating point types");

public:
    Vector3<T> origin;

    Vector3<T> direction;

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

template <typename T>
using HitSurface = atlas::spatial::SurfaceRayIntersection<T>;

template <typename T>
using Ray = atlas::spatial::Ray<T>;

using RayF = Ray<float>;
using RayD = Ray<double>;

}

#include <atlas/spatial/ray.hpp>