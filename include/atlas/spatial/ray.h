#pragma once
#include <atlas/math/math.h>
#include <limits>
#include <type_traits>

namespace atlas::spatial {

/**
 * @brief Result of ray vs. surface intersection.
 *
 * @details
 * This structure represents the closest intersection (if any) between a ray and
 * a geometric surface. It is used as a common return type for trace operators
 * (e.g., sphere/box/triangle/BVH tracing).
 *
 * Fields:
 * - `is_intersecting`: whether an intersection exists.
 * - `distance`: parametric ray distance \f$t\f$ at the hit (typically the smallest
 *   non-negative solution). When no hit, defaults to `max()`.
 * - `point`: hit position \f$P = O + tD\f$ in world space.
 * - `normal`: surface normal at the hit point. Conventionally unit-length; when
 *   no hit, defaults to +Z.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceRayIntersection {
    /// True if the ray intersects (or touches) the surface.
    bool is_intersecting = false;

    /// Parametric distance along the ray (t). Smallest valid hit is typically chosen.
    T distance = std::numeric_limits<T>::max();

    /// World-space hit position.
    Vector3<T> point { T(0), T(0), T(0) };

    /// World-space surface normal at the hit (usually normalized).
    Vector3<T> normal { T(0), T(0), T(1) };
};

/**
 * @brief 3D ray represented by an origin and a direction.
 *
 * @details
 * A ray is defined as:
 * \f[
 *   R(t) = O + tD
 * \f]
 * where:
 * - \f$O\f$ is `origin`
 * - \f$D\f$ is `direction`
 * - \f$t\f$ is a scalar parameter (often \f$t \ge 0\f$ for forward rays)
 *
 * Usage notes:
 * - Many intersection routines assume `direction` is normalized; this class does
 *   not enforce normalization. If `direction` is not unit-length, returned hit
 *   distances are still valid in parametric units of that direction vector.
 * - `point_at(t)` is a convenience helper to evaluate the parametric position.
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 */
template <typename T>
class Ray final {
    static_assert(std::is_floating_point_v<T>,
                  "Ray only can be instantiated with floating point types");

public:
    /// Ray origin (O).
    Vector3<T> origin;

    /// Ray direction (D).
    Vector3<T> direction;

    /**
     * @brief Construct a default ray.
     *
     * @details
     * The exact default values are defined in the implementation (.hpp). Typically
     * origin is (0,0,0) and direction is (0,0,1) or another sensible default.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept;

    /**
     * @brief Construct a ray from origin and direction.
     *
     * @param origin_ Ray origin.
     * @param direction_ Ray direction.
     *
     * @note This constructor does not normalize `direction_`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept;

    /// Copy construction.
    Ray(const Ray& other) noexcept = default;

    /**
     * @brief Evaluate the ray at parameter t.
     *
     * @param t Parametric distance along the ray.
     * @return Position `origin + direction * t`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    point_at(T t) const noexcept;
};

} // namespace atlas::spatial

namespace atlas {

template <typename T>
using HitSurface = atlas::spatial::SurfaceRayIntersection<T>;

template <typename T>
using Ray = atlas::spatial::Ray<T>;

using RayF = Ray<float>;
using RayD = Ray<double>;

} // namespace atlas

#include <atlas/spatial/ray.hpp>
