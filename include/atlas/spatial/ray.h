#pragma once

/**
 * @file ray.h
 * @brief Declares a lightweight geometric ray and a surface intersection record.
 *
 * @details
 * This header defines:
 * - @ref atlas::SurfaceRayIntersection, a structure describing the result
 *   of intersecting a ray with a surface,
 * - @ref atlas::Ray, a minimal geometric ray representation.
 *
 * ## Purpose
 * Rays are fundamental primitives used throughout Atlas for:
 * - ray casting and intersection tests,
 * - collision detection,
 * - closest-point and signed-distance queries,
 * - BVH traversal and acceleration structures.
 *
 * ## Conventions
 * A ray is defined parametrically as:
 * \f[
 * \mathbf{p}(t) = \mathbf{o} + t \mathbf{d}, \quad t \ge 0,
 * \f]
 * where:
 * - \f$\mathbf{o}\f$ is the origin,
 * - \f$\mathbf{d}\f$ is the direction,
 * - \f$t\f$ is the scalar parameter.
 *
 * The direction is not required to be normalized.
 *
 * ## Intersection result
 * Surface intersections are reported using
 * @ref SurfaceRayIntersection, which provides:
 * - a hit flag,
 * - a parametric distance,
 * - the hit point,
 * - the surface normal at the hit.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for coordinates and distances.
 */

#include <atlas/math/math.h>
#include <limits>
#include <type_traits>

namespace atlas {

/**
 * @brief Surface ray intersection record.
 *
 * @details
 * Represents the result of intersecting a ray with a surface.
 *
 * If an intersection occurs:
 * - @ref is_intersecting is `true`,
 * - @ref distance stores the parametric ray distance,
 * - @ref point stores the hit position,
 * - @ref normal stores the surface normal at the hit.
 *
 * If no intersection occurs:
 * - @ref is_intersecting is `false`,
 * - @ref distance is typically left at its maximum value.
 *
 * The coordinate space (world or local) of @ref point and @ref normal is
 * determined by the calling context.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry.
 */
template <typename T>
struct SurfaceRayIntersection {

    /**
     * @brief Whether an intersection was found.
     */
    bool is_intersecting = false;

    /**
     * @brief Parametric ray distance at the hit.
     *
     * @details
     * Corresponds to the parameter \f$t\f$ in the ray equation.
     */
    T distance = std::numeric_limits<T>::max();

    /**
     * @brief Hit point.
     *
     * @details
     * Represents the evaluated position on the ray at @ref distance.
     */
    Vector3<T> point { T(0), T(0), T(0) };

    /**
     * @brief Surface normal at the hit.
     *
     * @details
     * Orientation is typically outward-facing but depends on the geometry.
     */
    Vector3<T> normal { T(0), T(0), T(1) };
};

/**
 * @brief Lightweight ray defined by origin and direction.
 *
 * @details
 * Represents a parametric line used for intersection and traversal queries.
 *
 * The ray is expressed as:
 * \f[
 * \mathbf{p}(t) = \mathbf{origin} + t \cdot \mathbf{direction}.
 * \f]
 *
 * ## Notes
 * - The direction vector is not required to be normalized.
 * - No internal normalization or validation is enforced.
 * - The ray is intended to be trivially copyable and usable on both host and device.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for coordinates.
 */
template <typename T>
class Ray final {
    static_assert(std::is_floating_point_v<T>,
                  "Ray only can be instantiated with floating point types");

public:
    /**
     * @brief Ray origin.
     */
    Vector3<T> origin;

    /**
     * @brief Ray direction.
     *
     * @details
     * Not required to be normalized.
     */
    Vector3<T> direction;

    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the ray according to the implementation in `ray.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray() noexcept;

    /**
     * @brief Construct a ray from origin and direction.
     *
     * @param origin_ Ray origin.
     * @param direction_ Ray direction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept;

    /**
     * @brief Copy constructor.
     */
    Ray(const Ray& other) noexcept = default;

    /**
     * @brief Evaluate a point along the ray.
     *
     * @details
     * Computes:
     * \f[
     * \mathbf{origin} + t \cdot \mathbf{direction}.
     * \f]
     *
     * @param t Parametric distance along the ray.
     * @return Point at parameter \f$t\f$.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    point_at(T t) const noexcept;
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::SurfaceRayIntersection.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using HitSurface = atlas::SurfaceRayIntersection<T>;


/**
 * @brief Single-precision ray alias.
 */
using RayF = Ray<float>;

/**
 * @brief Double-precision ray alias.
 */
using RayD = Ray<double>;

} // namespace atlas

#include <atlas/spatial/ray.hpp>