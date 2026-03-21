#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct SphereGeometryOperator {
    const atlas::math::Vector<T, 3>* center = nullptr;
    const T* radius = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

/**
 * @brief Analytic sphere geometry primitive (center + radius).
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 *
 * @details
 * `Sphere<T>` represents a closed 3D ball:
 *
 *     S = { x ∈ ℝ³ | ||x - c|| ≤ r }
 *
 * where c is the center and r is the radius (r > 0).
 *
 * Core query formulas used by this geometry:
 * - Signed distance:
 *       d(p) = ||p - c|| - r
 * - Closest point on surface (p ≠ c):
 *       x* = c + r * (p - c) / ||p - c||
 * - Outward unit normal (p ≠ c):
 *       n(p) = (p - c) / ||p - c||
 *
 * Atlas integration:
 * - Provides a GeometryOperator for analytic ray–sphere intersection.
 * - Provides a GeometryOperator for closest-point / SDF queries.
 * - Supports host/device execution via ATLAS_* macros.
 */
template <typename T>
class Sphere final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Sphere requires a floating-point T");

public:
    class Builder;

public:
    /**
     * @brief Center of the sphere in ℝ³.
     *
     * @details
     * Defines c in:
     *
     *     S = { x | ||x - c|| ≤ r }
     */
    Vector3<T> center { T(0), T(0), T(0) };

    /**
     * @brief Radius of the sphere.
     *
     * @details
     * Must satisfy r > 0 for a valid sphere.
     * The surface satisfies:
     *
     *     ||x - center|| = radius
     */
    T radius { T(1) };

    /**
     * @brief Construct a unit sphere centered at the origin.
     *
     * @details
     * Initializes:
     * - center = (0,0,0)
     * - radius = 1
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere() noexcept;

    /**
     * @brief Construct a sphere with explicit center and radius.
     *
     * @param center_ Center point c.
     * @param radius_ Radius r (must be positive).
     *
     * @details
     * Creates the set:
     *
     *     S = { x | ||x - center_|| ≤ radius_ }
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere(const Vector3<T>& center_, T radius_) noexcept;

    /**
     * @brief Create a fluent builder for Sphere construction.
     *
     * @return A Sphere<T>::Builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Trivial copy.
    Sphere(const Sphere&) noexcept = default;

    /// @brief Virtual destructor (polymorphic base).
    ~Sphere() override = default;

    /**
     * @brief Create a ray–sphere trace operator for this sphere.
     *
     * @details
     * Provides analytic ray intersection by solving:
     *
     *     ||o + t d - c||² = r²
     *
     * which expands to a quadratic equation in t.
     */

    /**
     * @brief Create a spatial query operator for this sphere.
     *
     * @details
     * Used to dispatch queries such as closest-point, normals,
     * and signed-distance in a polymorphic context.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Closest point on the sphere surface to a query point.
     *
     * @param p Query point p ∈ ℝ³.
     * @return Closest point x* on the surface ∂S.
     *
     * @details
     * For p ≠ center:
     *
     *     x* = center + radius * (p - center) / ||p - center||
     *
     * Special case:
     * - If p = center, the direction is undefined; an arbitrary
     *   outward direction may be chosen deterministically.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Outward unit normal associated with a query point.
     *
     * @param p Query point p ∈ ℝ³.
     * @return Unit normal pointing outward.
     *
     * @details
     * For p ≠ center:
     *
     *     n(p) = (p - center) / ||p - center||
     *
     * On the surface, this matches the geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the sphere.
     *
     * @param p Query point p ∈ ℝ³.
     * @return Signed distance d(p).
     *
     * @details
     * Signed distance function:
     *
     *     d(p) = ||p - center|| - radius
     *
     * Interpretation:
     * - d(p) < 0 : inside the sphere
     * - d(p) = 0 : on the surface
     * - d(p) > 0 : outside the sphere
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the sphere within a tolerance.
     *
     * @param p Query point p ∈ ℝ³.
     * @param tolerance Allowed positive slack beyond the radius.
     * @return `true` if the point is classified as inside the sphere.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the sphere surface within a tolerance band.
     *
     * @param p Query point p ∈ ℝ³.
     * @param tolerance Allowed absolute deviation from the radius.
     * @return `true` if the point is classified as on the sphere surface.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Geometric centroid of the sphere.
     *
     * @return The center of the sphere.
     *
     * @details
     * For a uniform solid sphere, the centroid equals its center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Axis-aligned bounding box (AABB) enclosing the sphere.
     *
     * @return AABB that fully contains the sphere.
     *
     * @details
     * The AABB is:
     *
     *     [center - (r,r,r), center + (r,r,r)]
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Check whether this sphere is valid.
     *
     * @return True if parameters define a valid sphere.
     *
     * @details
     * Typical validity requirements:
     * - radius > 0
     * - center components are finite
     * - radius is finite
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this sphere.
     *
     * @details
     * Identifies the active type in a polymorphic context
     * (e.g., when using `Geometry<T>` pointers).
     *
     * @return `GeometryType::Sphere` for this class.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;
};

/**
 * @brief Fluent builder for Sphere<T>.
 *
 * @details
 * Used to construct `Sphere<T>` instances with explicit parameter setting and
 * validation before creation.
 */
template <typename T>
class Sphere<T>::Builder final {
public:
    /// @brief Construct a builder with default parameters (center=(0,0,0), radius=1).
    Builder() = default;

    /**
     * @brief Build a Sphere<T> value.
     *
     * @return Constructed Sphere<T>.
     *
     * @details
     * Performs validate() prior to construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sphere<T>
    build() const;

    /**
     * @brief Build a host-shared Sphere<T>.
     *
     * @return host_shared_ptr<Sphere<T>> owning the constructed sphere.
     *
     * @details
     * Performs validate() prior to construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sphere<T>>
    make_host_shared() const;

    /**
     * @brief Set the sphere center.
     *
     * @param c Center point.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& c) noexcept;

    /**
     * @brief Set the sphere radius.
     *
     * @param r Radius value (must be positive for a valid sphere).
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T r) noexcept;

private:
    /**
     * @brief Validate builder parameters.
     *
     * @details
     * Expected checks (Atlas policy dependent):
     * - radius > 0
     * - center and radius are finite
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Pending center value used for build().
    Vector3<T> _center { T(0), T(0), T(0) };

    /// @brief Pending radius value used for build().
    T _radius { T(1) };
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for atlas::geometry::Sphere<T>.
 */
template <typename T>
using Sphere = geometry::Sphere<T>;

/**
 * @brief Convenience alias for float sphere.
 */
using SphereF = geometry::Sphere<float>;

/**
 * @brief Convenience alias for double sphere.
 */
using SphereD = geometry::Sphere<double>;

/**
 * @brief Convenience alias for host_shared_ptr<Sphere<T>>.
 */
template <typename T>
using SphereHostPtr = atlas::host_shared_ptr<geometry::Sphere<T>>;

/**
 * @brief Convenience alias for device_shared_ptr<Sphere<T>>.
 */
template <typename T>
using SphereDevicePtr = atlas::device_shared_ptr<geometry::Sphere<T>>;

} // namespace atlas

#include <atlas/geometry/sphere.hpp>
