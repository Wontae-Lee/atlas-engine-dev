#pragma once

/**
 * @file cylinder.h
 * @brief Finite cylinder geometry primitive (axis-aligned in local space) and builder.
 *
 * @details
 * This header defines @ref atlas::geometry::Cylinder, a finite cylinder geometry described by:
 * - @ref center : cylinder center position (in the cylinder's local frame)
 * - @ref radius : cylinder radius
 * - @ref height : cylinder height
 *
 * The cylinder in this API is a **finite** cylinder with its axis aligned to a fixed axis
 * in its local coordinate system (implementation-defined; typically the Z axis).
 *
 * `Cylinder` implements the @ref atlas::Geometry interface and provides:
 * - query and trace operator construction for interop with spatial systems,
 * - closest-point and closest-normal evaluation,
 * - signed distance evaluation,
 * - centroid and AABB computation,
 * - validity checks,
 * - a host-side parameter setter for reconfiguration.
 *
 * ## Coordinate convention (important)
 * This cylinder is axis-aligned in its own coordinate frame. A common convention is:
 * - cylinder axis is the **Z axis**
 * - end caps lie at `z = center.z ± height/2`
 * - radial distance is measured in the XY plane from `(center.x, center.y)`
 *
 * If you need an arbitrarily oriented cylinder in world space, combine this geometry with
 * a transform/sync mechanism (e.g., @ref atlas::system::Sync).
 *
 * ## Construction
 * `Cylinder` can be default-constructed (center at origin, radius=1, height=1) or configured
 * via the nested fluent @ref Builder which:
 * - stages center/radius/height,
 * - validates parameters (positive radius/height, finite values),
 * - can build by value or as a `host_shared_ptr`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * - Most geometry evaluations assume a valid cylinder (radius > 0, height > 0).
 * - If invalid, results are implementation-defined; use @ref is_valid() to check.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/spatial/trace_operator.h>

#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Finite cylinder geometry primitive (axis-aligned in local space).
 *
 * @details
 * `Cylinder` represents a finite cylinder described by center, radius, and height.
 *
 * ### Typical interpretation
 * Let the cylinder axis be aligned to the Z axis (common convention):
 * - The cylinder's longitudinal interval is:
 *   \f[
 *   z \in [c_z - \tfrac{h}{2},\; c_z + \tfrac{h}{2}]
 *   \f]
 * - The radial constraint is:
 *   \f[
 *   (x-c_x)^2 + (y-c_y)^2 \le r^2
 *   \f]
 *
 * Points on the surface are either:
 * - on the side wall (radial constraint active, z within interval), or
 * - on the end caps (z clamped to an end plane, radial inside disk).
 *
 * ### Closest point / normal / signed distance
 * The exact details are implementation-defined, but typical behavior is:
 * - Outside: closest point lies on side wall or caps depending on which constraint is violated most.
 * - Inside: closest point on the surface is obtained by pushing to nearest boundary (cap or wall).
 * - Signed distance uses a standard finite-cylinder SDF; confirm sign convention in implementation.
 *
 * ### Host/device
 * - Evaluation methods are `ATLAS_ALL_DEVICE`.
 * - Operator construction is host-only, typically binding pointers to member data.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Cylinder final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Cylinder requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Cylinder.
     *
     * @details
     * See @ref Cylinder<T>::Builder for configuration methods and validation policy.
     */
    class Builder;

public:
    /**
     * @brief Cylinder center position.
     *
     * @details
     * Defines the center of the cylinder in the coordinate frame in which the cylinder is queried.
     * If using the common Z-axis convention, `center.z` is the midpoint between the end caps.
     */
    Vector3<T> center { T(0), T(0), T(0) };

    /**
     * @brief Cylinder radius.
     *
     * @details
     * Must be positive for a valid cylinder.
     */
    T radius = T(1);

    /**
     * @brief Cylinder height.
     *
     * @details
     * The finite extent along the cylinder axis. Must be positive for a valid cylinder.
     */
    T height = T(1);

    /**
     * @brief Default constructor (unit cylinder).
     *
     * @details
     * Initializes:
     * - center = (0,0,0)
     * - radius = 1
     * - height = 1
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder() noexcept;

    /**
     * @brief Construct a cylinder from center, radius, and height.
     *
     * @param center_ Cylinder center.
     * @param radius_ Cylinder radius (should be > 0).
     * @param height_ Cylinder height (should be > 0).
     *
     * @note
     * No automatic validation or clamping is implied by the declaration; use @ref is_valid()
     * or the @ref Builder for validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Defaulted copy constructor.
    Cylinder(const Cylinder&) noexcept = default;

    /// @brief Virtual destructor (geometry base).
    ~Cylinder() override = default;

    /**
     * @brief Create a trace operator bound to this cylinder.
     *
     * @details
     * Returns a @ref TraceOperator that can be used by tracing systems to intersect rays
     * with this cylinder. Implementations commonly store non-owning pointers to `center`,
     * `radius`, and `height`.
     *
     * @return Trace operator referencing this cylinder.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TraceOperator<T>
    make_trace_operator() const override;

    /**
     * @brief Create a query operator bound to this cylinder.
     *
     * @details
     * Returns a @ref QueryOperator that can be used by query systems to compute closest points,
     * signed distances, etc., against this cylinder.
     *
     * @return Query operator referencing this cylinder.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE QueryOperator<T>
    make_query_operator() const override;

    /**
     * @brief Compute the closest point on (or in) the cylinder to an input point.
     *
     * @details
     * Typical behavior:
     * - Outside: returns the closest point on the cylinder surface (side wall or cap).
     * - Inside: may return the point itself (volume-closest) or the closest surface point,
     *   depending on implementation policy.
     *
     * @param p Query point.
     * @return Closest point according to the implementation policy.
     *
     * @note
     * If you require a specific inside behavior (surface projection), confirm the implementation
     * in `cylinder.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the outward normal corresponding to the closest feature to a point.
     *
     * @details
     * Typical behavior:
     * - Side wall: radial normal in the XY plane.
     * - End caps: ±axis normal (e.g., ±Z) depending on which cap is closest.
     * - Edge ring (where wall meets cap): ambiguous; tie-breaking is implementation-defined.
     *
     * @param p Query point.
     * @return A (typically unit-length) normal vector.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the cylinder.
     *
     * @details
     * Common convention:
     * - positive outside,
     * - zero on the surface,
     * - negative inside.
     *
     * A typical finite-cylinder SDF combines:
     * - radial distance to the infinite cylinder, and
     * - axial distance to the cap planes,
     * with appropriate clamping and inside handling.
     *
     * @param p Query point.
     * @return Signed distance.
     *
     * @note Confirm the exact sign convention and formulation in `cylinder.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the cylinder within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Allowed positive slack beyond the finite cylinder boundary.
     * @return `true` if the point is classified as inside.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the cylinder surface within a tolerance band.
     *
     * @param p Query point.
     * @param tolerance Allowed absolute deviation from the surface.
     * @return `true` if the point is classified as on the surface.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return the centroid of the cylinder.
     *
     * @details
     * For a symmetric cylinder, the centroid is its @ref center.
     *
     * @return The center of the cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of this cylinder.
     *
     * @details
     * For an axis-aligned cylinder (typical Z-axis convention), the AABB is:
     * - x: [center.x - radius, center.x + radius]
     * - y: [center.y - radius, center.y + radius]
     * - z: [center.z - height/2, center.z + height/2]
     *
     * If your implementation uses a different axis convention, update accordingly.
     *
     * @return AABB in the same coordinate frame.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether this cylinder is valid.
     *
     * @details
     * A valid cylinder typically requires:
     * - finite center components,
     * - `radius > 0`,
     * - `height > 0`.
     *
     * @return `true` if parameters are well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this cylinder.
     *
     * @details
     * This identifies the active type in a polymorphic context (e.g., when using `Geometry<T>` pointers).
     *
     * @return `GeometryType::Cylinder` for this class.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow builder to configure internals.
    friend class Builder;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

/**
 * @brief Fluent builder for @ref Cylinder.
 *
 * @details
 * The builder stages cylinder parameters and validates them before producing a `Cylinder`.
 *
 * ## Typical usage
 * @code
 * atlas::CylinderF cyl = atlas::CylinderF::builder()
 *     .with_center({0,0,0})
 *     .with_radius(0.5f)
 *     .with_height(2.0f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks include:
 * - `radius > 0`
 * - `height > 0`
 * - finite values for center/radius/height
 *
 * The exact behavior is implemented in `cylinder.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Cylinder<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged values to a unit cylinder centered at the origin.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Cylinder (by value).
     *
     * @details
     * Validates parameters and constructs a @ref Cylinder.
     *
     * @return Constructed cylinder by value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder<T>
    build() const;

    /**
     * @brief Build a configured @ref Cylinder in a host_shared_ptr.
     *
     * @return `atlas::host_shared_ptr<Cylinder<T>>` owning the constructed cylinder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Cylinder<T>>
    make_host_shared() const;

    /**
     * @brief Set cylinder center.
     *
     * @param center_ Center position.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    /**
     * @brief Set cylinder radius.
     *
     * @param radius_ Radius (should be > 0).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T radius_) noexcept;

    /**
     * @brief Set cylinder height.
     *
     * @param height_ Height (should be > 0).
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_height(T height_) noexcept;

private:
    /**
     * @brief Validate staged parameters.
     *
     * @details
     * Expected checks:
     * - radius and height are positive
     * - all components are finite
     *
     * @note Implementation may `throw`, `assert`, or no-op depending on Atlas policy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Staged center.
    Vector3<T> _center { T(0), T(0), T(0) };

    /// @brief Staged radius.
    T _radius = T(1);

    /// @brief Staged height.
    T _height = T(1);
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for `atlas::geometry::Cylinder<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Cylinder = geometry::Cylinder<T>;

/// @brief Common specialization: float cylinder.
using CylinderF = geometry::Cylinder<float>;

/// @brief Common specialization: double cylinder.
using CylinderD = geometry::Cylinder<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to `atlas::geometry::Cylinder<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CylinderHostPtr = atlas::host_shared_ptr<geometry::Cylinder<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to `atlas::geometry::Cylinder<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CylinderDevicePtr = atlas::device_shared_ptr<geometry::Cylinder<T>>;

} // namespace atlas

#include <atlas/geometry/cylinder.hpp>
