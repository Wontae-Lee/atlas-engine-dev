#pragma once

/**
 * @file cylinder.h
 * @brief Declares an axis-aligned cylinder geometry primitive and its lightweight query/trace operator.
 *
 * @details
 * This header defines @ref atlas::geometry::Cylinder, a finite right circular
 * cylinder embedded in 3D space and represented by:
 * - a center point,
 * - a radius,
 * - a height.
 *
 * The cylinder is assumed to be axis-aligned with its symmetry axis parallel to
 * the z-axis of the query coordinate frame. Its finite extent is therefore:
 * - radial in the x-y plane, controlled by @ref radius,
 * - axial along z, controlled by @ref height.
 *
 * The header also defines @ref CylinderGeometryOperator, a lightweight non-owning
 * operator object that stores raw pointers to the cylinder parameters and exposes
 * backend-friendly geometric functionality such as:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - ray tracing against the cylinder surface.
 *
 * ## Geometry convention
 * Let:
 * - \f$\mathbf{c}=(c_x,c_y,c_z)\f$ be the cylinder center,
 * - \f$r\f$ be the cylinder radius,
 * - \f$h\f$ be the cylinder height.
 *
 * Then the cylinder is interpreted as the finite solid:
 * \f[
 * (x-c_x)^2 + (y-c_y)^2 \le r^2,
 * \qquad
 * |z-c_z| \le \frac{h}{2}.
 * \f]
 *
 * This corresponds to a right circular cylinder centered at @ref center with
 * flat caps perpendicular to the z-axis.
 *
 * ## Host/device split
 * - The owning @ref Cylinder object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref CylinderGeometryOperator provides a lightweight backend-friendly
 *   view that avoids host-side ownership and virtual dispatch.
 *
 * ## Construction
 * A cylinder may be:
 * - default-constructed,
 * - constructed directly from center, radius, and height,
 * - constructed through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and distances.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Lightweight non-owning geometry operator for querying and tracing a cylinder.
 *
 * @details
 * @ref CylinderGeometryOperator stores raw pointers to the defining parameters of
 * an axis-aligned finite cylinder and exposes geometric query operations that can
 * be used in backend code or host-side hot paths without relying on polymorphic
 * ownership.
 *
 * The operator is intended to be:
 * - cheap to copy,
 * - non-owning,
 * - suitable for device execution,
 * - consistent with the behavior of the owning @ref Cylinder object.
 *
 * ## Stored references
 * The operator references:
 * - @ref center : cylinder center,
 * - @ref radius : cylinder radius,
 * - @ref height : cylinder height,
 * - @ref open : whether the cylinder excludes top/bottom caps from surface queries.
 *
 * Since the stored pointers are non-owning, the referenced data must remain
 * valid for the duration of any use of the operator.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct CylinderGeometryOperator {
    /**
     * @brief Pointer to the cylinder center.
     *
     * @details
     * Non-owning pointer to the world-space center point of the cylinder.
     */
    const atlas::math::Vector<T, 3>* center = nullptr;

    /**
     * @brief Pointer to the cylinder radius.
     *
     * @details
     * Non-owning pointer to the radial extent in the x-y plane.
     */
    const T* radius = nullptr;

    /**
     * @brief Pointer to the cylinder height.
     *
     * @details
     * Non-owning pointer to the axial extent along the z-axis.
     */
    const T* height = nullptr;

    /**
     * @brief Pointer to the open-ended flag.
     *
     * @details
     * When non-null and `true`, only the lateral wall is considered part of the
     * surface. The top and bottom caps are ignored by surface queries and ray
     * tracing.
     */
    const bool* open = nullptr;

    /**
     * @brief Compute the closest point on the finite cylinder to a query point.
     *
     * @details
     * The result may lie on:
     * - the lateral cylindrical surface,
     * - the top cap,
     * - the bottom cap,
     * - or an edge circle where the side meets a cap,
     * depending on the position of the query point.
     *
     * @param p Query point in world space.
     * @return Closest point on the cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest outward normal associated with the cylinder.
     *
     * @details
     * Depending on the closest feature, the normal may correspond to:
     * - the radial outward direction on the side wall,
     * - the positive z direction on the top cap,
     * - the negative z direction on the bottom cap.
     *
     * Edge and axis-degenerate cases may require deterministic tie-breaking.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the cylinder.
     *
     * @details
     * The exact formulation is implementation-defined in `cylinder.hpp`, but the
     * result is intended to represent the shortest distance to the finite cylinder
     * with a consistent sign convention.
     *
     * @param p Query point in world space.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies inside the cylinder within a tolerance.
     *
     * @details
     * This typically checks:
     * - radial inclusion in the x-y plane,
     * - axial inclusion within the capped height range,
     * up to the supplied tolerance.
     *
     * @param p Query point.
     * @param tolerance Non-negative classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the cylinder surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as lying on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return the centroid of the cylinder.
     *
     * @details
     * For a uniformly dense finite cylinder, the centroid coincides with the center.
     *
     * @return Cylinder centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the cylinder.
     *
     * @details
     * Since the cylinder is axis-aligned, its bounding box is directly determined
     * by the center, radius, and half-height.
     *
     * @return Axis-aligned bounding box enclosing the cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the referenced cylinder parameters define a valid cylinder.
     *
     * @details
     * Typical validity checks include:
     * - all referenced pointers are non-null,
     * - center coordinates are finite,
     * - radius is finite and non-negative,
     * - height is finite and non-negative.
     *
     * @return `true` if the operator references a valid cylinder; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the finite cylinder.
     *
     * @details
     * This may involve intersections with:
     * - the lateral cylindrical surface,
     * - the top cap,
     * - the bottom cap,
     * followed by selection of the nearest valid forward hit.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the intersection result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    /**
     * @brief Function-call alias for @ref trace.
     *
     * @param ray Query ray.
     * @return Surface hit record.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

/**
 * @brief Axis-aligned finite cylinder geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref Cylinder represents a right circular cylinder whose axis is aligned with
 * the z-axis of the query coordinate frame.
 *
 * It is defined by:
 * - @ref center : the geometric center of the cylinder,
 * - @ref radius : the radial extent in the x-y plane,
 * - @ref height : the axial extent along z,
 * - @ref open : whether the cylinder excludes top and bottom caps.
 *
 * ## Geometric semantics
 * The cylinder is centered at @ref center and extends:
 * - radially by @ref radius in the x-y plane,
 * - axially by `height / 2` above and below `center.z`.
 *
 * This class provides:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box access,
 * - geometry-type reporting,
 * - creation of a bound lightweight geometry operator.
 *
 * ## Operator caching
 * The class maintains an internal cached @ref CylinderGeometryOperator that stores
 * pointers to the cylinder parameters. This operator is rebound whenever the
 * object is copied, moved, or otherwise reconstructed so that query paths remain valid.
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
     * The builder stages center, radius, and height values, validates them, and
     * constructs either:
     * - a cylinder by value, or
     * - a host-owned shared pointer to a cylinder.
     */
    class Builder;

public:
    /**
     * @brief Center of the cylinder.
     *
     * @details
     * World-space center point of the finite cylinder.
     */
    Vector3<T> center { T(0), T(0), T(0) };

    /**
     * @brief Radius of the cylinder.
     *
     * @details
     * Radial extent in the x-y plane.
     */
    T radius = T(1);

    /**
     * @brief Height of the cylinder.
     *
     * @details
     * Total axial extent along the z-axis.
     */
    T height = T(1);

    /**
     * @brief Whether the cylinder is open-ended.
     *
     * @details
     * When `true`, only the lateral wall is treated as part of the cylinder
     * surface. The top and bottom caps are excluded from closest-point,
     * surface-classification, and ray-trace queries.
     */
    bool open = false;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a unit-radius, unit-height cylinder centered at the origin.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder() noexcept;

    /**
     * @brief Construct a cylinder from explicit center, radius, and height.
     *
     * @param center_ Cylinder center.
     * @param radius_ Cylinder radius.
     * @param height_ Cylinder height.
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

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies geometric parameters and rebinds the cached operator so that its
     * internal pointers reference this object rather than the source object.
     *
     * @param other Source cylinder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(const Cylinder& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * @details
     * Moves geometric parameters and rebinds the cached operator to this object.
     *
     * @param other Source cylinder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(Cylinder&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Copies geometric parameters and refreshes the cached operator binding.
     *
     * @param other Source cylinder.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder&
    operator=(const Cylinder& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @details
     * Moves geometric parameters and refreshes the cached operator binding.
     *
     * @param other Source cylinder.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder&
    operator=(Cylinder&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Cylinder() override = default;

    /**
     * @brief Create a geometry operator bound to this cylinder.
     *
     * @details
     * Returns a lightweight non-owning operator that references this cylinder's
     * geometric parameters.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Compute the closest point on the cylinder to a query point.
     *
     * @param p Query point.
     * @return Closest point on the cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest outward normal associated with the cylinder.
     *
     * @param p Query point.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the cylinder.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the cylinder within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the cylinder surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return the centroid of the cylinder.
     *
     * @return Cylinder centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of the cylinder.
     *
     * @return Axis-aligned bounding box enclosing the cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this cylinder is valid.
     *
     * @details
     * Typical validity checks include:
     * - finite center coordinates,
     * - finite non-negative radius,
     * - finite non-negative height.
     *
     * @return `true` if the cylinder is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a cylinder.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow the builder to configure cylinder internals directly.
    friend class Builder;

    /**
     * @brief Bind the cached operator to this cylinder's storage.
     *
     * @details
     * Refreshes the raw-pointer fields of @ref _operator so that they reference
     * this instance's geometric parameters.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    /**
     * @brief Cached non-owning geometry operator bound to this cylinder.
     *
     * @details
     * Stores raw pointers to @ref center, @ref radius, and @ref height.
     */
    mutable CylinderGeometryOperator<T> _operator {};
};

/**
 * @brief Fluent builder for @ref Cylinder.
 *
 * @details
 * The builder provides a controlled construction path for cylinders by staging
 * center, radius, and height values before validation.
 *
 * ## Typical usage
 * @code
 * auto cylinder = atlas::CylinderF::builder()
 *     .with_center({0.0f, 0.0f, 0.0f})
 *     .with_radius(1.0f)
 *     .with_height(2.0f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the radius is finite and non-negative,
 * - the height is finite and non-negative.
 *
 * The exact validation rules are implementation-defined in `cylinder.hpp`.
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
     * Initializes staged parameters to a cylinder centered at the origin with
     * unit radius and unit height.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Cylinder by value after validation.
     *
     * @return Constructed cylinder value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder<T>
    build() const;

    /**
     * @brief Build a configured @ref Cylinder in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Cylinder<T>>` owning the constructed cylinder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Cylinder<T>>
    make_host_shared() const;

    /**
     * @brief Set the cylinder center.
     *
     * @param center_ Staged center point.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    /**
     * @brief Set the cylinder radius.
     *
     * @param radius_ Staged radius.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T radius_) noexcept;

    /**
     * @brief Set the cylinder height.
     *
     * @param height_ Staged height.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_height(T height_) noexcept;

    /**
     * @brief Set whether the cylinder is open-ended.
     *
     * @param open_ Whether top and bottom caps should be excluded.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_open(bool open_) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on center, radius, and height values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending center point.
     */
    Vector3<T> _center { T(0), T(0), T(0) };

    /**
     * @brief Pending radius.
     */
    T _radius = T(1);

    /**
     * @brief Pending height.
     */
    T _height = T(1);

    /**
     * @brief Pending open-ended flag.
     */
    bool _open = false;
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::geometry::Cylinder.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Cylinder = geometry::Cylinder<T>;

/**
 * @brief Common specialization of @ref atlas::geometry::Cylinder for `float`.
 */
using CylinderF = geometry::Cylinder<float>;

/**
 * @brief Common specialization of @ref atlas::geometry::Cylinder for `double`.
 */
using CylinderD = geometry::Cylinder<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::geometry::Cylinder.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CylinderHostPtr = atlas::host_shared_ptr<geometry::Cylinder<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::geometry::Cylinder.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CylinderDevicePtr = atlas::device_shared_ptr<geometry::Cylinder<T>>;

} // namespace atlas

#include <atlas/geometry/cylinder.hpp>
