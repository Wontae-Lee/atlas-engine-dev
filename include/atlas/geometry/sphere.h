#pragma once

/**
 * @file sphere.h
 * @brief Declares a sphere geometry primitive and its lightweight runtime query/trace operator.
 *
 * @details
 * This header defines @ref atlas::Sphere, a 3D sphere primitive
 * represented by:
 * - a center point,
 * - a scalar radius.
 *
 * It also defines @ref SphereGeometryOperator, a lightweight non-owning operator
 * object that stores raw pointers to the sphere parameters and exposes
 * backend-friendly query functionality such as:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside and surface classification,
 * - centroid and bounding-box queries,
 * - ray tracing against the sphere surface.
 *
 * ## Geometry interpretation
 * Let:
 * - \f$\mathbf{c}\f$ be the sphere center,
 * - \f$r\f$ be the sphere radius.
 *
 * Then the sphere is the set of points satisfying:
 * \f[
 * \|\mathbf{x} - \mathbf{c}\| \le r
 * \f]
 * for the solid interior interpretation, with the surface defined by:
 * \f[
 * \|\mathbf{x} - \mathbf{c}\| = r.
 * \f]
 *
 * ## Host/device split
 * - The owning @ref Sphere object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref SphereGeometryOperator provides a lightweight device-friendly view
 *   that avoids host-side ownership and virtual dispatch.
 *
 * ## Construction
 * A sphere may be:
 * - default-constructed,
 * - constructed directly from center and radius,
 * - constructed through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and distances.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas {

/**
 * @brief Lightweight non-owning runtime query operator for a sphere.
 *
 * @details
 * @ref SphereGeometryOperator stores raw pointers to the defining parameters of
 * a sphere and exposes query operations suitable for host or device execution
 * without relying on polymorphic ownership.
 *
 * The operator is intended to be:
 * - cheap to copy,
 * - non-owning,
 * - suitable for device execution,
 * - consistent with the behavior of the owning @ref Sphere object.
 *
 * ## Stored references
 * The operator references:
 * - @ref center : sphere center,
 * - @ref radius : sphere radius.
 *
 * Since the stored pointers are non-owning, the referenced data must remain
 * valid for the duration of any use of the operator.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SphereGeometryOperator {
    /**
     * @brief Pointer to the sphere center.
     *
     * @details
     * Non-owning pointer to the world-space center point of the sphere.
     */
    const atlas::Vector<T, 3>* center = nullptr;

    /**
     * @brief Pointer to the sphere radius.
     *
     * @details
     * Non-owning pointer to the scalar radius of the sphere.
     */
    const T* radius = nullptr;

    /**
     * @brief Compute the closest point on the sphere to a query point.
     *
     * @details
     * For points outside the sphere, this typically projects the query point
     * radially onto the sphere surface. For points inside the sphere, the exact
     * behavior is implementation-defined but commonly returns the nearest surface
     * point along the radial direction or another deterministic rule for
     * degenerate center cases.
     *
     * @param p Query point in world space.
     * @return Closest point on the sphere.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest outward normal associated with the sphere.
     *
     * @details
     * The normal is typically the normalized radial direction from the center to
     * the closest point on the sphere surface.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the sphere.
     *
     * @details
     * A common formulation is:
     * \f[
     * \|\mathbf{p} - \mathbf{c}\| - r
     * \f]
     * which is:
     * - positive outside the sphere,
     * - zero on the surface,
     * - negative inside.
     *
     * @param p Query point in world space.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies inside the sphere within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Non-negative classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the sphere surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as lying on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return the centroid of the sphere.
     *
     * @details
     * For a sphere, the centroid coincides with the center.
     *
     * @return Sphere centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the sphere.
     *
     * @details
     * The bounding box is centered at @ref center and extends by @ref radius
     * along each Cartesian axis.
     *
     * @return Axis-aligned bounding box enclosing the sphere.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the referenced sphere parameters define a valid sphere.
     *
     * @details
     * Typical validity checks include:
     * - all referenced pointers are non-null,
     * - center coordinates are finite,
     * - radius is finite and non-negative.
     *
     * @return `true` if the operator references a valid sphere; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the sphere.
     *
     * @details
     * Intersects the ray with the sphere and typically returns the nearest valid
     * forward hit, if one exists.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the ray-sphere intersection result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::Ray<T>& ray) const noexcept;

    /**
     * @brief Function-call alias for @ref trace.
     *
     * @param ray Query ray.
     * @return Surface hit record.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::Ray<T>& ray) const noexcept;
};

/**
 * @brief Sphere geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref Sphere represents a sphere in 3D space.
 *
 * It is defined by:
 * - @ref center : the geometric center of the sphere,
 * - @ref radius : the radial extent from the center.
 *
 * This class provides:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - geometry-type reporting,
 * - creation of a bound lightweight geometry operator.
 *
 * ## Operator caching
 * The class maintains an internal cached @ref SphereGeometryOperator that stores
 * pointers to the sphere parameters. This operator is rebound whenever the
 * object is copied, moved, or otherwise reconstructed so that query paths remain valid.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Sphere final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Sphere requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Sphere.
     *
     * @details
     * The builder stages center and radius values, validates them, and constructs
     * either:
     * - a sphere by value, or
     * - a host-owned shared pointer to a sphere.
     */
    class Builder;

public:
    /**
     * @brief Sphere center.
     *
     * @details
     * World-space center point of the sphere.
     */
    Vector3<T> center { T(0), T(0), T(0) };

    /**
     * @brief Sphere radius.
     *
     * @details
     * Scalar radial extent of the sphere.
     */
    T radius { T(1) };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a unit sphere centered at the origin.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere() noexcept;

    /**
     * @brief Construct a sphere from explicit center and radius.
     *
     * @param center_ Sphere center.
     * @param radius_ Sphere radius.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere(const Vector3<T>& center_, T radius_) noexcept;

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
     * @param other Source sphere.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere(const Sphere& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * @param other Source sphere.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere(Sphere&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @param other Source sphere.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sphere&
    operator=(const Sphere& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @param other Source sphere.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sphere&
    operator=(Sphere&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Sphere() override = default;

    /**
     * @brief Create a geometry operator bound to this sphere.
     *
     * @details
     * Returns a lightweight non-owning operator that references this sphere's
     * geometric parameters.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    /**
     * @brief Compute the closest point on the sphere to a query point.
     *
     * @param p Query point.
     * @return Closest point on the sphere.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest outward normal associated with the sphere.
     *
     * @param p Query point.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the sphere.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the sphere within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the sphere surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return the centroid of the sphere.
     *
     * @return Sphere centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of the sphere.
     *
     * @return Axis-aligned bounding box enclosing the sphere.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this sphere is valid.
     *
     * @details
     * Typical validity checks include:
     * - finite center coordinates,
     * - finite non-negative radius.
     *
     * @return `true` if the sphere is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a sphere.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow the builder to configure sphere internals directly.
    friend class Builder;

    /**
     * @brief Creates a lightweight runtime operator bound to this sphere's current parameters.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SphereGeometryOperator<T>
    make_sphere_operator() const noexcept;
};

/**
 * @brief Fluent builder for @ref Sphere.
 *
 * @details
 * The builder provides a controlled construction path for spheres by staging
 * center and radius values before validation.
 *
 * ## Typical usage
 * @code
 * auto sphere = atlas::SphereF::builder()
 *     .with_center({0.0f, 0.0f, 0.0f})
 *     .with_radius(1.5f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the radius is finite,
 * - the radius is non-negative,
 * - the center coordinates are finite.
 *
 * The exact validation rules are implementation-defined in `sphere.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Sphere<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged parameters to a unit sphere centered at the origin.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Sphere by value after validation.
     *
     * @return Constructed sphere value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Sphere<T>
    build() const;

    /**
     * @brief Build a configured @ref Sphere in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Sphere<T>>` owning the constructed sphere.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sphere<T>>
    make_host_shared() const;

    /**
     * @brief Set the sphere center.
     *
     * @param c Staged center point.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& c) noexcept;

    /**
     * @brief Set the sphere radius.
     *
     * @param r Staged radius.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T r) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged center and radius values.
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
    T _radius { T(1) };
};

} // namespace atlas

namespace atlas {


/**
 * @brief Common specialization of @ref atlas::Sphere for `float`.
 */
using SphereF = Sphere<float>;

/**
 * @brief Common specialization of @ref atlas::Sphere for `double`.
 */
using SphereD = Sphere<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::Sphere.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphereHostPtr = atlas::host_shared_ptr<Sphere<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::Sphere.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SphereDevicePtr = atlas::device_shared_ptr<Sphere<T>>;

} // namespace atlas

#include <atlas/geometry/sphere.hpp>
