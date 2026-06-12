#pragma once

/**
 * @file circle.h
 * @brief Declares an oriented circle geometry primitive and its lightweight query/trace operator.
 *
 * @details
 * This header defines @ref atlas::Circle, a planar circular geometry
 * embedded in 3D space and represented by:
 * - a center point,
 * - an orientation normal,
 * - a scalar radius.
 *
 * The circle is interpreted as a finite disk-like support surface lying in the
 * plane defined by the center and normal, with geometric queries restricted to
 * the circular region of radius @ref radius.
 *
 * The header also defines @ref CircleGeometryOperator, a lightweight non-owning
 * operator object that stores raw pointers to the circle parameters and exposes
 * backend-friendly query functionality such as:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - ray tracing against the circle surface.
 *
 * ## Geometry interpretation
 * Let:
 * - \f$\mathbf{c}\f$ be the circle center,
 * - \f$\mathbf{n}\f$ be the circle normal,
 * - \f$r\f$ be the circle radius.
 *
 * Then the circle lies in the plane:
 * \f[
 * (\mathbf{x} - \mathbf{c}) \cdot \mathbf{n} = 0
 * \f]
 * and includes only points whose in-plane radial distance to \f$\mathbf{c}\f$
 * is at most \f$r\f$.
 *
 * ## Host/device split
 * - The owning @ref Circle object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref CircleGeometryOperator provides a lightweight device-friendly view
 *   that avoids host-side ownership and virtual dispatch.
 *
 * ## Construction
 * A circle may be:
 * - default-constructed,
 * - constructed directly from center, normal, and radius,
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
 * @brief Lightweight non-owning geometry operator for querying and tracing a circle.
 *
 * @details
 * @ref CircleGeometryOperator stores raw pointers to the defining parameters of
 * a circle and exposes query operations that can be used in backend code or in
 * host-side hot paths without relying on polymorphic ownership.
 *
 * The operator is intended to be:
 * - cheap to copy,
 * - non-owning,
 * - suitable for device execution,
 * - consistent with the behavior of the owning @ref Circle object.
 *
 * ## Stored references
 * The operator references:
 * - @ref center : circle center,
 * - @ref normal : circle plane normal,
 * - @ref radius : circle radius.
 *
 * Since the stored pointers are non-owning, the referenced data must remain
 * valid for the duration of any use of the operator.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct CircleGeometryOperator {
    /**
     * @brief Pointer to the circle center.
     *
     * @details
     * Non-owning pointer to the world-space center point of the circle.
     */
    const atlas::Vector<T, 3>* center = nullptr;

    /**
     * @brief Pointer to the circle normal.
     *
     * @details
     * Non-owning pointer to the world-space orientation normal of the circle plane.
     */
    const atlas::Vector<T, 3>* normal = nullptr;

    /**
     * @brief Pointer to the circle radius.
     *
     * @details
     * Non-owning pointer to the scalar radius of the circle.
     */
    const T* radius = nullptr;

    /**
     * @brief Compute the closest point on the circle to a query point.
     *
     * @details
     * This function projects the query point onto the supporting plane of the
     * circle and clamps the in-plane radial component to the circle boundary as
     * required by the finite radius.
     *
     * @param p Query point in world space.
     * @return Closest point on the circle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest normal associated with the circle at a query point.
     *
     * @details
     * For planar geometries, the closest normal is typically aligned with the
     * circle plane normal up to sign or implementation-defined orientation rules.
     * Edge and center cases may require deterministic tie-breaking.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the circle.
     *
     * @details
     * The exact sign convention and handling of in-plane versus out-of-plane
     * distance components are implementation-defined in `circle.hpp`, but the
     * result is intended to represent the shortest distance to the finite circle.
     *
     * @param p Query point in world space.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies inside the finite circle within a tolerance.
     *
     * @details
     * The precise classification semantics are implementation-defined, but this
     * generally checks whether the point lies on or near the circle-supporting
     * disk region up to the supplied tolerance.
     *
     * @param p Query point.
     * @param tolerance Non-negative classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the circle surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as lying on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return the centroid of the circle.
     *
     * @details
     * For a planar circle, the centroid coincides with the center.
     *
     * @return Circle centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the circle.
     *
     * @details
     * The bounding box depends on the circle center, orientation, and radius.
     *
     * @return Axis-aligned bounding box enclosing the circle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the referenced circle parameters define a valid circle.
     *
     * @details
     * Typical validity checks include:
     * - all referenced pointers are non-null,
     * - the normal is finite and non-degenerate,
     * - the radius is finite and non-negative.
     *
     * @return `true` if the operator references a valid circle; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the circle.
     *
     * @details
     * Intersects the ray with the supporting plane and then checks whether the
     * hit point lies within the finite circle radius.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the intersection result.
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
 * @brief Oriented finite circle geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref Circle represents a finite circular surface embedded in 3D space.
 *
 * It is defined by:
 * - @ref center : the geometric center of the circle,
 * - @ref normal : the orientation normal of the supporting plane,
 * - @ref radius : the circle radius.
 *
 * ## Geometric semantics
 * The circle lies in the plane orthogonal to @ref normal and centered at
 * @ref center. Its finite extent is determined by @ref radius.
 *
 * This class provides:
 * - closest-point queries,
 * - normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - geometry-type reporting,
 * - creation of a bound lightweight geometry operator.
 *
 * ## Operator caching
 * The class maintains an internal cached @ref CircleGeometryOperator that stores
 * pointers to the circle parameters. This operator is rebound whenever the object
 * is copied, moved, or otherwise reconstructed so that query paths stay valid.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Circle final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Circle requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Circle.
     *
     * @details
     * The builder stages center, normal, and radius values, validates them, and
     * constructs either:
     * - a circle by value, or
     * - a host-owned shared pointer to a circle.
     */
    class Builder;

public:
    /**
     * @brief Center of the circle.
     *
     * @details
     * World-space center point of the circular surface.
     */
    Vector3<T> center { T(0), T(0), T(0) };

    /**
     * @brief Orientation normal of the circle plane.
     *
     * @details
     * World-space normal defining the plane in which the circle lies.
     *
     * @note
     * A valid circle generally requires this vector to be finite and non-zero.
     */
    Vector3<T> normal { T(0), T(0), T(1) };

    /**
     * @brief Radius of the circle.
     *
     * @details
     * Finite radial extent of the circular surface.
     */
    T radius { T(1) };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a unit circle centered at the origin in the plane whose normal
     * is the positive z-axis.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle() noexcept;

    /**
     * @brief Construct a circle from explicit center, normal, and radius.
     *
     * @param center_ Circle center.
     * @param normal_ Circle plane normal.
     * @param radius_ Circle radius.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(const Vector3<T>& center_, const Vector3<T>& normal_, T radius_) noexcept;

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
     * @param other Source circle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(const Circle& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * @param other Source circle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(Circle&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @param other Source circle.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Circle&
    operator=(const Circle& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @param other Source circle.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Circle&
    operator=(Circle&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Circle() override = default;

    /**
     * @brief Create a geometry operator bound to this circle.
     *
     * @details
     * Returns a lightweight non-owning operator that references this circle's
     * geometric parameters.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    /**
     * @brief Compute the closest point on the circle to a query point.
     *
     * @param p Query point.
     * @return Closest point on the circle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest normal associated with the circle.
     *
     * @param p Query point.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the circle.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the circle within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the circle surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return the centroid of the circle.
     *
     * @return Circle centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of the circle.
     *
     * @return Axis-aligned bounding box enclosing the circle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this circle is valid.
     *
     * @details
     * Typical validity checks include:
     * - finite center coordinates,
     * - finite and non-degenerate normal,
     * - finite and non-negative radius.
     *
     * @return `true` if the circle is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a circle.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow the builder to configure circle internals directly.
    friend class Builder;

    /**
     * @brief Creates a lightweight runtime operator bound to this circle's current parameters.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE CircleGeometryOperator<T>
    make_circle_operator() const noexcept;
};

/**
 * @brief Fluent builder for @ref Circle.
 *
 * @details
 * The builder provides a controlled construction path for circles by staging
 * center, normal, and radius values before validation.
 *
 * ## Typical usage
 * @code
 * auto circle = atlas::CircleF::builder()
 *     .with_center({0.0f, 0.0f, 0.0f})
 *     .with_normal({0.0f, 0.0f, 1.0f})
 *     .with_radius(2.0f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the normal is finite and non-zero,
 * - the radius is finite,
 * - the radius is non-negative.
 *
 * The exact validation rules are implementation-defined in `circle.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Circle<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged parameters to a unit circle centered at the origin with
     * the positive z-axis as its normal.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Circle by value after validation.
     *
     * @return Constructed circle value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Circle<T>
    build() const;

    /**
     * @brief Build a configured @ref Circle in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Circle<T>>` owning the constructed circle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Circle<T>>
    make_host_shared() const;

    /**
     * @brief Set the circle center.
     *
     * @param center_ Staged center point.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    /**
     * @brief Set the circle plane normal.
     *
     * @param normal_ Staged normal vector.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    /**
     * @brief Set the circle radius.
     *
     * @param radius_ Staged radius.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T radius_) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on center, normal, and radius values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending center point.
     */
    Vector3<T> _center { T(0), T(0), T(0) };

    /**
     * @brief Pending normal vector.
     */
    Vector3<T> _normal { T(0), T(0), T(1) };

    /**
     * @brief Pending radius.
     */
    T _radius { T(1) };
};

} // namespace atlas

namespace atlas {


/**
 * @brief Common specialization of @ref atlas::Circle for `float`.
 */
using CircleF = Circle<float>;

/**
 * @brief Common specialization of @ref atlas::Circle for `double`.
 */
using CircleD = Circle<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::Circle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CircleHostPtr = atlas::host_shared_ptr<Circle<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::Circle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using CircleDevicePtr = atlas::device_shared_ptr<Circle<T>>;

} // namespace atlas

#include <atlas/geometry/circle.hpp>
