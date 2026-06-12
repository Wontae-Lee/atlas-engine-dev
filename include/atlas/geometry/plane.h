#pragma once

/**
 * @file plane.h
 * @brief Declares an infinite plane geometry primitive and its lightweight query/trace operator.
 *
 * @details
 * This header defines @ref atlas::Plane, an infinite plane embedded in
 * 3D space and represented in implicit form by:
 * - a normal vector,
 * - a scalar offset.
 *
 * The header also defines @ref PlaneGeometryOperator, a lightweight non-owning
 * operator object that stores raw pointers to the plane parameters and exposes
 * backend-friendly geometric functionality such as:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - half-space and surface classification,
 * - centroid and bounding-volume queries,
 * - ray tracing against the plane.
 *
 * ## Plane representation
 * The plane is typically interpreted in implicit form as:
 * \f[
 * \mathbf{n} \cdot \mathbf{x} = d
 * \f]
 * where:
 * - \f$\mathbf{n}\f$ is the plane normal,
 * - \f$d\f$ is the offset,
 * - \f$\mathbf{x}\f$ is a point lying on the plane.
 *
 * Equivalently, the signed implicit function can be written as:
 * \f[
 * \phi(\mathbf{x}) = \mathbf{n} \cdot \mathbf{x} - d.
 * \f]
 *
 * ## Construction paths
 * A plane may be:
 * - default-constructed,
 * - constructed directly from a normal and offset,
 * - constructed from a point-normal representation,
 * - constructed through the nested fluent @ref Builder.
 *
 * ## Host/device split
 * - The owning @ref Plane object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref PlaneGeometryOperator provides a lightweight backend-friendly view
 *   that avoids host-side ownership and virtual dispatch.
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
 * @brief Lightweight non-owning geometry operator for querying and tracing a plane.
 *
 * @details
 * @ref PlaneGeometryOperator stores raw pointers to the defining parameters of
 * an infinite plane and exposes geometric query operations suitable for host or
 * device execution without relying on polymorphic ownership.
 *
 * The operator is intended to be:
 * - cheap to copy,
 * - non-owning,
 * - suitable for device execution,
 * - consistent with the behavior of the owning @ref Plane object.
 *
 * ## Stored references
 * The operator references:
 * - @ref normal : plane normal vector,
 * - @ref offset : scalar plane offset.
 *
 * Since the stored pointers are non-owning, the referenced data must remain
 * valid for the duration of any use of the operator.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct PlaneGeometryOperator {
    /**
     * @brief Pointer to the plane normal.
     *
     * @details
     * Non-owning pointer to the normal vector defining the plane orientation.
     */
    const atlas::Vector<T, 3>* normal = nullptr;

    /**
     * @brief Pointer to the scalar plane offset.
     *
     * @details
     * Non-owning pointer to the offset parameter in the plane equation.
     */
    const T* offset = nullptr;

    /**
     * @brief Compute the closest point on the plane to a query point.
     *
     * @details
     * The closest point is obtained by orthogonally projecting the query point
     * onto the infinite plane.
     *
     * @param p Query point in world space.
     * @return Orthogonal projection of @p p onto the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the plane normal associated with a query point.
     *
     * @details
     * For an infinite plane, the closest normal is typically the plane normal
     * itself up to implementation-defined orientation and normalization rules.
     *
     * @param p Query point in world space.
     * @return Plane normal associated with the query.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the plane.
     *
     * @details
     * The sign convention is determined by the orientation of @ref normal.
     * Positive and negative values correspond to opposite half-spaces.
     *
     * @param p Query point in world space.
     * @return Signed distance from @p p to the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies in the "inside" half-space of the plane.
     *
     * @details
     * For an infinite plane, "inside" is typically interpreted as one of the
     * two half-spaces defined by the plane equation and the orientation of the
     * normal. The exact convention is implementation-defined.
     *
     * @param p Query point.
     * @param tolerance Non-negative classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the plane surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as lying on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return a representative centroid of the plane.
     *
     * @details
     * Since an infinite plane has no finite centroid in the strict geometric
     * sense, implementations typically return a representative point lying on
     * the plane, often the point nearest to the origin.
     *
     * @return Representative point on the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the plane.
     *
     * @details
     * Since the plane is infinite, the returned bounding box may be unbounded
     * or represented using implementation-defined sentinel values.
     *
     * @return Axis-aligned bounding representation of the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the referenced plane parameters define a valid plane.
     *
     * @details
     * Typical validity checks include:
     * - all referenced pointers are non-null,
     * - the normal is finite and non-degenerate,
     * - the offset is finite.
     *
     * @return `true` if the operator references a valid plane; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the plane.
     *
     * @details
     * Computes the intersection between the ray and the infinite plane,
     * typically returning the nearest forward hit if one exists.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the ray-plane intersection result.
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
 * @brief Infinite plane geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref Plane represents an infinite plane in 3D space.
 *
 * It is defined by:
 * - @ref normal : the orientation vector of the plane,
 * - @ref offset : the scalar offset in the implicit plane equation.
 *
 * ## Geometric semantics
 * The plane is commonly interpreted as:
 * \f[
 * \mathbf{n} \cdot \mathbf{x} = d
 * \f]
 * where:
 * - \f$\mathbf{n}\f$ is @ref normal,
 * - \f$d\f$ is @ref offset.
 *
 * The class provides:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - half-space and surface classification,
 * - representative centroid and bounding queries,
 * - geometry-type reporting,
 * - creation of a bound lightweight geometry operator.
 *
 * ## Operator caching
 * The class maintains an internal cached @ref PlaneGeometryOperator that stores
 * pointers to the plane parameters. This operator is rebound whenever the object
 * is copied, moved, or otherwise reconstructed so that query paths remain valid.
 *
 * ## Point-normal construction
 * In addition to the implicit normal-offset representation, the class can be
 * constructed from:
 * - a point lying on the plane,
 * - a normal vector,
 * from which the corresponding offset is derived.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Plane final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Plane requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Plane.
     *
     * @details
     * The builder stages a plane definition through normal/offset or point/normal
     * data, validates it, and constructs either:
     * - a plane by value, or
     * - a host-owned shared pointer to a plane.
     */
    class Builder;

public:
    /**
     * @brief Plane normal vector.
     *
     * @details
     * Defines the orientation of the plane and the sign convention of the signed
     * distance and half-space classification.
     */
    Vector3<T> normal { T(0), T(0), T(1) };

    /**
     * @brief Scalar plane offset.
     *
     * @details
     * Used together with @ref normal in the implicit plane equation.
     */
    T offset { T(0) };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs the plane:
     * \f[
     * z = 0
     * \f]
     * with normal `(0,0,1)` and offset `0`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane() noexcept;

    /**
     * @brief Construct a plane from explicit normal and offset.
     *
     * @param normal_ Plane normal vector.
     * @param offset_ Plane offset.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Vector3<T>& normal_, T offset_) noexcept;

    /**
     * @brief Construct a plane from a point-normal representation.
     *
     * @details
     * Derives the implicit offset from the supplied point and normal.
     *
     * @param point Point lying on the plane.
     * @param normal_ Plane normal vector.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;

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
     * @param other Source plane.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Plane& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * @param other Source plane.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(Plane&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @param other Source plane.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Plane&
    operator=(const Plane& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @param other Source plane.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Plane&
    operator=(Plane&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Plane() override = default;

    /**
     * @brief Create a geometry operator bound to this plane.
     *
     * @details
     * Returns a lightweight non-owning operator that references this plane's
     * defining parameters.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    /**
     * @brief Compute the closest point on the plane to a query point.
     *
     * @param p Query point.
     * @return Orthogonal projection onto the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Return the closest plane normal associated with a query point.
     *
     * @param p Query point.
     * @return Plane normal associated with the query.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the plane.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies in the inside half-space within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the plane within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return a representative centroid point for the plane.
     *
     * @return Representative point on the plane.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding representation of the plane.
     *
     * @return Axis-aligned bounding-box representation.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this plane is valid.
     *
     * @details
     * Typical validity checks include:
     * - finite offset,
     * - finite and non-degenerate normal.
     *
     * @return `true` if the plane is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a plane.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow the builder to configure plane internals directly.
    friend class Builder;

    /**
     * @brief Creates a lightweight runtime operator bound to this plane's current parameters.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE PlaneGeometryOperator<T>
    make_plane_operator() const noexcept;
};

/**
 * @brief Fluent builder for @ref Plane.
 *
 * @details
 * The builder provides a controlled construction path for planes by staging
 * parameters through either:
 * - a normal-offset representation, or
 * - a point-normal representation.
 *
 * ## Typical usage
 * @code
 * auto plane = atlas::PlaneF::builder()
 *     .with_normal_offset({0.0f, 0.0f, 1.0f}, 0.0f)
 *     .build();
 * @endcode
 *
 * or
 *
 * @code
 * auto plane = atlas::PlaneF::builder()
 *     .with_point_normal({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the normal is finite and non-zero,
 * - the offset is finite.
 *
 * The exact validation rules are implementation-defined in `plane.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Plane<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged parameters to the plane `z = 0`.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Plane by value after validation.
     *
     * @return Constructed plane value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Plane<T>
    build() const;

    /**
     * @brief Build a configured @ref Plane in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Plane<T>>` owning the constructed plane.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Plane<T>>
    make_host_shared() const;

    /**
     * @brief Set the plane normal.
     *
     * @param normal_ Staged normal vector.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    /**
     * @brief Set the plane offset.
     *
     * @param offset_ Staged offset value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_offset(T offset_) noexcept;

    /**
     * @brief Set the plane from a normal-offset representation.
     *
     * @param normal_ Staged normal vector.
     * @param offset_ Staged offset value.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept;

    /**
     * @brief Set the plane from a point-normal representation.
     *
     * @details
     * Computes the staged offset from the supplied point and normal.
     *
     * @param point Point lying on the plane.
     * @param normal_ Plane normal vector.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_point_normal(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged plane parameters.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending normal vector.
     */
    Vector3<T> _normal { T(0), T(0), T(1) };

    /**
     * @brief Pending plane offset.
     */
    T _offset { T(0) };
};

} // namespace atlas

namespace atlas {


/**
 * @brief Common specialization of @ref atlas::Plane for `float`.
 */
using PlaneF = Plane<float>;

/**
 * @brief Common specialization of @ref atlas::Plane for `double`.
 */
using PlaneD = Plane<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::Plane.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using PlaneHostPtr = atlas::host_shared_ptr<Plane<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::Plane.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using PlaneDevicePtr = atlas::device_shared_ptr<Plane<T>>;

} // namespace atlas

#include <atlas/geometry/plane.hpp>
