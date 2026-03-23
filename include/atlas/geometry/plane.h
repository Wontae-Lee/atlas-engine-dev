#pragma once

/**
 * @file plane.h
 * @brief Infinite plane geometry primitive and builder.
 *
 * @details
 * This header defines @ref atlas::geometry::Plane, an **infinite plane** geometry described by:
 * - a plane normal vector @ref normal
 * - a scalar offset @ref offset
 *
 * `Plane` implements the @ref atlas::Geometry interface and provides:
 * - query and trace operator construction for interop with spatial systems,
 * - closest-point and closest-normal evaluation,
 * - signed distance evaluation,
 * - a centroid accessor (implementation-defined for an infinite object),
 * - an axis-aligned bounding box accessor (implementation-defined for an infinite object),
 * - validity checks.
 *
 * ## Plane equation convention
 * This API stores the plane in a common implicit form:
 * \f[
 *   \mathbf{n}\cdot \mathbf{x} = d
 * \f]
 * where:
 * - \f$\mathbf{n}\f$ is @ref normal
 * - \f$d\f$ is @ref offset
 *
 * Equivalently:
 * \f[
 *   f(\mathbf{x}) = \mathbf{n}\cdot \mathbf{x} - d = 0
 * \f]
 *
 * With this convention, the **signed distance** (when \f$\|\mathbf{n}\|=1\f$) is:
 * \f[
 *   \phi(\mathbf{x}) = \mathbf{n}\cdot \mathbf{x} - d
 * \f]
 *
 * If the normal is not unit length, the returned signed distance is typically scaled by
 * \f$\|\mathbf{n}\|\f$ unless the implementation explicitly normalizes.
 *
 * ## Construction options
 * - Default construction produces the XY plane through the origin with normal (0,0,1).
 * - Construct from (normal, offset).
 * - Construct from a point on the plane and a normal.
 * - Fluent @ref Builder that stages and validates parameters.
 *
 * ## Infinite-geometry caveats
 * A plane is unbounded, so:
 * - @ref centroid() is not mathematically unique; implementations usually return a representative
 *   point (often the closest point to the origin).
 * - @ref bound() cannot be a finite box; implementations may return an "infinite" AABB, a very
 *   large box, or a sentinel bounding box (implementation-defined).
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * For predictable signed-distance semantics, provide a **normalized** normal vector (unit length),
 * unless your implementation explicitly normalizes and adjusts offset accordingly.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct PlaneGeometryOperator {
    const atlas::math::Vector<T, 3>* normal = nullptr;
    const T* offset                         = nullptr;

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
 * @brief Infinite plane geometry primitive.
 *
 * @details
 * `Plane` represents an infinite plane using an implicit equation parameterization.
 *
 * ### Stored representation
 * The plane is stored using (@ref normal, @ref offset) such that:
 * \f[
 *   \mathbf{n}\cdot \mathbf{x} = d
 * \f]
 *
 * where \f$\mathbf{n}\f$ is the normal and \f$d\f$ is the offset.
 *
 * ### Closest point
 * For a unit normal, the closest point of \f$\mathbf{p}\f$ onto the plane is:
 * \f[
 *   \mathbf{p}_{proj} = \mathbf{p} - (\mathbf{n}\cdot \mathbf{p} - d)\,\mathbf{n}
 * \f]
 *
 * ### Signed distance
 * For a unit normal:
 * \f[
 *   \phi(\mathbf{p}) = \mathbf{n}\cdot \mathbf{p} - d
 * \f]
 *
 * ### Host/device
 * - Evaluation methods are `ATLAS_ALL_DEVICE`.
 * - Operator construction is host-only.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Plane final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Plane requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Plane.
     *
     * @details
     * See @ref Plane<T>::Builder for supported configuration routes and validation.
     */
    class Builder;

public:
    /**
     * @brief Plane normal vector \f$\mathbf{n}\f$.
     *
     * @details
     * The normal defines the plane orientation. For meaningful signed-distance output,
     * this normal should typically be unit length.
     *
     * @note
     * The implementation may or may not normalize. If it does normalize, it must also adjust
     * @ref offset to keep the represented plane unchanged.
     */
    Vector3<T> normal { T(0), T(0), T(1) };

    /**
     * @brief Plane offset \f$d\f$ in the equation \f$\mathbf{n}\cdot \mathbf{x} = d\f$.
     *
     * @details
     * Interpreting `offset` depends on whether @ref normal is normalized:
     * - If `||normal|| == 1`, `offset` is the signed distance from the origin to the plane
     *   measured along the normal direction.
     * - If not normalized, `offset` scales with `||normal||`.
     */
    T offset { T(0) };

    /**
     * @brief Default constructor (XY plane through origin).
     *
     * @details
     * Initializes:
     * - normal = (0,0,1)
     * - offset = 0
     * corresponding to the plane z = 0.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane() noexcept;

    /**
     * @brief Construct a plane from a normal and an offset.
     *
     * @param normal_ Plane normal.
     * @param offset_ Plane offset `d` in `normal·x = d`.
     *
     * @note
     * The caller is responsible for consistent normalization unless the implementation
     * explicitly normalizes.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Vector3<T>& normal_, T offset_) noexcept;

    /**
     * @brief Construct a plane from a point on the plane and a normal.
     *
     * @details
     * Given a point \f$\mathbf{p}\f$ on the plane and normal \f$\mathbf{n}\f$, the offset is:
     * \f[
     *   d = \mathbf{n}\cdot \mathbf{p}
     * \f]
     *
     * @param point A point lying on the plane.
     * @param normal_ Plane normal.
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

    /// @brief Defaulted copy constructor.
    Plane(const Plane&) noexcept = default;

    /// @brief Virtual destructor (geometry base).
    ~Plane() override = default;

    /**
     * @brief Create a trace operator bound to this plane.
     *
     * @details
     * Returns a @ref GeometryOperator usable by tracing systems to intersect rays with the plane.
     * Implementations typically store non-owning pointers to @ref normal and @ref offset.
     *
     * @return Trace operator referencing this plane.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */

    /**
     * @brief Create a query operator bound to this plane.
     *
     * @details
     * Returns a @ref GeometryOperator usable by query systems for closest-point and signed-distance
     * evaluation against the plane. Implementations typically store non-owning pointers to members.
     *
     * @return Query operator referencing this plane.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Compute the closest point on the plane to a query point.
     *
     * @param p Query point.
     * @return Orthogonal projection of `p` onto the plane.
     *
     * @note
     * If the normal is not unit length and the implementation does not normalize, the projection
     * formula must account for `||normal||^2`. Confirm behavior in `plane.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Return the (constant) plane normal associated with the closest point.
     *
     * @details
     * For an infinite plane, the closest normal is generally the plane normal (or its normalized form).
     *
     * @param p Query point (unused in many implementations).
     * @return Plane normal direction.
     *
     * @note
     * Some implementations always return a normalized normal; others may return the stored normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the plane.
     *
     * @details
     * For unit normal, the signed distance is:
     * \f[
     *   \phi(\mathbf{p}) = \mathbf{n}\cdot \mathbf{p} - d
     * \f]
     *
     * @param p Query point.
     * @return Signed distance (implementation-defined if normal is not unit length).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies in the plane's negative half-space within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Allowed positive slack relative to the plane equation.
     * @return `true` if the point satisfies the plane-side classification.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the plane within a tolerance band.
     *
     * @param p Query point.
     * @param tolerance Allowed absolute deviation from the plane equation.
     * @return `true` if the point is classified as on the plane surface.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return a representative "centroid" for the plane.
     *
     * @details
     * A plane is infinite and has no unique centroid. Implementations typically return a stable
     * representative point such as:
     * - the closest point on the plane to the origin, or
     * - any point satisfying the plane equation.
     *
     * @return Representative point on the plane (implementation-defined).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return an axis-aligned bounding box for the plane.
     *
     * @details
     * A plane is unbounded, so a finite AABB does not exist. Implementations may return:
     * - an AABB with infinite extents (if supported),
     * - a very large sentinel AABB,
     * - or a conservative placeholder.
     *
     * @return Bounding volume for broad-phase systems (implementation-defined).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether this plane representation is valid.
     *
     * @details
     * Typical checks include:
     * - normal is finite,
     * - normal length is non-zero,
     * - offset is finite.
     *
     * @return `true` if parameters are well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this plane.
     *
     * @details
     * This identifies the active type in a polymorphic context (e.g., when using `Geometry<T>` pointers).
     *
     * @return `GeometryType::Plane` for this class.
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
 * @brief Fluent builder for @ref Plane.
 *
 * @details
 * The builder stages plane parameters and supports multiple construction routes:
 * - normal + offset
 * - point + normal (computes offset as `dot(normal, point)`)
 *
 * ## Typical usage
 * @code
 * atlas::PlaneF p = atlas::PlaneF::builder()
 *     .with_point_normal({0,0,1}, {0,0,1})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks include:
 * - normal is finite and non-zero length
 * - offset is finite
 *
 * The exact behavior is implemented in `plane.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Plane<T>::Builder final {
public:
    /// @brief Default constructor (stages the default plane).
    Builder() = default;

    /**
     * @brief Build a configured @ref Plane (by value).
     *
     * @return Constructed plane by value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Plane<T>
    build() const;

    /**
     * @brief Build a configured @ref Plane in a host_shared_ptr.
     *
     * @return `atlas::host_shared_ptr<Plane<T>>` owning the constructed plane.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Plane<T>>
    make_host_shared() const;

    /**
     * @brief Set plane normal.
     *
     * @param normal_ Plane normal.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    /**
     * @brief Set plane offset.
     *
     * @param offset_ Plane offset `d` in `normal·x = d`.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_offset(T offset_) noexcept;

    /**
     * @brief Set plane normal and offset together.
     *
     * @param normal_ Plane normal.
     * @param offset_ Plane offset.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept;

    /**
     * @brief Configure the plane from a point on the plane and a normal.
     *
     * @details
     * Computes:
     * \f[
     *   d = \mathbf{n}\cdot \mathbf{p}
     * \f]
     *
     * @param point A point on the plane.
     * @param normal_ Plane normal.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_point_normal(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;

private:
    /**
     * @brief Validate staged parameters.
     *
     * @details
     * Expected checks:
     * - normal is finite and non-zero length
     * - offset is finite
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Staged normal.
    Vector3<T> _normal { T(0), T(0), T(1) };

    /// @brief Staged offset.
    T _offset { T(0) };
};

} // namespace atlas::geometry

namespace atlas {

template <typename T>
using Plane  = geometry::Plane<T>;
using PlaneF = geometry::Plane<float>;
using PlaneD = geometry::Plane<double>;

template <typename T>
using PlaneHostPtr = atlas::host_shared_ptr<geometry::Plane<T>>;
template <typename T>
using PlaneDevicePtr = atlas::device_shared_ptr<geometry::Plane<T>>;

} // namespace atlas

#include <atlas/geometry/plane.hpp>
