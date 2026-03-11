#pragma once

/**
 * @file box.h
 * @brief Axis-aligned box geometry primitive and builder (AABB).
 *
 * @details
 * This header defines @ref atlas::geometry::Box, an axis-aligned box (AABB) geometry
 * described by two corners:
 * - @ref lower_corner : minimum corner (x_min, y_min, z_min)
 * - @ref upper_corner : maximum corner (x_max, y_max, z_max)
 *
 * `Box` implements the @ref atlas::Geometry interface and provides:
 * - query and trace operator construction for spatial acceleration/interoperability,
 * - closest-point and closest-normal evaluation,
 * - signed distance evaluation,
 * - centroid and bounding box accessors,
 * - validity checks.
 *
 * ## Coordinate convention
 * The box is assumed to be axis-aligned in the coordinate frame in which it is queried.
 * Any world-space orientation/pose must be applied externally (e.g., via a @ref atlas::system::Sync).
 *
 * ## Construction
 * `Box` can be default-constructed (unit box [-1, +1]^3) or configured via the nested
 * fluent @ref Builder which:
 * - stages lower/upper corners,
 * - optionally validates the bounds (e.g., lower <= upper component-wise),
 * - can build by value or as a `host_shared_ptr`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * - Most geometry operations assume the box is valid (lower <= upper).
 * - If the box is invalid, results are implementation-defined; use @ref is_valid() to check.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/spatial/trace_operator.h>

#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Axis-aligned box geometry primitive (AABB) implementing @ref atlas::Geometry.
 *
 * @details
 * `Box` represents an axis-aligned bounding box parameterized by its lower and upper corners.
 * It is commonly used as:
 * - a simple solid for collisions,
 * - a bounding volume,
 * - a quick analytic geometry for distance and projection operations,
 * - a building block for larger CSG-like or compound geometries.
 *
 * ### Mathematical definition
 * Let:
 * - \f$\mathbf{l} = (l_x,l_y,l_z)\f$ be @ref lower_corner
 * - \f$\mathbf{u} = (u_x,u_y,u_z)\f$ be @ref upper_corner
 * - a point \f$\mathbf{p}\f$ is inside the box iff \f$l_i \le p_i \le u_i\f$ for all axes.
 *
 * The closest point on the box to an arbitrary point \f$\mathbf{p}\f$ is obtained by clamping:
 * \f[
 * \mathbf{c} = \mathrm{clamp}(\mathbf{p}, \mathbf{l}, \mathbf{u})
 * \f]
 * (with additional inside-case handling if you want the closest point on the *surface*).
 *
 * Signed distance is typically:
 * - positive outside, negative inside, zero on the surface (implementation-defined sign convention).
 *
 * ### Host/device
 * - Pure geometry evaluation methods are `ATLAS_ALL_DEVICE`.
 * - Operator construction is host-only (`ATLAS_HOST`) because it typically binds host pointers.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Box final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Box requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Box.
     *
     * @details
     * See @ref Box<T>::Builder for configuration methods and validation policy.
     */
    class Builder;

public:
    /**
     * @brief Lower (minimum) corner of the box.
     *
     * @details
     * Represents the component-wise minimum bound \f$(x_{min},y_{min},z_{min})\f$.
     * The invariant for a valid box is:
     * \f[
     * \texttt{lower_corner} \le \texttt{upper_corner} \;\; \text{(component-wise)}
     * \f]
     */
    Vector3<T> lower_corner { T(-1), T(-1), T(-1) };

    /**
     * @brief Upper (maximum) corner of the box.
     *
     * @details
     * Represents the component-wise maximum bound \f$(x_{max},y_{max},z_{max})\f$.
     */
    Vector3<T> upper_corner { T(+1), T(+1), T(+1) };

    /**
     * @brief Default constructor (unit box centered at origin).
     *
     * @details
     * Initializes bounds to:
     * - lower = (-1,-1,-1)
     * - upper = (+1,+1,+1)
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box() noexcept;

    /**
     * @brief Construct a box from explicit lower/upper corners.
     *
     * @param lower_corner_ Lower (minimum) corner.
     * @param upper_corner_ Upper (maximum) corner.
     *
     * @note
     * No automatic validation or reordering is implied by the declaration; use @ref is_valid()
     * or the @ref Builder for validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Vector3<T>& lower_corner_,
        const Vector3<T>& upper_corner_) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Defaulted copy constructor.
    Box(const Box& other) noexcept = default;

    /// @brief Virtual destructor (geometry base).
    ~Box() override = default;

    /**
     * @brief Create a trace operator bound to this box.
     *
     * @details
     * Returns a @ref TraceOperator that can be used by tracing systems to intersect rays
     * with this box. Implementations commonly store non-owning pointers to this object’s bounds.
     *
     * @return Trace operator referencing this box.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TraceOperator<T>
    make_trace_operator() const override;

    /**
     * @brief Create a query operator bound to this box.
     *
     * @details
     * Returns a @ref QueryOperator that can be used by query systems to compute closest points,
     * signed distances, etc., against this box. Implementations commonly store non-owning pointers
     * to this object’s bounds.
     *
     * @return Query operator referencing this box.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE QueryOperator<T>
    make_query_operator() const override;

    /**
     * @brief Compute the closest point on (or in) the box to an input point.
     *
     * @details
     * Typical behavior:
     * - Outside point: returns clamped point on the box surface.
     * - Inside point: behavior may differ by policy:
     *   - either returns the point itself (closest point in volume),
     *   - or projects to the nearest face (closest point on surface).
     *
     * @param p Query point.
     * @return Closest point in the chosen sense (implementation-defined inside behavior).
     *
     * @note
     * If you need *surface* projection semantics when inside, confirm the implementation in
     * `box.hpp` (or use the query operator if it encodes explicit policy).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the outward normal corresponding to the closest feature to a point.
     *
     * @details
     * Typical behavior:
     * - Outside point: returns normal of the face that the closest point lies on.
     * - Inside point: returns normal of the nearest face (or an implementation-defined rule).
     *
     * @param p Query point.
     * @return A (typically unit-length) normal vector.
     *
     * @note
     * In edge/corner cases, the normal selection may be ambiguous; implementations typically
     * break ties in a deterministic way (e.g., smallest axis distance).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the box.
     *
     * @details
     * Common convention:
     * - positive outside the box,
     * - zero on the surface,
     * - negative inside.
     *
     * For an AABB, a typical SDF is computed using the distance to the clamped point outside
     * and the maximum penetration depth inside (implementation-defined exact formulation).
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Return the centroid of the box.
     *
     * @details
     * For axis-aligned bounds:
     * \f[
     * \mathbf{c} = \frac{\mathbf{l} + \mathbf{u}}{2}
     * \f]
     *
     * @return Center point of the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of this geometry.
     *
     * @details
     * For @ref Box, this is typically the box itself.
     *
     * @return Axis-aligned bounding box in the same coordinate frame.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether this box is valid.
     *
     * @details
     * A valid box requires:
     * - finite components, and
     * - `lower_corner[i] <= upper_corner[i]` for i in {x,y,z}.
     *
     * @return `true` if bounds are well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this box.
     *
     * @details
     * This identifies the active type in a polymorphic context (e.g., when using `Geometry<T>` pointers).
     *
     * @return `GeometryType::Box` for this class.
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
 * @brief Fluent builder for @ref Box.
 *
 * @details
 * The builder provides a controlled way to construct a @ref Box while applying a validation step.
 *
 * ## Typical usage
 * @code
 * atlas::BoxF box = atlas::BoxF::builder()
 *     .with_bounds({-1,-1,-1}, {+1,+1,+1})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. A typical policy checks:
 * - component-wise ordering: lower <= upper
 * - finiteness of all components
 *
 * The exact policy is implementation-defined and lives in `box.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Box<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged bounds to the default unit box [-1,+1]^3.
     */
    Builder() = default;

    /**
     * @brief Set the lower (minimum) corner.
     *
     * @param lower_corner_ Lower bound corner.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& lower_corner_) noexcept;

    /**
     * @brief Set the upper (maximum) corner.
     *
     * @param upper_corner_ Upper bound corner.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& upper_corner_) noexcept;

    /**
     * @brief Build a configured @ref Box (by value).
     *
     * @details
     * Validates staged bounds and constructs a @ref Box with those bounds.
     *
     * @return Constructed box by value.
     *
     * @note
     * Validation behavior depends on the implementation in `box.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box<T>
    build() const;

    /**
     * @brief Build a configured @ref Box in a host_shared_ptr.
     *
     * @details
     * Builds a box and returns a host shared pointer owning it.
     *
     * @return `atlas::host_shared_ptr<Box<T>>` owning the constructed box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Box<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate staged bounds.
     *
     * @details
     * Expected checks include:
     * - lower <= upper (component-wise)
     * - finite bounds
     *
     * @note
     * Implementation may `throw`, `assert`, or no-op depending on Atlas policy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Staged lower corner.
    Vector3<T> _lower_corner { T(-1), T(-1), T(-1) };

    /// @brief Staged upper corner.
    Vector3<T> _upper_corner { T(+1), T(+1), T(+1) };
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for `atlas::geometry::Box<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Box = geometry::Box<T>;

/// @brief Common specialization: float box.
using BoxF = geometry::Box<float>;

/// @brief Common specialization: double box.
using BoxD = geometry::Box<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to `atlas::geometry::Box<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using BoxHostPtr = atlas::host_shared_ptr<geometry::Box<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to `atlas::geometry::Box<T>`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using BoxDevicePtr = atlas::device_shared_ptr<geometry::Box<T>>;

} // namespace atlas

#include <atlas/geometry/box.hpp>
