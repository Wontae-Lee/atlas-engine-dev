#pragma once

/**
 * @file box.h
 * @brief Declares the axis-aligned box geometry type, its lightweight runtime
 *        query operator, and related convenience aliases.
 *
 * @details
 * This header introduces two closely related components:
 *
 * - @ref atlas::BoxGeometryOperator :
 *   a small query object that references box bounds through raw pointers and
 *   provides geometry evaluation routines such as closest-point queries,
 *   signed-distance queries, containment checks, and ray intersection tests.
 *
 * - @ref atlas::Box :
 *   a concrete axis-aligned box geometry type derived from @ref Geometry,
 *   storing lower and upper corners directly and exposing the standard
 *   geometry interface required by the Atlas geometry system.
 *
 * The box is axis-aligned in the coordinate frame in which it is queried.
 * No orientation is stored in this type. If the box must appear rotated or
 * transformed in another frame, that transform is expected to be handled
 * outside this class.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas {

/**
 * @brief Runtime query operator for an axis-aligned box.
 *
 * @details
 * This operator is a lightweight adapter around a box's lower and upper
 * corners. It does not own geometry data. Instead, it stores pointers to
 * externally managed corner vectors and performs geometric queries directly
 * against those values.
 *
 * Typical uses include:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - point containment and surface classification,
 * - centroid and bounding-box queries,
 * - ray/box intersection tests.
 *
 * The operator is intentionally small so it can be created cheaply and used
 * as a runtime geometry-query object in both host and device code paths.
 *
 * @tparam T Floating-point scalar type used for coordinate storage and query
 *           computations.
 */
template <typename T>
struct BoxGeometryOperator {
    const atlas::Vector<T, 3>* lower_corner = nullptr; ///< Pointer to the minimum corner of the box, interpreted component-wise as (x_min, y_min, z_min).
    const atlas::Vector<T, 3>* upper_corner = nullptr; ///< Pointer to the maximum corner of the box, interpreted component-wise as (x_max, y_max, z_max).

    /**
     * @brief Computes the closest point on the box to a query point.
     *
     * @details
     * For a point outside the box, the result is the Euclidean closest point
     * obtained by clamping the query coordinates into the box bounds.
     *
     * For a point inside the box, implementations commonly return the nearest
     * point on the box surface rather than the point itself, depending on the
     * operator policy used in the corresponding inline definition.
     *
     * @param p Query point expressed in the same coordinate frame as the box.
     * @return Closest point on the box according to the operator's query policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Computes an outward-facing normal associated with the closest box feature.
     *
     * @details
     * The returned normal typically corresponds to the face selected by the
     * closest-point logic. For exterior points, it usually points away from
     * the box toward the query point. For interior points, it is commonly the
     * outward normal of the nearest face.
     *
     * Tie-breaking for points nearest to edges or corners is implementation-defined
     * by the corresponding inline definition.
     *
     * @param p Query point expressed in the same coordinate frame as the box.
     * @return Outward-facing normal vector associated with the closest feature.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Computes the signed distance from a point to the box.
     *
     * @details
     * The usual sign convention is:
     * - negative inside the box,
     * - zero on the surface,
     * - positive outside the box.
     *
     * The exact numerical behavior, especially on degenerate boxes or invalid
     * operator state, is defined by the inline implementation in the
     * corresponding source header.
     *
     * @param p Query point expressed in the same coordinate frame as the box.
     * @return Signed distance from @p p to the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Tests whether a point lies inside the box within a tolerance band.
     *
     * @details
     * Containment is evaluated component-wise against the lower and upper box
     * bounds. A positive tolerance expands the accepted region, while zero
     * tolerance performs an exact inclusive bound test.
     *
     * @param p Query point expressed in the same coordinate frame as the box.
     * @param tolerance Non-negative or user-defined tolerance applied to each bound.
     * @return `true` if the point is considered inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Tests whether a point lies on or sufficiently near the box surface.
     *
     * @details
     * This check is typically based on the absolute signed distance being less
     * than or equal to the supplied tolerance.
     *
     * @param p Query point expressed in the same coordinate frame as the box.
     * @param tolerance Accepted absolute distance from the surface.
     * @return `true` if the point is considered on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Returns the centroid of the box.
     *
     * @details
     * For a valid axis-aligned box, the centroid is the midpoint of the lower
     * and upper corners.
     *
     * @return Geometric center of the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Returns the axis-aligned bounding box enclosing this box.
     *
     * @details
     * Since the geometry itself is already axis-aligned, the returned bounding
     * box is generally identical to the box bounds referenced by this operator.
     *
     * @return Axis-aligned bounding box enclosing the geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Validates the referenced box bounds.
     *
     * @details
     * A valid box requires:
     * - both corner pointers to be non-null, and
     * - each component of the lower corner to be less than or equal to the
     *   corresponding component of the upper corner.
     *
     * Degenerate but ordered boxes may still be considered valid.
     *
     * @return `true` if the referenced bounds form a valid axis-aligned box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Intersects a ray with the box.
     *
     * @details
     * This routine returns a @ref HitSurface record describing whether the ray
     * intersects the box and, if so, the distance, hit point, and outward
     * normal associated with the selected intersection.
     *
     * Implementations usually follow a slab-based ray/AABB intersection method.
     *
     * @param ray Ray expressed in the same coordinate frame as the box.
     * @return Hit record describing the ray/box intersection result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::Ray<T>& ray) const noexcept;

    /**
     * @brief Callable shorthand for @ref trace.
     *
     * @param ray Ray expressed in the same coordinate frame as the box.
     * @return Hit record describing the ray/box intersection result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::Ray<T>& ray) const noexcept;
};

/**
 * @brief Axis-aligned box geometry.
 *
 * @details
 * @ref Box is a concrete geometry type representing an axis-aligned box
 * described by two corners:
 * - @ref lower_corner : component-wise minimum corner,
 * - @ref upper_corner : component-wise maximum corner.
 *
 * The class derives from @ref Geometry and provides the full box-specific
 * implementation of the standard geometry query interface, including:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance queries,
 * - containment and surface checks,
 * - centroid and bounding-box evaluation,
 * - runtime geometry operator creation.
 *
 * Device/runtime views are created on demand from the current corner data.
 *
 * @tparam T Floating-point scalar type used for coordinates and geometric queries.
 */
template <typename T>
class Box final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Box requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> lower_corner { T(-1), T(-1), T(-1) }; ///< Component-wise minimum corner of the box.
    Vector3<T> upper_corner { T(+1), T(+1), T(+1) }; ///< Component-wise maximum corner of the box.

    /**
     * @brief Constructs a default canonical axis-aligned box.
     *
     * @details
     * The default box spans from (-1, -1, -1) to (+1, +1, +1), providing a
     * valid, non-degenerate volume immediately usable in geometric queries.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box() noexcept;

    /**
     * @brief Constructs a box directly from lower and upper corners.
     *
     * @param lower_corner_ Component-wise minimum corner.
     * @param upper_corner_ Component-wise maximum corner.
     *
     * @details
     * This constructor stores the supplied corners directly. Any validity
     * requirements on corner ordering are enforced by convention or by explicit
     * validation elsewhere, such as through the builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Vector3<T>& lower_corner_,
        const Vector3<T>& upper_corner_) noexcept;

    /**
     * @brief Creates a fluent builder for staged box construction.
     *
     * @return New builder instance initialized with the default box bounds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Copy-constructs a box.
     *
     * @param other Source box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Box& other) noexcept;

    /**
     * @brief Move-constructs a box.
     *
     * @param other Source box to move from.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(Box&& other) noexcept;

    /**
     * @brief Copy-assigns a box.
     *
     * @param other Source box.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(const Box& other) noexcept;

    /**
     * @brief Move-assigns a box.
     *
     * @param other Source box to move from.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(Box&& other) noexcept;

    /**
     * @brief Destroys the box.
     */
    ~Box() override = default;

    /**
     * @brief Creates a runtime geometry operator bound to this box.
     *
     * @details
     * The returned operator references this instance's corner data and can be
     * used to execute box queries through the generic runtime geometry path.
     *
     * @return Geometry operator bound to this box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    /**
     * @brief Computes the closest point on the box to a query point.
     *
     * @param p Query point in the box's coordinate frame.
     * @return Closest point on the box according to the class query policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Computes an outward-facing normal associated with the closest box feature.
     *
     * @param p Query point in the box's coordinate frame.
     * @return Closest-feature outward normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Computes the signed distance from a point to the box.
     *
     * @param p Query point in the box's coordinate frame.
     * @return Signed distance from @p p to the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Tests whether a point lies inside the box within a tolerance.
     *
     * @param p Query point in the box's coordinate frame.
     * @param tolerance Tolerance applied to the containment test.
     * @return `true` if the point is considered inside.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Tests whether a point lies on or near the box surface.
     *
     * @param p Query point in the box's coordinate frame.
     * @param tolerance Accepted distance from the surface.
     * @return `true` if the point is considered on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Returns the centroid of the box.
     *
     * @return Geometric center of the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Returns the axis-aligned bounding box of the box geometry.
     *
     * @return Bounding box enclosing this geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Tests whether the stored box bounds are valid.
     *
     * @return `true` if @ref lower_corner is component-wise less than or equal
     *         to @ref upper_corner.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Returns the runtime geometry type tag for this class.
     *
     * @return @ref GeometryType::Box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

    /**
     * @brief Creates a lightweight runtime operator bound to the current corners.
     *
     * @details
     * This helper avoids persistent pointer caches while still sharing the
     * query implementation with @ref BoxGeometryOperator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE BoxGeometryOperator<T>
    make_box_operator() const noexcept;
};

template <typename T>
class Box<T>::Builder final {
public:
    /**
     * @brief Constructs a builder initialized with the default box bounds.
     */
    Builder() = default;

    /**
     * @brief Builds a validated @ref Box value from the current builder state.
     *
     * @details
     * This function validates the stored parameters and then returns a concrete
     * box object initialized from them.
     *
     * @return Constructed box instance.
     *
     * @throws std::runtime_error
     * Thrown when the stored parameters do not define a valid box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box<T>
    build() const;

    /**
     * @brief Builds a validated box and returns it in host-shared ownership.
     *
     * @return Host-shared pointer to the constructed box.
     *
     * @throws std::runtime_error
     * Thrown when the stored parameters do not define a valid box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Box<T>>
    make_host_shared() const;

    /**
     * @brief Sets the lower corner to use for the built box.
     *
     * @param lower_corner_ Component-wise minimum corner.
     * @return Reference to this builder for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& lower_corner_) noexcept;

    /**
     * @brief Sets the upper corner to use for the built box.
     *
     * @param upper_corner_ Component-wise maximum corner.
     * @return Reference to this builder for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& upper_corner_) noexcept;

private:
    /**
     * @brief Validates the currently stored builder parameters.
     *
     * @details
     * Validation checks that the lower corner is component-wise less than or
     * equal to the upper corner so that the resulting box is geometrically valid.
     *
     * @throws std::runtime_error
     * Thrown when the stored corners do not satisfy the required ordering.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _lower_corner { T(-1), T(-1), T(-1) }; ///< Staged lower corner used for the next build.
    Vector3<T> _upper_corner { T(+1), T(+1), T(+1) }; ///< Staged upper corner used for the next build.
};

} // namespace atlas

namespace atlas {


/**
 * @brief Single-precision axis-aligned box type.
 */
using BoxF = Box<float>;

/**
 * @brief Double-precision axis-aligned box type.
 */
using BoxD = Box<double>;

/**
 * @brief Convenience alias for a host-shared box pointer.
 *
 * @tparam T Floating-point scalar type used by the box.
 */
template <typename T>
using BoxHostPtr = atlas::host_shared_ptr<Box<T>>;

/**
 * @brief Convenience alias for a device-shared box pointer.
 *
 * @tparam T Floating-point scalar type used by the box.
 */
template <typename T>
using BoxDevicePtr = atlas::device_shared_ptr<Box<T>>;

} // namespace atlas

#include <atlas/geometry/box.hpp>
