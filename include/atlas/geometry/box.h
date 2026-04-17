#pragma once

/**
 * @file box.h
 * @brief Declares axis-aligned box geometry types and query operators.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Non-owning geometry query operator for an axis-aligned box.
 *
 * This lightweight operator evaluates geometric queries using externally owned
 * lower and upper corner vectors. It does not own the corner data itself;
 * instead, it reads from the pointers stored in @ref lower_corner and
 * @ref upper_corner.
 *
 * Typical queries include:
 * - closest point on the box surface
 * - closest outward normal
 * - signed distance
 * - inside/surface tests
 * - centroid and bounding box evaluation
 * - ray intersection
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
struct BoxGeometryOperator {
    /**
     * @brief Pointer to the lower corner of the box.
     *
     * This pointer is non-owning and may be null.
     */
    const atlas::math::Vector<T, 3>* lower_corner = nullptr;

    /**
     * @brief Pointer to the upper corner of the box.
     *
     * This pointer is non-owning and may be null.
     */
    const atlas::math::Vector<T, 3>* upper_corner = nullptr;

    /**
     * @brief Returns the closest point on the box surface to a query point.
     *
     * For points outside the box, this is the clamped point on the box.
     * For points inside the box, the result is projected onto the nearest face.
     *
     * @param p Query point.
     * @return Closest point on the box surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Returns the outward surface normal associated with the closest point.
     *
     * For interior points, the normal of the nearest face is returned.
     * For exterior points, the normal is inferred from the dominant displacement
     * between the point and its clamped projection onto the box.
     *
     * @param p Query point.
     * @return Outward surface normal at the closest location.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Evaluates the signed distance from a point to the box.
     *
     * The returned value is:
     * - negative for points inside the box,
     * - zero on the surface,
     * - positive for points outside the box.
     *
     * @param p Query point.
     * @return Signed distance to the box surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Tests whether a point lies inside the box within a tolerance.
     *
     * The box extent is expanded by @p tolerance in each direction when
     * evaluating containment.
     *
     * @param p Query point.
     * @param tolerance Non-negative tolerance margin.
     * @return True if the point is inside the tolerance-expanded box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Tests whether a point lies on the box surface within a tolerance.
     *
     * This is typically evaluated using the absolute signed distance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return True if the point is within @p tolerance of the box surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Returns the centroid of the box.
     *
     * @return Midpoint between the lower and upper corners.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Returns the axis-aligned bounding box of this box geometry.
     *
     * @return Bounding box spanning the stored lower and upper corners.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Returns whether the referenced corners define a valid box.
     *
     * A box is valid when both corners are available and the upper corner is
     * component-wise greater than or equal to the lower corner.
     *
     * @return True if the box definition is valid.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Intersects a ray with the box.
     *
     * This function performs a ray-box intersection query and returns the
     * closest valid hit information if an intersection exists.
     *
     * @param ray Query ray.
     * @return Surface hit result for the box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    /**
     * @brief Callable shorthand for @ref trace.
     *
     * @param ray Query ray.
     * @return Surface hit result for the box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

/**
 * @brief Axis-aligned box geometry object.
 *
 * This class stores the actual lower and upper corners of an axis-aligned box
 * and provides the standard geometry query interface defined by Geometry<T>.
 *
 * Internally, the object maintains a lightweight BoxGeometryOperator<T> that
 * references its corner members. Because the operator stores raw pointers to
 * those members, copy and move operations must rebind the internal operator.
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
class Box final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Box requires a floating-point T");

public:
    /**
     * @brief Builder for constructing validated Box objects.
     */
    class Builder;

public:
    /**
     * @brief Lower corner of the box.
     */
    Vector3<T> lower_corner { T(-1), T(-1), T(-1) };

    /**
     * @brief Upper corner of the box.
     */
    Vector3<T> upper_corner { T(+1), T(+1), T(+1) };

    /**
     * @brief Default constructor.
     *
     * Initializes the box to the range [-1, +1] on each axis.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box() noexcept;

    /**
     * @brief Constructs a box from lower and upper corners.
     *
     * @param lower_corner_ Lower corner of the box.
     * @param upper_corner_ Upper corner of the box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Vector3<T>& lower_corner_,
        const Vector3<T>& upper_corner_) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent box construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Copy constructor.
     *
     * Copies the box corners and rebinds the internal geometry operator.
     *
     * @param other Source box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Box& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * Moves the box corners and rebinds the internal geometry operator.
     *
     * @param other Source box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(Box&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @param other Source box.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(const Box& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @param other Source box.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(Box&& other) noexcept;

    /**
     * @brief Destructor.
     */
    ~Box() override = default;

    /**
     * @brief Returns a generic geometry operator wrapper for this box.
     *
     * @return Geometry operator representing this box.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Returns the closest point on the box surface to a query point.
     *
     * @param p Query point.
     * @return Closest point on the box surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Returns the outward surface normal associated with the closest point.
     *
     * @param p Query point.
     * @return Outward surface normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Evaluates the signed distance from a point to the box.
     *
     * @param p Query point.
     * @return Signed distance to the box surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Tests whether a point lies inside the box within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Non-negative tolerance margin.
     * @return True if the point is inside the tolerance-expanded box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Tests whether a point lies on the box surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return True if the point is within @p tolerance of the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Returns the centroid of the box.
     *
     * @return Box centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Returns the axis-aligned bounding box of this geometry.
     *
     * @return Bounding box of this box geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether this box is geometrically valid.
     *
     * @return True if the upper corner is component-wise greater than or equal
     *         to the lower corner.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Returns the geometry type identifier.
     *
     * @return GeometryType::Box.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

    /**
     * @brief Rebinds the internal query operator to this object's corner members.
     *
     * This is required after construction, copy, and move operations because
     * the internal operator stores raw pointers to @ref lower_corner and
     * @ref upper_corner.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    /**
     * @brief Internal non-owning geometry query operator bound to this box.
     */
    mutable BoxGeometryOperator<T> _operator {};
};

/**
 * @brief Builder for Box.
 *
 * This builder stores lower and upper corner values, validates them, and
 * constructs Box instances with a fluent API.
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
class Box<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a validated Box object.
     *
     * @return Constructed Box object.
     *
     * @throw std::runtime_error Thrown if the corner configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Box<T>
    build() const;

    /**
     * @brief Builds a host-side shared Box object.
     *
     * @return Host shared pointer to a constructed Box object.
     *
     * @throw std::runtime_error Thrown if the corner configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Box<T>>
    make_host_shared() const;

    /**
     * @brief Sets the lower corner for the box being built.
     *
     * @param lower_corner_ Lower corner value.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& lower_corner_) noexcept;

    /**
     * @brief Sets the upper corner for the box being built.
     *
     * @param upper_corner_ Upper corner value.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& upper_corner_) noexcept;

private:
    /**
     * @brief Validates the current builder state.
     *
     * Ensures that the upper corner is component-wise greater than or equal to
     * the lower corner.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Lower corner used for construction.
     */
    Vector3<T> _lower_corner { T(-1), T(-1), T(-1) };

    /**
     * @brief Upper corner used for construction.
     */
    Vector3<T> _upper_corner { T(+1), T(+1), T(+1) };
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Alias for atlas::geometry::Box.
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
using Box = geometry::Box<T>;

/**
 * @brief Float specialization alias for Box.
 */
using BoxF = geometry::Box<float>;

/**
 * @brief Double specialization alias for Box.
 */
using BoxD = geometry::Box<double>;

/**
 * @brief Host-side shared pointer alias for Box.
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
using BoxHostPtr = atlas::host_shared_ptr<geometry::Box<T>>;

/**
 * @brief Device-side shared pointer alias for Box.
 *
 * @tparam T Floating-point scalar type used by the box geometry.
 */
template <typename T>
using BoxDevicePtr = atlas::device_shared_ptr<geometry::Box<T>>;

} // namespace atlas

#include <atlas/geometry/box.hpp>