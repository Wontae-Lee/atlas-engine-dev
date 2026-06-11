#pragma once

/**
 * @file square.h
 * @brief Declares an oriented square geometry primitive and its lightweight query operator.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Lightweight query operator for an oriented square.
 *
 * The operator references externally owned square parameters and provides
 * geometry queries such as closest-point projection, signed distance,
 * containment tests, bounding-box construction, and ray intersection.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SquareGeometryOperator {
    const atlas::math::Vector<T, 3>* center = nullptr;
    const atlas::math::Vector<T, 3>* normal = nullptr;
    const T* side_length                    = nullptr;

    /**
     * @brief Return the closest point on the square to a query point.
     *
     * @param p Query point.
     * @return Closest point on the square.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Return the square normal closest to the query point.
     *
     * Since the square is planar, this simply returns the normalized surface
     * normal when valid.
     *
     * @param p Query point.
     * @return Unit surface normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Return the signed distance from a point to the square.
     *
     * The distance sign is determined relative to the square normal direction.
     *
     * @param p Query point.
     * @return Signed distance to the square.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies on or inside the square within tolerance.
     *
     * Because the square is an infinitesimally thin planar primitive, this test
     * checks both proximity to the supporting plane and in-plane extents.
     *
     * @param p Query point.
     * @param tolerance Allowed tolerance around the square plane and edges.
     * @return True if the point is inside the square footprint within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the square surface within tolerance.
     *
     * For this thin planar primitive, this is equivalent to `is_inside`.
     *
     * @param p Query point.
     * @param tolerance Allowed tolerance around the square plane and edges.
     * @return True if the point is on the surface within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return the square centroid.
     *
     * @return Square center.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the square.
     *
     * @return Bounding box enclosing the square.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Check whether the referenced square parameters are valid.
     *
     * @return True if all references are bound and parameters define a valid square.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Intersect a ray with the square.
     *
     * @param ray Ray to trace.
     * @return Surface hit description.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    /**
     * @brief Shorthand call operator forwarding to `trace`.
     *
     * @param ray Ray to trace.
     * @return Surface hit description.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;

private:
    /**
     * @brief Build an orthonormal basis aligned with the square plane.
     *
     * @param input_normal Raw square normal.
     * @param unit_normal Output normalized normal.
     * @param tangent Output tangent axis.
     * @param bitangent Output bitangent axis.
     * @return `true` when basis construction succeeds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    build_basis(const atlas::math::Vector<T, 3>& input_normal,
                atlas::math::Vector<T, 3>& unit_normal,
                atlas::math::Vector<T, 3>& tangent,
                atlas::math::Vector<T, 3>& bitangent) const noexcept;
};

/**
 * @brief Oriented square geometry primitive.
 *
 * The square is represented by a center point, a surface normal, and a side
 * length. Query operations are delegated to a lightweight bound operator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Square final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Square requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> center { T(0), T(0), T(0) };
    Vector3<T> normal { T(0), T(0), T(1) };
    T side_length { T(1) };

    /**
     * @brief Construct a default square centered at the origin.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Square() noexcept;

    /**
     * @brief Construct a square from explicit parameters.
     *
     * @param center_ Square center.
     * @param normal_ Square surface normal.
     * @param side_length_ Square side length.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(const Vector3<T>& center_,
           const Vector3<T>& normal_,
           T side_length_) noexcept;

    /**
     * @brief Copy-construct a square.
     *
     * @param other Source square.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(const Square& other) noexcept;

    /**
     * @brief Move-construct a square.
     *
     * @param other Source square.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(Square&& other) noexcept;

    /**
     * @brief Copy-assign a square.
     *
     * @param other Source square.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Square&
    operator=(const Square& other) noexcept;

    /**
     * @brief Move-assign a square.
     *
     * @param other Source square.
     * @return Reference to this object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Square&
    operator=(Square&& other) noexcept;

    ~Square() override = default;

    /**
     * @brief Create a builder for `Square`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Return a polymorphic geometry operator for this square.
     *
     * @return Geometry operator bound to this square instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

    /**
     * @brief Bind the internal operator to this square's storage.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    mutable SquareGeometryOperator<T> _operator {};
};

/**
 * @brief Builder for `Square`.
 *
 * The builder stores square parameters directly and validates them before
 * construction.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Square<T>::Builder final {
public:
    Builder() = default;

    /**
     * @brief Set the square center.
     *
     * @param center_ Square center.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    /**
     * @brief Set the square normal.
     *
     * @param normal_ Square surface normal.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    /**
     * @brief Set the square side length.
     *
     * @param side_length_ Square side length.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_side_length(T side_length_) noexcept;

    /**
     * @brief Build a validated square.
     *
     * @return Constructed square.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Square<T>
    build() const;

    /**
     * @brief Build a validated square and wrap it in a host-shared pointer.
     *
     * @return Host-shared pointer to the constructed square.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Square<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate the builder configuration.
     *
     * @throws std::runtime_error Thrown when parameters are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _center { T(0), T(0), T(0) };
    Vector3<T> _normal { T(0), T(0), T(1) };
    T _side_length { T(1) };
};

} // namespace atlas::geometry

namespace atlas {

template <typename T>
using Square = geometry::Square<T>;

using SquareF = geometry::Square<float>;
using SquareD = geometry::Square<double>;

template <typename T>
using SquareHostPtr = atlas::host_shared_ptr<geometry::Square<T>>;

template <typename T>
using SquareDevicePtr = atlas::device_shared_ptr<geometry::Square<T>>;

} // namespace atlas

#include <atlas/geometry/square.hpp>
