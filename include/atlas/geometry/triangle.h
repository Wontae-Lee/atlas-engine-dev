#pragma once

/**
 * @file triangle.h
 * @brief Declares a triangle geometry primitive and its lightweight runtime query/trace operator.
 *
 * @details
 * This header defines @ref atlas::geometry::Triangle, a finite planar triangle
 * embedded in 3D space and represented by three vertices:
 * - @ref a
 * - @ref b
 * - @ref c
 *
 * It also defines @ref TriangleGeometryOperator, a lightweight non-owning
 * operator object that stores raw pointers to the triangle data and exposes
 * backend-friendly geometric functionality such as:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside and surface classification,
 * - centroid and bounding-box queries,
 * - ray tracing against the triangle,
 * - barycentric-style point reasoning through the owning triangle API.
 *
 * ## Geometry interpretation
 * A triangle is the finite convex planar region spanned by its three vertices.
 * The associated supporting plane is determined by the vertex positions, while
 * the oriented surface normal may be:
 * - derived from the vertex winding, or
 * - explicitly overridden/stored by the implementation.
 *
 * The triangle is typically interpreted as the set of points:
 * \f[
 * \mathbf{p} = u\mathbf{a} + v\mathbf{b} + w\mathbf{c},
 * \qquad
 * u,v,w \ge 0,
 * \qquad
 * u+v+w=1.
 * \f]
 *
 * ## Host/device split
 * - The owning @ref Triangle object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref TriangleGeometryOperator provides a lightweight device-friendly view
 *   that avoids host-side ownership and virtual dispatch.
 *
 * ## Construction
 * A triangle may be:
 * - default-constructed,
 * - constructed directly from three vertices,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and distances.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <optional>
#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Lightweight non-owning runtime query operator for a triangle.
 *
 * @details
 * @ref TriangleGeometryOperator stores raw pointers to the defining parameters
 * of a triangle and exposes query operations suitable for host or device
 * execution without relying on polymorphic ownership.
 *
 * The operator is intended to be:
 * - cheap to copy,
 * - non-owning,
 * - suitable for device execution,
 * - consistent with the behavior of the owning @ref Triangle object.
 *
 * ## Stored references
 * The operator references:
 * - @ref a : first triangle vertex,
 * - @ref b : second triangle vertex,
 * - @ref c : third triangle vertex,
 * - @ref n : auxiliary edge or cross-product-like vector used internally by the implementation,
 * - @ref normal : oriented surface normal of the triangle.
 *
 * Since the stored pointers are non-owning, the referenced data must remain
 * valid for the duration of any use of the operator.
 *
 * @note
 * The exact semantic role of @ref n is implementation-defined in `triangle.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct TriangleGeometryOperator {
    /**
     * @brief Pointer to the first triangle vertex.
     */
    const atlas::math::Vector<T, 3>* a = nullptr;

    /**
     * @brief Pointer to the second triangle vertex.
     */
    const atlas::math::Vector<T, 3>* b = nullptr;

    /**
     * @brief Pointer to the third triangle vertex.
     */
    const atlas::math::Vector<T, 3>* c = nullptr;

    /**
     * @brief Pointer to an auxiliary precomputed vector.
     *
     * @details
     * This member is used internally by the implementation for query support.
     * Its exact meaning is implementation-defined.
     */
    const atlas::math::Vector<T, 3>* n = nullptr;

    /**
     * @brief Pointer to the oriented triangle surface normal.
     */
    const atlas::math::Vector<T, 3>* normal = nullptr;

    /**
     * @brief Compute the closest point on the triangle to a query point.
     *
     * @details
     * The result may lie:
     * - in the interior of the triangular face,
     * - on one of the triangle edges,
     * - or at one of the triangle vertices,
     * depending on the location of the query point.
     *
     * @param p Query point in world space.
     * @return Closest point on the triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest outward normal associated with the triangle.
     *
     * @details
     * For interior face points, this is typically the triangle's oriented surface
     * normal. Edge and vertex cases may still return the face normal according to
     * implementation policy.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the triangle.
     *
     * @details
     * The exact sign convention and edge-handling details are implementation-defined
     * in `triangle.hpp`, but the result is intended to represent the shortest
     * distance to the finite triangle with a consistent sign rule.
     *
     * @param p Query point in world space.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies inside the triangle within a tolerance.
     *
     * @details
     * For a finite triangle, this generally means the point lies on or near the
     * triangle-supporting face region with barycentric coordinates inside the
     * admissible range, up to the supplied tolerance.
     *
     * @param p Query point.
     * @param tolerance Non-negative classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the triangle surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as lying on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return the centroid of the triangle.
     *
     * @details
     * For a triangle with vertices \f$\mathbf{a},\mathbf{b},\mathbf{c}\f$, the
     * centroid is:
     * \f[
     * \frac{\mathbf{a}+\mathbf{b}+\mathbf{c}}{3}.
     * \f]
     *
     * @return Triangle centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the triangle.
     *
     * @return Axis-aligned bounding box enclosing the triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the referenced triangle parameters define a valid triangle.
     *
     * @details
     * Typical validity checks include:
     * - all referenced pointers are non-null,
     * - all vertex coordinates are finite,
     * - the triangle is not degenerate,
     * - the stored normal is finite and consistent with the geometry.
     *
     * @return `true` if the operator references a valid triangle; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the triangle.
     *
     * @details
     * Intersects the ray with the triangle-supporting plane and checks whether the
     * hit point lies inside the finite triangular region.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the ray-triangle intersection result.
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
 * @brief Finite triangle geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref Triangle represents a single triangular surface primitive in 3D space.
 *
 * It is defined by:
 * - @ref a : first vertex,
 * - @ref b : second vertex,
 * - @ref c : third vertex,
 * - @ref normal : stored oriented face normal.
 *
 * ## Geometric semantics
 * The triangle is the finite planar region enclosed by the three line segments:
 * - `(a, b)`
 * - `(b, c)`
 * - `(c, a)`
 *
 * It supports:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - geometry-type reporting,
 * - vertex updates,
 * - barycentric coordinate evaluation.
 *
 * ## Normal handling
 * The stored @ref normal may be:
 * - derived from the vertices automatically, or
 * - explicitly supplied through the builder.
 *
 * The exact precedence and normalization policy are implementation-defined in
 * `triangle.hpp`.
 *
 * ## Operator caching
 * The class maintains an internal cached @ref TriangleGeometryOperator that stores
 * pointers to the triangle data. This operator is rebound whenever the object is
 * copied, moved, or otherwise reconstructed so that query paths remain valid.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Triangle final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Triangle requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Triangle.
     *
     * @details
     * The builder stages vertices and an optional explicit normal, validates them,
     * and constructs either:
     * - a triangle by value, or
     * - a host-owned shared pointer to a triangle.
     */
    class Builder;

public:
    /**
     * @brief First triangle vertex.
     */
    Vector3<T> a {};

    /**
     * @brief Second triangle vertex.
     */
    Vector3<T> b {};

    /**
     * @brief Third triangle vertex.
     */
    Vector3<T> c {};

    /**
     * @brief Stored oriented face normal.
     *
     * @details
     * May be derived from the vertex winding or explicitly supplied.
     */
    Vector3<T> normal { T(0), T(0), T(1) };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a default-initialized triangle. The exact geometric meaning of
     * the default vertices is implementation-defined.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle() noexcept;

    /**
     * @brief Construct a triangle from three explicit vertices.
     *
     * @param a_ First vertex.
     * @param b_ Second vertex.
     * @param c_ Third vertex.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

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
     * Copies vertex and normal data and rebinds the cached operator so that its
     * internal pointers reference this object rather than the source object.
     *
     * @param other Source triangle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(const Triangle& other) noexcept;

    /**
     * @brief Move constructor.
     *
     * @details
     * Moves vertex and normal data and rebinds the cached operator to this object.
     *
     * @param other Source triangle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(Triangle&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Copies triangle data and refreshes the cached operator binding.
     *
     * @param other Source triangle.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Triangle&
    operator=(const Triangle& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @details
     * Moves triangle data and refreshes the cached operator binding.
     *
     * @param other Source triangle.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Triangle&
    operator=(Triangle&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~Triangle() override = default;

    /**
     * @brief Create a geometry operator bound to this triangle.
     *
     * @details
     * Returns a lightweight non-owning operator that references this triangle's
     * geometric data.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Compute the closest point on the triangle to a query point.
     *
     * @param p Query point.
     * @return Closest point on the triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest outward normal associated with the triangle.
     *
     * @param p Query point.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the triangle.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the triangle within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the triangle surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return the centroid of the triangle.
     *
     * @return Triangle centroid.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of the triangle.
     *
     * @return Axis-aligned bounding box enclosing the triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this triangle is valid.
     *
     * @details
     * Typical validity checks include:
     * - finite vertices,
     * - non-degenerate area,
     * - finite stored normal.
     *
     * @return `true` if the triangle is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a triangle.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

    /**
     * @brief Replace the triangle vertices.
     *
     * @details
     * Updates the geometry to use the supplied vertices and typically refreshes
     * dependent quantities such as the stored normal and cached operator binding.
     *
     * @param a_ First vertex.
     * @param b_ Second vertex.
     * @param c_ Third vertex.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    /**
     * @brief Compute barycentric coordinates of a point with respect to the triangle.
     *
     * @details
     * Computes coefficients \f$(u,v,w)\f$ such that:
     * \f[
     * \mathbf{p} = u\mathbf{a} + v\mathbf{b} + w\mathbf{c},
     * \qquad
     * u+v+w=1.
     * \f]
     *
     * The result may be used for:
     * - point-in-triangle testing,
     * - interpolation over the triangle,
     * - proximity reasoning.
     *
     * @param p Query point.
     * @param u Output barycentric weight associated with vertex @ref a.
     * @param v Output barycentric weight associated with vertex @ref b.
     * @param w Output barycentric weight associated with vertex @ref c.
     * @return `true` if barycentric coordinates were successfully computed; otherwise `false`.
     *
     * @note
     * Degenerate triangles may cause this computation to fail.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept;

private:
    /// @brief Allow the builder to configure triangle internals directly.
    friend class Builder;

    /**
     * @brief Bind the cached operator to this triangle's storage.
     *
     * @details
     * Refreshes the raw-pointer fields of @ref _operator so that they reference
     * this instance's geometric data.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    /**
     * @brief Cached runtime operator bound to this triangle.
     *
     * @details
     * Stores raw pointers to the triangle vertices and associated auxiliary data.
     */
    mutable TriangleGeometryOperator<T> _operator {};
};

/**
 * @brief Fluent builder for @ref Triangle.
 *
 * @details
 * The builder provides a controlled construction path for triangles by staging:
 * - the three vertices,
 * - an optional explicit surface normal.
 *
 * ## Typical usage
 * @code
 * auto tri = atlas::TriangleF::builder()
 *     .with_vertices({0.0f, 0.0f, 0.0f},
 *                    {1.0f, 0.0f, 0.0f},
 *                    {0.0f, 1.0f, 0.0f})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - vertices are finite,
 * - the triangle is non-degenerate,
 * - an explicitly supplied normal is finite and non-zero.
 *
 * The exact validation rules are implementation-defined in `triangle.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Triangle<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes staged vertices to their default values and leaves the
     * explicit normal unset.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Triangle by value after validation.
     *
     * @return Constructed triangle value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Triangle<T>
    build() const;

    /**
     * @brief Build a configured @ref Triangle in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Triangle<T>>` owning the constructed triangle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Triangle<T>>
    make_host_shared() const;

    /**
     * @brief Set the first triangle vertex.
     *
     * @param a_ Staged first vertex.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_a(const Vector3<T>& a_) noexcept;

    /**
     * @brief Set the second triangle vertex.
     *
     * @param b_ Staged second vertex.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_b(const Vector3<T>& b_) noexcept;

    /**
     * @brief Set the third triangle vertex.
     *
     * @param c_ Staged third vertex.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_c(const Vector3<T>& c_) noexcept;

    /**
     * @brief Set all three triangle vertices at once.
     *
     * @param a_ Staged first vertex.
     * @param b_ Staged second vertex.
     * @param c_ Staged third vertex.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    /**
     * @brief Set an explicit surface normal.
     *
     * @details
     * This overrides the default behavior of deriving the normal from the vertex
     * winding, subject to the implementation policy.
     *
     * @param normal_ Staged explicit normal.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged vertices and optional normal.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending first vertex.
     */
    Vector3<T> _a {};

    /**
     * @brief Pending second vertex.
     */
    Vector3<T> _b {};

    /**
     * @brief Pending third vertex.
     */
    Vector3<T> _c {};

    /**
     * @brief Pending explicit normal.
     *
     * @details
     * When unset, the implementation typically derives the normal from the vertices.
     */
    std::optional<Vector3<T>> _normal;
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::geometry::Triangle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Triangle = geometry::Triangle<T>;

/**
 * @brief Common specialization of @ref atlas::geometry::Triangle for `float`.
 */
using TriangleF = geometry::Triangle<float>;

/**
 * @brief Common specialization of @ref atlas::geometry::Triangle for `double`.
 */
using TriangleD = geometry::Triangle<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::geometry::Triangle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using TriangleHostPtr = atlas::host_shared_ptr<geometry::Triangle<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::geometry::Triangle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using TriangleDevicePtr = atlas::device_shared_ptr<geometry::Triangle<T>>;

} // namespace atlas

#include <atlas/geometry/triangle.hpp>