#pragma once

/**
 * @file triangle.h
 * @brief Single-triangle geometry primitive and builder.
 *
 * @details
 * This header defines @ref atlas::geometry::Triangle, a single triangle geometry primitive
 * represented by three vertices (`a`, `b`, `c`) and an (optionally cached) normal vector.
 *
 * `Triangle` implements the @ref atlas::Geometry interface and provides:
 * - query and trace operator construction for interop with spatial systems,
 * - closest-point and closest-normal evaluation,
 * - signed distance evaluation (with sign convention tied to the triangle normal),
 * - centroid and AABB computation,
 * - validity checks,
 * - host-side vertex setter and barycentric coordinate evaluation.
 *
 * ## Orientation and normal
 * A triangle is oriented by the vertex winding order. A common convention is that
 * (a, b, c) ordered counter-clockwise as viewed from the "front" produces an outward normal:
 * \f[
 *   \mathbf{n} \propto (b-a) \times (c-a)
 * \f]
 *
 * This class stores a `normal` member. Depending on implementation policy:
 * - it may be computed automatically when vertices are set, or
 * - it may be user-provided via the builder, or
 * - it may be treated as a cached value that must be kept consistent by the caller.
 *
 * Confirm the exact behavior in `triangle.hpp`.
 *
 * ## Barycentric coordinates
 * The member function @ref barycentric computes weights (u,v,w) such that:
 * \f[
 *   \mathbf{p} = u\mathbf{a} + v\mathbf{b} + w\mathbf{c}, \quad u+v+w = 1
 * \f]
 * For points in the triangle (including edges), typically u,v,w are all in [0,1].
 *
 * ## Host/device
 * - Geometry evaluation methods are `ATLAS_ALL_DEVICE`.
 * - Operator construction and vertex setters are host-only (`ATLAS_HOST`) as declared here.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * - Degenerate triangles (zero area) are invalid; use @ref is_valid() to detect.
 * - For consistent signed-distance sign, ensure `normal` matches the vertex winding.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <optional>
#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct TriangleGeometryOperator {
    const atlas::math::Vector<T, 3>* a      = nullptr;
    const atlas::math::Vector<T, 3>* b      = nullptr;
    const atlas::math::Vector<T, 3>* c      = nullptr;
    const atlas::math::Vector<T, 3>* n      = nullptr;
    const atlas::math::Vector<T, 3>* normal = nullptr;

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
 * @brief Single triangle geometry primitive.
 *
 * @details
 * `Triangle` stores three vertices and a normal and supports standard geometric queries.
 *
 * ### Closest point
 * Closest-point computations commonly use Voronoi region classification:
 * - if projection falls inside, return projection onto the triangle plane
 * - otherwise return closest point on an edge or vertex
 *
 * ### Signed distance
 * A typical signed distance uses:
 * - unsigned distance = ||p - closest_point(p)||
 * - sign determined by the triangle plane (dot(p - a, normal)) (implementation-defined)
 *
 * Because a triangle is not a closed surface, signed distance can be ambiguous far from the
 * triangle plane; confirm intended semantics in `triangle.hpp`.
 *
 * ### Bounds
 * The triangle AABB is the component-wise min/max of its vertices.
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
     */
    class Builder;

public:
    /**
     * @brief First vertex of the triangle.
     *
     * @details
     * Together with @ref b and @ref c defines the triangle geometry and orientation (winding).
     */
    Vector3<T> a {};

    /**
     * @brief Second vertex of the triangle.
     */
    Vector3<T> b {};

    /**
     * @brief Third vertex of the triangle.
     */
    Vector3<T> c {};

    /**
     * @brief Cached triangle normal.
     *
     * @details
     * Typically expected to be proportional to `(b-a) x (c-a)` and often normalized.
     *
     * @note
     * Whether this is auto-computed or treated as user-managed cache is implementation-defined.
     */
    Vector3<T> normal { T(0), T(0), T(1) };

    /**
     * @brief Default constructor.
     *
     * @details
     * Leaves vertices at default-initialized values. The triangle is likely invalid until
     * vertices are set.
     */
    Triangle() noexcept = default;

    /**
     * @brief Construct a triangle from three vertices.
     *
     * @param a_ Vertex A.
     * @param b_ Vertex B.
     * @param c_ Vertex C.
     *
     * @note
     * The implementation may compute/update @ref normal from these vertices.
     * Confirm in `triangle.hpp`.
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

    /// @brief Defaulted copy constructor.
    Triangle(const Triangle&) noexcept = default;

    /// @brief Virtual destructor (geometry base).
    ~Triangle() override = default;

    /**
     * @brief Create a trace operator bound to this triangle.
     *
     * @details
     * Returns a @ref GeometryOperator usable by tracing systems to intersect rays with the triangle.
     * Implementations commonly store non-owning pointers to vertices and/or precomputed normal.
     *
     * @return Trace operator referencing this triangle.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */

    /**
     * @brief Create a query operator bound to this triangle.
     *
     * @details
     * Returns a @ref GeometryOperator usable by query systems for closest point/normal and distance.
     *
     * @return Query operator referencing this triangle.
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Compute the closest point on the triangle to a query point.
     *
     * @param p Query point.
     * @return Closest point on the triangle (including edges and vertices).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest normal corresponding to the closest feature.
     *
     * @details
     * Many implementations return the triangle's (possibly normalized) plane normal regardless
     * of the closest feature. Some may handle edge/vertex regions differently.
     *
     * @param p Query point (may be unused).
     * @return A normal vector, typically aligned with @ref normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the triangle.
     *
     * @details
     * Typical implementation:
     * - compute closest point c = closest_point(p)
     * - unsigned distance = ||p - c||
     * - sign determined by the triangle plane using @ref normal
     *
     * Because a triangle is not a closed solid, interpret signed distance with care.
     *
     * @param p Query point.
     * @return Signed distance value (convention implementation-defined).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies on the triangle's negative signed side within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Allowed positive slack relative to the oriented triangle surface.
     * @return `true` if the point is classified as inside by the triangle query rule.
     *
     * @note
     * Because a triangle is an open surface, this classification is local and orientation-dependent.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the triangle surface within a tolerance band.
     *
     * @param p Query point.
     * @param tolerance Allowed absolute deviation from the triangle surface.
     * @return `true` if the point is classified as on the surface.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Centroid of the triangle.
     *
     * @details
     * The centroid (barycenter) is:
     * \f[
     *   \frac{a+b+c}{3}
     * \f]
     *
     * @return Centroid position.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Axis-aligned bounding box of the triangle.
     *
     * @details
     * Computed as component-wise min/max over vertices a, b, c.
     *
     * @return AABB bounds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether the triangle parameters are valid.
     *
     * @details
     * Typical checks include:
     * - finite vertices,
     * - non-degenerate area (edges not collinear; cross product magnitude > 0),
     * - normal is finite and (optionally) consistent with vertices.
     *
     * @return `true` if valid; otherwise `false`.
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

    /**
     * @brief Set triangle vertices (host-side convenience).
     *
     * @details
     * Updates vertices @ref a, @ref b, @ref c. The implementation may also update @ref normal.
     *
     * @param a_ Vertex A.
     * @param b_ Vertex B.
     * @param c_ Vertex C.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    /**
     * @brief Compute barycentric coordinates of a point w.r.t. this triangle.
     *
     * @details
     * Computes weights (u,v,w) such that:
     * \f[
     *   p = u a + v b + w c,\quad u+v+w=1
     * \f]
     *
     * The function typically returns `false` for degenerate triangles (zero area), and may
     * return `true` even if the point lies outside the triangle (u,v,w then can be outside [0,1]).
     *
     * @param p Query point.
     * @param u Output weight for vertex a.
     * @param v Output weight for vertex b.
     * @param w Output weight for vertex c.
     * @return `true` if barycentrics could be computed (non-degenerate); otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept;

private:
    /// @brief Allow builder to configure internals.
    friend class Builder;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

/**
 * @brief Fluent builder for @ref Triangle.
 *
 * @details
 * Stages vertices (and optionally a normal) before building a `Triangle`.
 *
 * ## Typical usage
 * @code
 * atlas::TriangleF tri = atlas::TriangleF::builder()
 *     .with_vertices({0,0,0}, {1,0,0}, {0,1,0})
 *     .build();
 * @endcode
 *
 * ## Normal handling
 * If you provide @ref with_normal, the builder will store it in the triangle. Whether the
 * builder validates consistency between the provided normal and vertex winding is policy-dependent.
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks:
 * - vertices finite
 * - triangle non-degenerate
 * - normal finite (and optionally non-zero)
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Triangle<T>::Builder final {
public:
    /// @brief Default constructor.
    Builder() = default;

    /**
     * @brief Build a configured @ref Triangle (by value).
     *
     * @return Constructed triangle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Triangle<T>
    build() const;

    /**
     * @brief Build a configured @ref Triangle in a host_shared_ptr.
     *
     * @return Shared pointer owning the constructed triangle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Triangle<T>>
    make_host_shared() const;

    /// @brief Set vertex A.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_a(const Vector3<T>& a_) noexcept;

    /// @brief Set vertex B.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_b(const Vector3<T>& b_) noexcept;

    /// @brief Set vertex C.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_c(const Vector3<T>& c_) noexcept;

    /// @brief Set all vertices at once.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    /**
     * @brief Provide an explicit normal.
     *
     * @details
     * Stores a normal vector into the triangle. The vector is typically expected to be
     * unit-length and consistent with the vertex winding, but enforcement is implementation-defined.
     *
     * @param normal_ Normal vector.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

private:
    /// @brief Validate staged parameters.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _a {};
    Vector3<T> _b {};
    Vector3<T> _c {};
    std::optional<Vector3<T>> _normal;
};

} // namespace atlas::geometry

namespace atlas {

template <typename T>
using Triangle  = geometry::Triangle<T>;
using TriangleF = geometry::Triangle<float>;
using TriangleD = geometry::Triangle<double>;

template <typename T>
using TriangleHostPtr = atlas::host_shared_ptr<geometry::Triangle<T>>;
template <typename T>
using TriangleDevicePtr = atlas::device_shared_ptr<geometry::Triangle<T>>;

} // namespace atlas

#include <atlas/geometry/triangle.hpp>
