#pragma once

/**
 * @file geometry.h
 * @brief Abstract base interface for geometric primitives in Atlas.
 *
 * @details
 * This header defines @ref atlas::geometry::Geometry, the **common abstract base class**
 * for all geometric primitives used in Atlas (e.g., Box, Sphere, Plane, Triangle,
 * TriangleMesh, Cylinder).
 *
 * The Geometry interface provides a **minimal but complete contract** for:
 * - spatial queries (closest point, normal, signed distance),
 * - bounding volume queries (AABB),
 * - centroid computation,
 * - validity checking,
 * - construction of lightweight *operator objects* for tracing and querying.
 *
 * ## Design goals
 * - **Polymorphic on the host**: Geometry objects are typically managed via
 *   `host_shared_ptr<Geometry<T>>`.
 * - **GPU-friendly execution model**: expensive virtual dispatch is avoided on device
 *   code by converting geometry into plain POD-style operators
 *   (see @ref QueryOperator and @ref TraceOperator).
 * - **Clear separation of responsibilities**:
 *   - Geometry objects own parameters and validate consistency.
 *   - Operators perform raw math and are suitable for kernels.
 *
 * ## Host vs Device semantics
 * - Operator construction (`make_query_operator`, `make_trace_operator`) is **host-only**.
 * - Query evaluation functions are marked `ATLAS_ALL_DEVICE` and may be called from
 *   host or device code.
 *
 * ## Typical usage
 * @code
 * atlas::GeometryHostPtr<float> geom = atlas::BoxF::builder()
 *     .with_bounds({0,0,0}, {1,1,1})
 *     .make_host_shared();
 *
 * auto qop = geom->make_query_operator(); // host
 *
 * Vector3f p = qop.closest_point(x);      // host or device
 * @endcode
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 */

#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/trace_operator.h>
#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Abstract base class for all geometry types.
 *
 * @details
 * `Geometry` defines the **required interface** that all concrete geometry classes
 * must implement. It is intentionally small and stable.
 *
 * ### Key responsibilities
 * Each derived geometry must provide:
 * - Construction of a @ref TraceOperator for ray–geometry intersection.
 * - Construction of a @ref QueryOperator for closest-point and distance queries.
 * - Direct evaluation routines for closest point, normal, signed distance, centroid,
 *   and axis-aligned bounding box.
 * - A validity predicate.
 *
 * ### Why operators?
 * Geometry objects may contain rich state, ownership, and host-only resources.
 * To execute efficiently on the GPU, Atlas extracts **operator objects** that:
 * - are trivially copyable,
 * - contain only raw values or pointers,
 * - avoid virtual dispatch.
 *
 * These operators are then uploaded or embedded into device-side data structures.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Geometry {
    static_assert(std::is_floating_point_v<T>, "Geometry requires a floating-point T");

public:
    /// @brief Default constructor.
    Geometry() = default;

    /**
     * @brief Virtual destructor.
     *
     * @note
     * Required for safe polymorphic deletion through base pointers.
     */
    virtual ~Geometry() = default;

    /* ------------------------------------------------------------------
     * Operator construction (host-only)
     * ------------------------------------------------------------------ */

    /**
     * @brief Construct a trace operator for this geometry.
     *
     * @details
     * The returned @ref TraceOperator is a lightweight, non-owning object suitable
     * for ray tracing and intersection tests.
     *
     * Implementations typically:
     * - store pointers to geometry parameters,
     * - or copy small POD parameters directly.
     *
     * @return A trace operator bound to this geometry.
     *
     * @note
     * Host-only: this function may bind host memory addresses.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual atlas::spatial::TraceOperator<T>
    make_trace_operator() const = 0;

    /**
     * @brief Construct a query operator for this geometry.
     *
     * @details
     * The returned @ref QueryOperator is used for closest-point queries,
     * signed distance evaluation, and related operations.
     *
     * @return A query operator bound to this geometry.
     *
     * @note
     * Host-only: this function may bind host memory addresses.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual atlas::geometry::QueryOperator<T>
    make_query_operator() const = 0;

    /* ------------------------------------------------------------------
     * Geometric queries (host + device)
     * ------------------------------------------------------------------ */

    /**
     * @brief Compute the closest point on the geometry to a query point.
     *
     * @param p Query point in world coordinates.
     * @return Closest point on the geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    /**
     * @brief Compute the normal at the closest point to a query point.
     *
     * @details
     * For smooth surfaces, this is typically the surface normal.
     * For non-smooth geometry (edges, corners), the returned normal
     * follows implementation-defined conventions.
     *
     * @param p Query point in world coordinates.
     * @return Normal vector (typically unit length).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    /**
     * @brief Compute the signed distance from a point to the geometry.
     *
     * @details
     * The sign convention is geometry-dependent:
     * - Closed solids often return negative values inside.
     * - Open surfaces may return unsigned or locally signed distances.
     *
     * @param p Query point in world coordinates.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept
        = 0;

    /**
     * @brief Test whether a point lies inside the geometry within a tolerance.
     *
     * @details
     * The interpretation follows the geometry's signed-distance convention.
     * For closed solids this typically means negative distance is inside.
     *
     * @param p Query point in world coordinates.
     * @param tolerance Allowed positive slack around the interior boundary.
     * @return `true` if the point is classified as inside.
     *
     * @note
     * Host-only in the current design because implementations forward through
     * @ref make_query_operator(), which is also host-only.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance ) const noexcept
        = 0;

    /**
     * @brief Test whether a point lies on the surface within a tolerance.
     *
     * @param p Query point in world coordinates.
     * @param tolerance Allowed absolute deviation from the surface.
     * @return `true` if the point is within the surface band.
     *
     * @note
     * Host-only in the current design because implementations forward through
     * @ref make_query_operator(), which is also host-only.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance ) const noexcept
        = 0;

    /**
     * @brief Compute a representative centroid of the geometry.
     *
     * @details
     * Interpretation depends on geometry type:
     * - analytic centroid for primitives,
     * - average or area-weighted centroid for meshes,
     * - bounding-box centroid for fallback implementations.
     *
     * @return Centroid position.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    centroid() const noexcept = 0;

    /**
     * @brief Compute the axis-aligned bounding box (AABB) of the geometry.
     *
     * @details
     * The bounding box must fully enclose the geometry.
     *
     * @return Axis-aligned bounding box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept = 0;

    /**
     * @brief Check whether the geometry parameters are valid.
     *
     * @details
     * Typical checks include:
     * - finite parameters,
     * - positive radii/lengths,
     * - non-degenerate shapes,
     * - consistent internal state.
     *
     * @return `true` if geometry is valid and safe to query.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept
        = 0;

    /**
     * @brief Get the runtime type tag of this geometry.
     *
     * @details
     * This tag identifies the concrete geometry type (e.g., Box, Sphere, etc.)
     * and is used for type-based dispatch in operators and kernels.
     *
     * @return Geometry type tag.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual GeometryType
    type() const noexcept
        = 0;
};

} // namespace atlas::geometry

/* ====================================================================== */
/* Public aliases                                                          */
/* ====================================================================== */

namespace atlas {

template <typename T>
using Geometry = geometry::Geometry<T>;

template <typename T>
using GeometryHostPtr = atlas::host_shared_ptr<geometry::Geometry<T>>;

template <typename T>
using GeometryDevicePtr = atlas::device_shared_ptr<geometry::Geometry<T>>;

} // namespace atlas
