#pragma once

/**
 * @file geometry.h
 * @brief Declares the abstract geometry interface and related convenience aliases used throughout Atlas.
 *
 * @details
 * This header defines @ref atlas::geometry::Geometry, the abstract base class
 * implemented by all queryable geometric primitives in Atlas.
 *
 * A geometry object represents a spatial primitive that supports a standard set
 * of geometric queries, including:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - containment and surface tests,
 * - centroid queries,
 * - axis-aligned bounding-box queries,
 * - runtime type identification,
 * - export of a value-type @ref GeometryOperator for backend-portable execution.
 *
 * ## Role in the Atlas geometry system
 * The @ref Geometry interface provides a uniform abstraction for analytic and
 * derived primitives such as:
 * - boxes,
 * - circles,
 * - cylinders,
 * - and any other supported queryable geometry type.
 *
 * This common interface allows higher-level systems to:
 * - manipulate heterogeneous geometry objects polymorphically on the host,
 * - request geometric queries without knowing the concrete primitive type,
 * - construct backend-portable operators through @ref make_geometry_operator.
 *
 * ## Host/device split
 * Atlas separates geometry usage into two complementary forms:
 * - the **owning polymorphic geometry object**, represented by @ref Geometry,
 * - the **value-type backend-friendly operator**, represented by
 *   @ref GeometryOperator.
 *
 * The owning object is convenient for host-side configuration, storage, and
 * polymorphic dispatch. The value-type operator is intended for backend code
 * where dynamic polymorphism and host ownership are undesirable.
 *
 * ## Query semantics
 * Each concrete geometry implementation defines the exact semantics of:
 * - what counts as "inside",
 * - how normals are selected in ambiguous cases,
 * - how signed distance is computed,
 * - what constitutes a valid primitive,
 * while adhering to the common interface defined here.
 *
 * ## Type requirements
 * The template parameter @p T must be a floating-point type. This is enforced
 * by a compile-time `static_assert`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates, distances, and geometric computations.
 */

#include <atlas/geometry/geometry_type.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Forward declaration of the value-type backend-portable geometry operator.
 *
 * @details
 * Concrete geometry objects can export a @ref GeometryOperator through
 * @ref Geometry::make_geometry_operator so runtime code can perform queries
 * without relying on host-side ownership or virtual dispatch.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct GeometryOperator;

/**
 * @brief Abstract base class for queryable geometric primitives.
 *
 * @details
 * @ref Geometry defines the common query interface implemented by all Atlas
 * geometry primitives.
 *
 * A concrete geometry type is expected to provide implementations for:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid computation,
 * - axis-aligned bounding-box generation,
 * - runtime type reporting,
 * - export of a backend-portable @ref GeometryOperator.
 *
 * ## Design intent
 * This interface enables host-side polymorphic use of heterogeneous geometries
 * while still supporting efficient backend execution through exported operator
 * objects.
 *
 * ## Typical usage
 * A caller may use a geometry object to:
 * - perform host-side point or distance queries directly,
 * - retrieve a bounding box for acceleration structures,
 * - classify points relative to a primitive,
 * - construct a @ref GeometryOperator for device-side runtime use.
 *
 * ## Validity expectations
 * Concrete implementations are responsible for defining what it means for a
 * primitive to be valid. Callers may use @ref is_valid before relying on a
 * geometry instance in downstream computations.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Geometry {
    static_assert(std::is_floating_point_v<T>, "Geometry requires a floating-point T");

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs the abstract geometry base with default-initialized state.
     */
    Geometry() = default;

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Declared virtual so concrete geometry instances can be destroyed correctly
     * through base-class pointers.
     */
    virtual ~Geometry() = default;

    /**
     * @brief Create a value-type operator for host/device runtime queries.
     *
     * @details
     * Returns a backend-portable @ref GeometryOperator that captures the concrete
     * geometry state in a form suitable for runtime code without host-side
     * ownership or virtual dispatch.
     *
     * @return Value-type geometry operator bound to this geometry instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual atlas::geometry::GeometryOperator<T>
    make_geometry_operator() const = 0;

    /**
     * @brief Compute the closest point on the geometry to a query point.
     *
     * @param p Query point in world space.
     * @return Closest point on the geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    /**
     * @brief Compute the closest geometric normal associated with a query point.
     *
     * @details
     * The exact normal-selection rule is implementation-defined for each
     * primitive, especially in edge, corner, or otherwise ambiguous cases.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    /**
     * @brief Compute the signed distance from a query point to the geometry.
     *
     * @details
     * The exact sign convention is defined by the concrete implementation, but
     * this function is intended to provide a consistent distance-style scalar
     * query relative to the primitive.
     *
     * @param p Query point in world space.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    /**
     * @brief Test whether a point lies inside the geometry within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    /**
     * @brief Test whether a point lies on the geometry surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface-classification tolerance.
     * @return `true` if the point is classified as being on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    /**
     * @brief Return the centroid of the geometry.
     *
     * @return Geometric centroid of the primitive.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    centroid() const noexcept = 0;

    /**
     * @brief Return the axis-aligned bounding box of the geometry.
     *
     * @details
     * This bounding box encloses the concrete primitive in the same coordinate
     * frame in which the geometry is queried.
     *
     * @return Axis-aligned bounding box enclosing the geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept = 0;

    /**
     * @brief Return whether the geometry is valid.
     *
     * @details
     * Concrete implementations define validity according to their own geometric
     * invariants, such as non-negative radii, non-degenerate normals, or
     * component-wise ordered bounds.
     *
     * @return `true` if the geometry is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept = 0;

    /**
     * @brief Return the runtime geometry type tag.
     *
     * @details
     * Identifies the concrete primitive category in polymorphic contexts.
     *
     * @return Corresponding @ref GeometryType value.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual GeometryType
    type() const noexcept = 0;
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::geometry::GeometryOperator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GeometryOperator = geometry::GeometryOperator<T>;

/**
 * @brief Convenience alias for @ref atlas::geometry::Geometry.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Geometry = geometry::Geometry<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::geometry::Geometry.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GeometryHostPtr = atlas::host_shared_ptr<geometry::Geometry<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::geometry::Geometry.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GeometryDevicePtr = atlas::device_shared_ptr<geometry::Geometry<T>>;

} // namespace atlas

namespace atlas::spatial {

}