#pragma once

/**
 * @file geometry_operator.h
 * @brief Declares the backend-portable tagged-union wrapper over all supported geometry operators.
 *
 * @details
 * This header defines @ref atlas::geometry::GeometryOperator, a value-type
 * runtime geometry wrapper that erases the concrete primitive type into a
 * compact tagged-union representation.
 *
 * The primary purpose of this abstraction is to bridge:
 * - host-side polymorphic geometry objects derived from @ref Geometry, and
 * - backend/device execution code that requires a copyable, non-virtual,
 *   ownership-free geometry representation.
 *
 * ## Design overview
 * Rather than storing geometries through inheritance or dynamic allocation,
 * @ref GeometryOperator uses:
 * - a runtime type discriminator @ref GeometryType, and
 * - a union of concrete lightweight geometry operator payloads.
 *
 * Each concrete payload is itself a non-owning operator type that references the
 * underlying primitive parameters through raw pointers or other lightweight
 * fields. This allows @ref GeometryOperator to:
 * - be copied into kernels,
 * - be embedded inside other runtime structures,
 * - dispatch queries at runtime without virtual calls,
 * - preserve a uniform interface across heterogeneous primitive types.
 *
 * ## Supported primitive families
 * The operator currently supports the following concrete variants:
 * - box,
 * - circle,
 * - cylinder,
 * - plane,
 * - sphere,
 * - square,
 * - triangle,
 * - triangle mesh.
 *
 * ## Query interface
 * Once constructed, a @ref GeometryOperator exposes a uniform set of geometric
 * queries, including:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box access,
 * - validity checks,
 * - ray tracing through @ref trace and @ref operator().
 *
 * ## Lifetime model
 * Since the union stores concrete operator payloads directly, the wrapper is
 * responsible for:
 * - tracking the active payload through @ref type,
 * - constructing the correct payload during initialization,
 * - copying the correct payload during copy construction and assignment.
 *
 * Concrete payload operators are generally lightweight and non-owning, so the
 * lifetime of any externally referenced primitive data must outlive the use of
 * the corresponding @ref GeometryOperator.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates, distances, and query results.
 */

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

namespace atlas::geometry {

/**
 * @brief Tagged-union wrapper over all supported lightweight geometry operators.
 *
 * @details
 * @ref GeometryOperator is the backend-portable runtime representation of a
 * geometry object in Atlas.
 *
 * It stores:
 * - a runtime type tag identifying the active primitive family,
 * - a union containing the corresponding concrete geometry operator payload.
 *
 * This makes it possible to pass heterogeneous geometry objects through a
 * single, value-type interface suitable for:
 * - device kernels,
 * - backend-parallel closures,
 * - runtime acceleration structures,
 * - collision and tracing subsystems.
 *
 * ## Internal representation
 * At any given time:
 * - @ref type identifies the active primitive variant,
 * - exactly one union member is expected to be active and consistent with
 *   that type tag.
 *
 * ## Construction paths
 * A @ref GeometryOperator may be:
 * - default-constructed,
 * - copy-constructed,
 * - copy-assigned,
 * - explicitly constructed from any supported concrete operator payload.
 *
 * ## Runtime dispatch
 * All query member functions dispatch to the currently active concrete operator
 * based on @ref type. The exact query semantics are therefore inherited from the
 * active payload type.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct GeometryOperator {
    /**
     * @brief Active geometry variant.
     *
     * @details
     * Identifies which concrete union member is currently active and therefore
     * which primitive type should handle geometric queries.
     */
    GeometryType type = GeometryType::Sphere;

    /**
     * @brief Tagged union storing the active concrete geometry operator payload.
     *
     * @details
     * Exactly one of these members is expected to be active at a time, as
     * determined by @ref type.
     */
    union {
        /**
         * @brief Box geometry operator payload.
         */
        BoxGeometryOperator<T> box;

        /**
         * @brief Circle geometry operator payload.
         */
        CircleGeometryOperator<T> circle;

        /**
         * @brief Cylinder geometry operator payload.
         */
        CylinderGeometryOperator<T> cylinder;

        /**
         * @brief Plane geometry operator payload.
         */
        PlaneGeometryOperator<T> plane;

        /**
         * @brief Sphere geometry operator payload.
         */
        SphereGeometryOperator<T> sphere;

        /**
         * @brief Square geometry operator payload.
         */
        SquareGeometryOperator<T> square;

        /**
         * @brief Triangle geometry operator payload.
         */
        TriangleGeometryOperator<T> triangle;

        /**
         * @brief Triangle-mesh geometry operator payload.
         */
        TriangleMeshGeometryOperator<T> triangle_mesh;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs the wrapper with a default active geometry variant.
     *
     * The default type tag is initialized to @ref GeometryType::Sphere, and the
     * corresponding union member is expected to be constructed according to the
     * implementation in `geometry_operator.hpp`.
     */
    ATLAS_ALL_DEVICE
    GeometryOperator() noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies the active type tag and constructs the matching concrete payload
     * from the source operator.
     *
     * @param other Source geometry operator.
     */
    ATLAS_ALL_DEVICE
    GeometryOperator(const GeometryOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Replaces the current active payload with the one referenced by @p other,
     * preserving consistency between the stored type tag and union state.
     *
     * @param other Source geometry operator.
     * @return `*this`.
     */
    ATLAS_ALL_DEVICE GeometryOperator&
    operator=(const GeometryOperator& other) noexcept;

    /**
     * @brief Construct the wrapper from a box geometry operator.
     *
     * @param op Box operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const BoxGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a circle geometry operator.
     *
     * @param op Circle operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const CircleGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a cylinder geometry operator.
     *
     * @param op Cylinder operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const CylinderGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a plane geometry operator.
     *
     * @param op Plane operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const PlaneGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a sphere geometry operator.
     *
     * @param op Sphere operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const SphereGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a square geometry operator.
     *
     * @param op Square operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const SquareGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a triangle geometry operator.
     *
     * @param op Triangle operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleGeometryOperator<T>& op);

    /**
     * @brief Construct the wrapper from a triangle-mesh geometry operator.
     *
     * @param op Triangle-mesh operator payload to store.
     */
    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleMeshGeometryOperator<T>& op);

    /**
     * @brief Dispatch the closest-point query to the active geometry variant.
     *
     * @details
     * Calls the corresponding @ref closest_point implementation of the active
     * concrete payload selected by @ref type.
     *
     * @param p Query point in world space.
     * @return Closest point on the active geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Dispatch the closest-normal query to the active geometry variant.
     *
     * @param p Query point in world space.
     * @return Closest geometric normal associated with the active geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Dispatch the signed-distance query to the active geometry variant.
     *
     * @param p Query point in world space.
     * @return Signed distance to the active geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Dispatch the inside-classification query to the active geometry variant.
     *
     * @param p Query point in world space.
     * @param tolerance Optional classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Dispatch the surface-classification query to the active geometry variant.
     *
     * @param p Query point in world space.
     * @param tolerance Optional surface-classification tolerance.
     * @return `true` if the point is classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Dispatch the centroid query to the active geometry variant.
     *
     * @return Centroid of the active geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Dispatch the bounding-box query to the active geometry variant.
     *
     * @return Axis-aligned bounding box enclosing the active geometry.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Dispatch the validity check to the active geometry variant.
     *
     * @return `true` if the active geometry payload is valid; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Dispatch the ray-tracing query to the active geometry variant.
     *
     * @details
     * Calls the corresponding @ref trace implementation of the active concrete
     * payload selected by @ref type.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the ray-geometry intersection result.
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

} // namespace atlas::geometry

#include <atlas/geometry/geometry_operator.hpp>
