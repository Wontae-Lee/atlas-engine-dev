#pragma once

/**
 * @file query_operator.h
 * @brief Device-friendly geometry query operators (closest point/normal, signed distance, centroid, bounds).
 *
 * @details
 * This header defines a family of **non-owning** query operators used to evaluate geometric
 * proximity information against different primitive types (and triangle meshes).
 *
 * The operators are designed to be:
 * - **POD-like** (simple structs) so they can be copied cheaply and passed into kernels,
 * - **non-owning**: they store raw pointers to geometry parameters (or arrays),
 * - **callable from host and device** (`ATLAS_ALL_DEVICE`) for hot-path queries.
 *
 * ## What is a "QueryOperator"?
 * A `QueryOperator` is a lightweight handle that provides a uniform API:
 * - `closest_point(p)`
 * - `closest_normal(p)`
 * - `signed_distance(p)`
 * - `centroid()`
 * - `bound()`
 * - `is_valid()`
 *
 * It enables generic algorithms (collision response, distance fields, constraints, etc.)
 * to operate on different geometry types without templating on the geometry class itself.
 *
 * ## Ownership and lifetime
 * Query operators do **not** own the referenced data. For example:
 * - @ref BoxQueryOperator holds pointers to `lower_corner` and `upper_corner`
 * - @ref SphereQueryOperator holds pointers to `center` and `radius`
 * - @ref TriangleMeshQueryOperator holds pointers to `vertices` and `indices`
 *
 * The referenced storage must remain valid for the lifetime of the operator usage.
 * This is especially important when uploading operators to the GPU: pointers must be
 * device-addressable if used in device code.
 *
 * ## Dispatch strategy
 * The top-level @ref atlas::geometry::QueryOperator is a tagged union holding one of the
 * concrete operators and a @ref QueryOpType tag. Member functions dispatch on that tag.
 *
 * @warning
 * Because `QueryOperator` uses a union, correct construction/copy/assignment must ensure
 * the active member matches the tag. Do not mutate `type` manually.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */

#include <atlas/geometry/geometry_type.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::geometry {

/* ====================================================================== */
/* BoxQueryOperator                                                        */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for an axis-aligned box (AABB).
 *
 * @details
 * Stores pointers to:
 * - `lower_corner` : component-wise minimum
 * - `upper_corner` : component-wise maximum
 *
 * Typical behavior:
 * - closest point is computed by clamping each component to [lo, hi]
 * - closest normal depends on whether the query point is inside/outside and which face is closest
 * - signed distance uses an AABB SDF (implementation-defined exact formulation)
 *
 * @tparam T Floating-point scalar type.
 *
 * @note
 * Pointers may be null; use @ref is_valid() before calling other methods.
 */
template <typename T>
struct BoxQueryOperator {
    /// @brief Pointer to lower (minimum) corner.
    const atlas::math::Vector<T, 3>* lower_corner = nullptr;

    /// @brief Pointer to upper (maximum) corner.
    const atlas::math::Vector<T, 3>* upper_corner = nullptr;

    /**
     * @brief Closest point on (or in) the box to a query point.
     *
     * @param p Query point.
     * @return Closest point per the operator's policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Closest outward normal of the box relative to a query point.
     *
     * @param p Query point.
     * @return Normal vector (typically unit length).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Signed distance from a point to the box.
     *
     * @param p Query point.
     * @return Signed distance (convention implementation-defined; commonly + outside, - inside).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Centroid of the box.
     *
     * @return (lo + hi) / 2.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief AABB bounds of the box.
     *
     * @return Bounding box for broad-phase operations (often itself).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Returns whether the operator has valid, usable parameters.
     *
     * @details
     * Typical checks:
     * - pointers are non-null
     * - finite corner values
     * - lower <= upper component-wise
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* SphereQueryOperator                                                     */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for a sphere.
 *
 * @details
 * Stores pointers to:
 * - `center` : sphere center
 * - `radius` : sphere radius
 *
 * Typical behavior:
 * - closest point: center + normalize(p-center) * radius (outside),
 *   and policy-defined behavior inside
 * - normal: outward radial direction
 * - signed distance: ||p-center|| - radius
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SphereQueryOperator {
    /// @brief Pointer to sphere center.
    const atlas::math::Vector<T, 3>* center = nullptr;

    /// @brief Pointer to sphere radius.
    const T* radius = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* PlaneQueryOperator                                                      */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for an infinite plane.
 *
 * @details
 * Stores pointers to:
 * - `normal` : plane normal (not necessarily unit length unless enforced)
 * - `offset` : plane offset `d` in `normal·x = d` (common convention)
 *
 * @tparam T Floating-point scalar type.
 *
 * @note
 * Planes are infinite; centroid and bounds are implementation-defined placeholders.
 */
template <typename T>
struct PlaneQueryOperator {
    /// @brief Pointer to plane normal vector.
    const atlas::math::Vector<T, 3>* normal = nullptr;

    /// @brief Pointer to plane offset.
    const T* offset = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* CylinderQueryOperator                                                   */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for a finite cylinder (axis-aligned in local space).
 *
 * @details
 * Stores pointers to:
 * - `center` : cylinder center
 * - `radius` : cylinder radius
 * - `height` : cylinder height
 *
 * Axis convention (typical): cylinder axis aligned with Z in the query frame, with caps at
 * z = center.z ± height/2. Confirm in implementation.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct CylinderQueryOperator {
    /// @brief Pointer to cylinder center.
    const atlas::math::Vector<T, 3>* center = nullptr;

    /// @brief Pointer to cylinder radius.
    const T* radius = nullptr;

    /// @brief Pointer to cylinder height.
    const T* height = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* TriangleQueryOperator                                                   */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for a single triangle.
 *
 * @details
 * Stores pointers to:
 * - vertices `a`, `b`, `c`
 * - (optional) triangle normal `n` (can be precomputed for speed)
 *
 * Typical behavior:
 * - closest point computed using point-triangle projection (Voronoi region classification)
 * - normal returned as either the stored `n` or computed cross product direction (implementation-defined)
 * - signed distance may measure distance to the triangle surface, with sign determined by the normal
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct TriangleQueryOperator {
    const atlas::math::Vector<T, 3>* a = nullptr;
    const atlas::math::Vector<T, 3>* b = nullptr;
    const atlas::math::Vector<T, 3>* c = nullptr;
    const atlas::math::Vector<T, 3>* n = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* TriangleMeshQueryOperator                                               */
/* ====================================================================== */

/**
 * @brief Non-owning query operator for a triangle mesh (indexed).
 *
 * @details
 * Stores pointers to mesh data:
 * - `vertices` : array of vertex positions
 * - `indices`  : array of triangle indices (implementation-defined stride; typically 3 ints per triangle)
 * - `triangle_count` : number of triangles in the mesh
 *
 * The operator evaluates queries by iterating over triangles and taking the best result.
 * This is O(N) in triangle count and is intended as a baseline / fallback; production usage
 * often pairs this with spatial acceleration structures.
 *
 * @tparam T Floating-point scalar type.
 *
 * @warning
 * Performance can be poor for large meshes if no acceleration is used.
 */
template <typename T>
struct TriangleMeshQueryOperator {
    /// @brief Pointer to contiguous vertex position array.
    const atlas::math::Vector<T, 3>* vertices = nullptr;

    /// @brief Pointer to contiguous index array.
    const int* indices = nullptr;

    /// @brief Number of triangles in the mesh.
    int triangle_count = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

/* ====================================================================== */
/* QueryOperator (tagged union wrapper)                                    */
/* ====================================================================== */

/**
 * @brief Tagged union wrapper that dispatches geometric queries at runtime.
 *
 * @details
 * `QueryOperator` holds:
 * - a runtime tag @ref type
 * - a union of concrete operator structs
 *
 * Member functions dispatch to the active operator based on @ref type, providing a uniform
 * query API regardless of the underlying shape.
 *
 * ## Construction
 * - Default construction typically yields a valid operator of some type (often Sphere), but the
 *   referenced pointers may still be null; always use @ref is_valid().
 * - Host constructors from specific operator types select the appropriate union member.
 *
 * ## Union safety
 * Since this uses a union, copy/assignment must preserve the active member correctly; do not
 * modify @ref type directly.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct QueryOperator {
    /// @brief Active operator type selector.
    GeometryType type = GeometryType::Sphere;

    /// @brief Union storage for the concrete operator.
    union {
        BoxQueryOperator<T> box;
        CylinderQueryOperator<T> cylinder;
        PlaneQueryOperator<T> plane;
        SphereQueryOperator<T> sphere;
        TriangleQueryOperator<T> triangle;
        TriangleMeshQueryOperator<T> triangle_mesh;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the operator to a default type (commonly Sphere).
     * Validity still depends on pointer fields inside the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    QueryOperator() noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies @ref type and the matching union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    QueryOperator(const QueryOperator& other) noexcept;

    /**
     * @brief Copy assignment.
     *
     * @details
     * Assigns @ref type and copies the matching union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE QueryOperator&
    operator=(const QueryOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * @note
     * Defaulted; safe if union members are trivially destructible.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~QueryOperator() noexcept = default;

    /// @brief Construct from a box operator (host convenience).
    ATLAS_HOST
    QueryOperator(const BoxQueryOperator<T>& op);

    /// @brief Construct from a cylinder operator (host convenience).
    ATLAS_HOST
    QueryOperator(const CylinderQueryOperator<T>& op);

    /// @brief Construct from a plane operator (host convenience).
    ATLAS_HOST
    QueryOperator(const PlaneQueryOperator<T>& op);

    /// @brief Construct from a sphere operator (host convenience).
    ATLAS_HOST
    QueryOperator(const SphereQueryOperator<T>& op);

    /// @brief Construct from a triangle operator (host convenience).
    ATLAS_HOST
    QueryOperator(const TriangleQueryOperator<T>& op);

    /// @brief Construct from a triangle mesh operator (host convenience).
    ATLAS_HOST
    QueryOperator(const TriangleMeshQueryOperator<T>& op);

    /**
     * @brief Closest point query (runtime dispatch).
     *
     * @param p Query point.
     * @return Closest point result from the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Closest normal query (runtime dispatch).
     *
     * @param p Query point.
     * @return Closest normal from the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Signed distance query (runtime dispatch).
     *
     * @param p Query point.
     * @return Signed distance from the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Centroid query (runtime dispatch).
     *
     * @return Centroid/representative point from the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Bounds query (runtime dispatch).
     *
     * @return AABB bounds from the active operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Validity query (runtime dispatch).
     *
     * @return `true` if the active operator's parameters are usable.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;
};

} // namespace atlas::geometry

namespace atlas {
/**
 * @brief Convenience alias for `atlas::geometry::QueryOperator<T>`.
 */
template <typename T>
using QueryOperator = geometry::QueryOperator<T>;
} // namespace atlas

#include <atlas/geometry/query_operator.hpp>
