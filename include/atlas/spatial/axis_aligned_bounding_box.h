#pragma once

/**
 * @file axis_aligned_bounding_box.h
 * @brief Declares an axis-aligned bounding box (AABB), ray-intersection result type, and helper construction/merge utilities.
 *
 * @details
 * This header defines:
 * - @ref atlas::spatial::AxisAlignedBoundingBoxRayIntersection, a lightweight
 *   result object describing the overlap interval between a ray and an AABB,
 * - @ref atlas::spatial::AxisAlignedBoundingBox, a general-purpose axis-aligned
 *   bounding box used for broad-phase spatial queries and hierarchy construction,
 * - helper free functions such as @ref make_aabb and @ref merge_aabb.
 *
 * ## Purpose
 * Axis-aligned bounding boxes are fundamental spatial primitives used throughout
 * Atlas for:
 * - broad-phase overlap tests,
 * - point containment,
 * - ray-box intersection,
 * - hierarchy construction such as BVHs,
 * - bounding analytic or mesh geometry,
 * - conservative spatial pruning.
 *
 * ## Geometric convention
 * An AABB is represented by two corners:
 * - @ref lower_corner : component-wise minimum corner,
 * - @ref upper_corner : component-wise maximum corner.
 *
 * A valid box satisfies:
 * \f[
 * \texttt{lower\_corner}[i] \le \texttt{upper\_corner}[i]
 * \quad \text{for each axis } i \in \{x,y,z\}.
 * \f]
 *
 * ## Ray tracing
 * Ray-box intersection is reported through
 * @ref AxisAlignedBoundingBoxRayIntersection, which provides:
 * - a boolean intersection flag,
 * - parametric entry distance,
 * - parametric exit distance.
 *
 * ## Host/device usage
 * Most operations are marked `ATLAS_ALL_DEVICE`, allowing the AABB type to be
 * used consistently in:
 * - host-side geometry code,
 * - device kernels,
 * - backend-portable query utilities.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates, distances, and extents.
 */
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <limits>

namespace atlas::spatial {

/**
 * @brief Stores the result of tracing a ray against an axis-aligned bounding box.
 *
 * @details
 * This structure describes the parametric overlap interval between a ray and an
 * AABB, typically using the slab-intersection method.
 *
 * If the ray intersects the box, then:
 * - @ref is_intersecting is `true`,
 * - @ref enter is the parametric entry distance,
 * - @ref exit is the parametric exit distance.
 *
 * If no intersection occurs, @ref is_intersecting remains `false`, while the
 * numeric contents of @ref enter and @ref exit follow the initialization or the
 * implementation-specific trace result.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for ray parameters.
 */
template <typename T>
struct AxisAlignedBoundingBoxRayIntersection {
    /**
     * @brief Whether the ray intersects the box interval.
     *
     * @details
     * `true` indicates that the ray overlaps the AABB for some parametric interval.
     */
    bool is_intersecting = false;

    /**
     * @brief Parametric entry distance along the ray.
     *
     * @details
     * Represents the first parametric value at which the ray enters the box.
     */
    T enter = T(0);

    /**
     * @brief Parametric exit distance along the ray.
     *
     * @details
     * Represents the last parametric value at which the ray leaves the box.
     */
    T exit = std::numeric_limits<T>::max();
};

/**
 * @brief Axis-aligned bounding box used for broad-phase spatial tests and hierarchy construction.
 *
 * @details
 * @ref AxisAlignedBoundingBox represents a 3D rectangular volume aligned with
 * the coordinate axes.
 *
 * It is defined by:
 * - @ref lower_corner : component-wise minimum corner,
 * - @ref upper_corner : component-wise maximum corner.
 *
 * ## Provided functionality
 * The class supports:
 * - geometric size queries,
 * - overlap and containment tests,
 * - ray intersection and interval tracing,
 * - centroid/extents/diagonal computations,
 * - validity checks,
 * - reset/merge/expand mutation helpers,
 * - point clamping and corner extraction.
 *
 * ## Typical use cases
 * AABBs are commonly used for:
 * - broad-phase collision detection,
 * - BVH node bounds,
 * - mesh primitive bounds,
 * - conservative bounding of analytic geometry,
 * - fast point and ray rejection tests.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and distances.
 */
template <typename T>
class AxisAlignedBoundingBox final {
public:
    /**
     * @brief Minimum corner of the box.
     *
     * @details
     * Stores the component-wise lower bound of the AABB.
     */
    Vector3<T> lower_corner;

    /**
     * @brief Maximum corner of the box.
     *
     * @details
     * Stores the component-wise upper bound of the AABB.
     */
    Vector3<T> upper_corner;

    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the box according to the implementation in
     * `axis_aligned_bounding_box.hpp`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox() noexcept;

    /**
     * @brief Construct an AABB from two points.
     *
     * @details
     * The implementation typically interprets the two input points as opposite
     * corners and computes component-wise minima and maxima as needed.
     *
     * @param point1 First input point.
     * @param point2 Second input point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @param other Source bounding box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Return the surface area of the box.
     *
     * @details
     * For side lengths \f$w,h,d\f$, the area is typically:
     * \f[
     * 2(wh + wd + hd).
     * \f]
     *
     * @return Surface area of the AABB.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    area() const noexcept;

    /**
     * @brief Return the box width along the x-axis.
     *
     * @return `upper_corner.x - lower_corner.x`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    width() const noexcept;

    /**
     * @brief Return the box height along the y-axis.
     *
     * @return `upper_corner.y - lower_corner.y`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    height() const noexcept;

    /**
     * @brief Return the box depth along the z-axis.
     *
     * @return `upper_corner.z - lower_corner.z`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    depth() const noexcept;

    /**
     * @brief Return the box length along a selected axis.
     *
     * @param axis Axis index in `{0,1,2}`.
     * @return Side length along the selected axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length(std::size_t axis) const noexcept;

    /**
     * @brief Return whether this box overlaps another AABB.
     *
     * @param other Other bounding box.
     * @return `true` if the boxes overlap; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AxisAlignedBoundingBox& other) const noexcept;

    /**
     * @brief Return whether this box contains a point.
     *
     * @details
     * A point is typically considered contained if each coordinate lies within
     * the closed interval defined by the lower and upper corners.
     *
     * @param point Query point.
     * @return `true` if the point lies inside or on the boundary; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Vector3<T>& point) const noexcept;

    /**
     * @brief Return whether a ray intersects this box.
     *
     * @param ray Query ray.
     * @return `true` if the ray intersects the AABB; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray<T>& ray) const noexcept;

    /**
     * @brief Trace a ray against this box and return the overlap interval.
     *
     * @param ray Query ray.
     * @return Ray-box intersection interval result.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBoxRayIntersection<T>
    trace(const Ray<T>& ray) const noexcept;

    /**
     * @brief Return the center point of the box.
     *
     * @details
     * Typically:
     * \f[
     * \frac{\texttt{lower\_corner} + \texttt{upper\_corner}}{2}.
     * \f]
     *
     * @return Box center.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    center() const noexcept;

    /**
     * @brief Return the extents of the box.
     *
     * @details
     * The exact convention is implementation-defined, but this commonly means:
     * - full side lengths, or
     * - half-lengths from the center.
     *
     * @return Box extents vector.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    extents() const noexcept;

    /**
     * @brief Return the length of the box diagonal.
     *
     * @return Euclidean length of `upper_corner - lower_corner`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length() const noexcept;

    /**
     * @brief Return the squared length of the box diagonal.
     *
     * @return Squared Euclidean length of `upper_corner - lower_corner`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length_squared() const noexcept;

    /**
     * @brief Return whether the AABB is valid.
     *
     * @details
     * A valid AABB typically requires:
     * - finite corner coordinates,
     * - `lower_corner[i] <= upper_corner[i]` for all axes.
     *
     * @return `true` if the box is well-formed; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Reset the box to an empty or invalid state.
     *
     * @details
     * The exact reset convention is implementation-defined, but it is typically
     * chosen so that subsequent @ref merge operations can grow the box from scratch.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Expand this box to include a point.
     *
     * @param point Point to merge into the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Vector3<T>& point) noexcept;

    /**
     * @brief Expand this box to include another AABB.
     *
     * @param other Other bounding box to merge into this one.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Expand the box outward by a uniform scalar amount.
     *
     * @details
     * Typically decreases the lower corner and increases the upper corner by
     * `delta` along each axis.
     *
     * @param delta Expansion amount.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(T delta) noexcept;

    /**
     * @brief Return a selected corner of the box.
     *
     * @details
     * The 3 bits of @p idx typically select lower/upper on each axis.
     *
     * @param idx Corner index, usually in `[0,7]`.
     * @return Selected corner point.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    corner(std::size_t idx) const noexcept;

    /**
     * @brief Clamp a point to the box.
     *
     * @details
     * Each coordinate of the result is clamped independently to the closed
     * interval defined by the corresponding lower and upper bounds.
     *
     * @param point Query point.
     * @return Clamped point.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    clamp(const Vector3<T>& point) const noexcept;

    /**
     * @brief Return whether the box is empty.
     *
     * @details
     * The exact emptiness convention is implementation-defined, but it commonly
     * reflects zero volume, invalid bounds, or a reset sentinel state.
     *
     * @return `true` if the box is considered empty; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept;
};

/**
 * @brief Construct a degenerate AABB from a single point.
 *
 * @details
 * The resulting box has identical lower and upper corners equal to @p p.
 *
 * @param p Point used as both box corners.
 * @return Degenerate AABB located at @p p.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
make_aabb(const Vector3<T>& p) noexcept {
    AxisAlignedBoundingBox<T> b;
    b.lower_corner = p;
    b.upper_corner = p;
    return b;
}

/**
 * @brief Return the merged union of two AABBs.
 *
 * @details
 * The resulting box is the smallest axis-aligned box containing both @p a and @p b.
 *
 * @param a First bounding box.
 * @param b Second bounding box.
 * @return Merged bounding box.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
merge_aabb(const AxisAlignedBoundingBox<T>& a,
           const AxisAlignedBoundingBox<T>& b) noexcept {
    AxisAlignedBoundingBox<T> out;
    out.lower_corner = math::cmin(a.lower_corner, b.lower_corner);
    out.upper_corner = math::cmax(a.upper_corner, b.upper_corner);
    return out;
}

} // namespace atlas::spatial

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::spatial::AxisAlignedBoundingBox.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using AxisAlignedBoundingBox = atlas::spatial::AxisAlignedBoundingBox<T>;

/**
 * @brief Convenience alias for @ref atlas::spatial::AxisAlignedBoundingBoxRayIntersection.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using HitAABB = atlas::spatial::AxisAlignedBoundingBoxRayIntersection<T>;

/**
 * @brief Convenience alias for @ref atlas::spatial::AxisAlignedBoundingBox.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using AABB = atlas::spatial::AxisAlignedBoundingBox<T>;

/**
 * @brief Common specialization of @ref atlas::spatial::AxisAlignedBoundingBox for `float`.
 */
using AABBF = spatial::AxisAlignedBoundingBox<float>;

/**
 * @brief Common specialization of @ref atlas::spatial::AxisAlignedBoundingBox for `double`.
 */
using AABBD = spatial::AxisAlignedBoundingBox<double>;

/**
 * @brief Common specialization of @ref atlas::spatial::AxisAlignedBoundingBoxRayIntersection for `float`.
 */
using AABBRayInteractionF = spatial::AxisAlignedBoundingBoxRayIntersection<float>;

/**
 * @brief Common specialization of @ref atlas::spatial::AxisAlignedBoundingBoxRayIntersection for `double`.
 */
using AABBRayInteractionD = spatial::AxisAlignedBoundingBoxRayIntersection<double>;

} // namespace atlas

#include <atlas/spatial/axis_aligned_bounding_box.hpp>