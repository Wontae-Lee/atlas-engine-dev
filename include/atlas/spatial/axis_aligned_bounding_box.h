#pragma once
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <limits>

namespace atlas::spatial {

/**
 * @brief Result of ray vs. axis-aligned bounding box (AABB) intersection.
 *
 * @details
 * This structure represents the parametric interval \f$[t_{enter}, t_{exit}]\f$
 * along a ray where it overlaps the AABB:
 * - `enter` is the first parameter value where the ray enters the box.
 * - `exit`  is the last parameter value where the ray remains inside the box.
 *
 * When `is_intersecting == false`, the `enter/exit` values should be considered
 * unspecified (though defaults are provided for convenience).
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct AxisAlignedBoundingBoxRayIntersection {
    /// True if the ray intersects (or touches) the AABB.
    bool is_intersecting = false;

    /// Parametric entry distance along the ray.
    T enter = T(0);

    /// Parametric exit distance along the ray.
    T exit = std::numeric_limits<T>::max();
};

/**
 * @brief Axis-aligned bounding box (AABB) in 3D.
 *
 * @details
 * An AABB is defined by two corners:
 * - `lower_corner`: component-wise minimum (x/y/z)
 * - `upper_corner`: component-wise maximum (x/y/z)
 *
 * Common uses:
 * - Broad-phase collision and overlap tests
 * - Spatial acceleration structures (BVH, grids, hash grids)
 * - Conservative bounds for geometry and particles
 *
 * Conventions and invariants:
 * - A "valid" non-empty AABB typically satisfies
 *   `lower_corner.x <= upper_corner.x` (and similarly for y, z).
 * - This class also supports an "empty" state produced by `reset()`,
 *   where lower > upper (implementation-defined) so merges expand it.
 *
 * Performance notes:
 * - Most methods are `ATLAS_ALL_DEVICE` and `ATLAS_FORCE_INLINE` to be usable
 *   in both host and device code paths.
 *
 * @tparam T Floating-point scalar type (e.g., float, double).
 */
template <typename T>
class AxisAlignedBoundingBox final {
public:
    /// Component-wise minimum corner.
    Vector3<T> lower_corner;

    /// Component-wise maximum corner.
    Vector3<T> upper_corner;

    /**
     * @brief Construct an AABB in an "empty" or default-initialized state.
     *
     * @details
     * The exact default contents depend on the out-of-line implementation.
     * Typically, it is either:
     * - an empty AABB (so merging works naturally), or
     * - a zero-sized AABB at the origin.
     *
     * Prefer `reset()` if you need a guaranteed empty box before merging.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox() noexcept;

    /**
     * @brief Construct an AABB that encloses two points.
     *
     * @param point1 First point.
     * @param point2 Second point.
     *
     * @details
     * The resulting box encloses both points, i.e.:
     * - `lower_corner = cmin(point1, point2)`
     * - `upper_corner = cmax(point1, point2)`
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept;

    /**
     * @brief Copy-construct an AABB.
     *
     * @param other Source AABB.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Surface area of the AABB.
     *
     * @return \f$ 2(w h + w d + h d) \f$ for non-empty boxes.
     *
     * @note For empty boxes, the returned value is implementation-defined.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    area() const noexcept;

    /**
     * @brief Box extent along x (width).
     *
     * @return `upper_corner.x - lower_corner.x` for non-empty boxes.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    width() const noexcept;

    /**
     * @brief Box extent along y (height).
     *
     * @return `upper_corner.y - lower_corner.y` for non-empty boxes.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    height() const noexcept;

    /**
     * @brief Box extent along z (depth).
     *
     * @return `upper_corner.z - lower_corner.z` for non-empty boxes.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    depth() const noexcept;

    /**
     * @brief Length of the AABB along an axis.
     *
     * @param axis Axis index: 0 -> x, 1 -> y, 2 -> z.
     * @return Component extent on the requested axis.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length(std::size_t axis) const noexcept;

    /**
     * @brief Test AABB vs. AABB overlap.
     *
     * @param other Another AABB.
     * @return True if boxes overlap (including touching faces/edges/vertices).
     *
     * @note If either box is empty, the result is typically false.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AxisAlignedBoundingBox& other) const noexcept;

    /**
     * @brief Test if a point is contained in the AABB.
     *
     * @param point Query point.
     * @return True if `point` lies inside or on the boundary.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Vector3<T>& point) const noexcept;

    /**
     * @brief Fast boolean ray intersection test.
     *
     * @param ray Ray to test.
     * @return True if the ray intersects (or touches) the AABB.
     *
     * @details
     * This is typically implemented using the "slab" method:
     * compute intersection intervals on x/y/z slabs and intersect them.
     * Use `trace()` if you also need `enter/exit` distances.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray<T>& ray) const noexcept;

    /**
     * @brief Ray intersection with parametric entry/exit values.
     *
     * @param ray Ray to test.
     * @return A structure containing `is_intersecting`, `enter`, and `exit`.
     *
     * @details
     * For a typical slab implementation:
     * - `enter` is the maximum of per-axis entry parameters
     * - `exit`  is the minimum of per-axis exit parameters
     * - intersection occurs when `enter <= exit` and the interval overlaps
     *   the ray's valid range (often `exit >= 0`)
     *
     * This result is useful for:
     * - BVH traversal (ordering nodes by nearest hit)
     * - computing segment overlap with the box
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBoxRayIntersection<T>
    trace(const Ray<T>& ray) const noexcept;

    /**
     * @brief Center point of the box.
     *
     * @return `(lower_corner + upper_corner) * 0.5`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    center() const noexcept;

    /**
     * @brief Full extents of the box.
     *
     * @return `upper_corner - lower_corner`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    extents() const noexcept;

    /**
     * @brief Length of the diagonal (Euclidean).
     *
     * @return `|upper_corner - lower_corner|`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length() const noexcept;

    /**
     * @brief Squared diagonal length.
     *
     * @return `dot(upper_corner - lower_corner, upper_corner - lower_corner)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length_squared() const noexcept;

    /**
     * @brief Returns whether the AABB stores finite, ordered corners.
     *
     * @details
     * A valid box requires:
     * - all corner components are finite
     * - `lower_corner <= upper_corner` component-wise
     *
     * This excludes the inverted "empty" state produced by @ref reset().
     *
     * @return `true` if the bounds can be safely used for spatial queries or sampling.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Reset the AABB to an empty state.
     *
     * @details
     * After `reset()`, merging a point or another AABB should yield a correct box.
     * A common strategy is:
     * - `lower_corner = +inf`, `upper_corner = -inf`
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Expand the AABB to include a point.
     *
     * @param point Point to include.
     *
     * @details
     * Performs component-wise min/max updates on the corners.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Vector3<T>& point) noexcept;

    /**
     * @brief Expand the AABB to include another AABB.
     *
     * @param other Box to merge.
     *
     * @details
     * Equivalent to merging both corners of `other` (component-wise).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Expand the box uniformly by a delta in all directions.
     *
     * @param delta Amount to expand.
     *
     * @details
     * Typical behavior:
     * - `lower_corner -= Vector3(delta)`
     * - `upper_corner += Vector3(delta)`
     *
     * Useful for adding padding / thickness / numerical robustness.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(T delta) noexcept;

    /**
     * @brief Return one of the 8 corners by index.
     *
     * @param idx Corner index in [0, 7].
     * @return The requested corner position.
     *
     * @details
     * Common convention:
     * - bit 0 selects x (0=lower, 1=upper)
     * - bit 1 selects y
     * - bit 2 selects z
     *
     * i.e. corner(0) = (lx, ly, lz), corner(7) = (ux, uy, uz).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    corner(std::size_t idx) const noexcept;

    /**
     * @brief Clamp a point to the AABB.
     *
     * @param point Input point.
     * @return Closest point inside/on the box (component-wise clamp).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    clamp(const Vector3<T>& point) const noexcept;

    /**
     * @brief Check if the AABB is empty.
     *
     * @return True if the box represents an empty interval.
     *
     * @details
     * This implementation treats any non-positive extent as empty, i.e.
     * it reports true when at least one axis satisfies
     * `upper_corner[i] <= lower_corner[i]`.
     * That convention classifies both inverted boxes and zero-thickness boxes
     * as empty for volume-based acceleration structure logic.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept;
};

/**
 * @brief Create a degenerate AABB located at a single point.
 *
 * @param p The point.
 * @return AABB with `lower_corner == upper_corner == p`.
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
 * @brief Merge two AABBs and return the enclosing AABB.
 *
 * @param a First AABB.
 * @param b Second AABB.
 * @return Enclosing AABB whose corners are component-wise min/max.
 *
 * @note This is a pure function variant of `AxisAlignedBoundingBox::merge`.
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

template <typename T>
using AxisAlignedBoundingBox = atlas::spatial::AxisAlignedBoundingBox<T>;

template <typename T>
using HitAABB = atlas::spatial::AxisAlignedBoundingBoxRayIntersection<T>;

template <typename T>
using AABB = atlas::spatial::AxisAlignedBoundingBox<T>;

using AABBF = spatial::AxisAlignedBoundingBox<float>;
using AABBD = spatial::AxisAlignedBoundingBox<double>;

using AABBRayInteractionF = spatial::AxisAlignedBoundingBoxRayIntersection<float>;
using AABBRayInteractionD = spatial::AxisAlignedBoundingBoxRayIntersection<double>;

} // namespace atlas

#include <atlas/spatial/axis_aligned_bounding_box.hpp>
