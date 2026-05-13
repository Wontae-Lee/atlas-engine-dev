#pragma once

/**
 * @file axis_aligned_bounding_box.h
 * @brief Declares axis-aligned bounding box types and utility functions.
 *
 * @details
 * This header defines the axis-aligned bounding box, abbreviated as AABB, used
 * by Atlas spatial queries.
 *
 * An AABB is represented by two corner points:
 *
 * - @ref atlas::spatial::AxisAlignedBoundingBox::lower_corner, the component-wise
 *   minimum corner,
 * - @ref atlas::spatial::AxisAlignedBoundingBox::upper_corner, the component-wise
 *   maximum corner.
 *
 * For a valid three-dimensional AABB, each coordinate interval satisfies:
 *
 * @f[
 *     \mathrm{lower}_i \le \mathrm{upper}_i,
 *     \qquad i \in \{0,1,2\}.
 * @f]
 *
 * The box describes the closed set:
 *
 * @f[
 *     B =
 *     \left\{
 *         \mathbf{x} \in \mathbb{R}^{3}
 *         \mid
 *         \mathrm{lower}_i \le x_i \le \mathrm{upper}_i
 *         \ \mathrm{for}\ i \in \{0,1,2\}
 *     \right\}.
 * @f]
 *
 * AABBs are commonly used for:
 *
 * - broad-phase collision and overlap tests,
 * - point containment tests,
 * - ray-box intersection tests,
 * - BVH and spatial hierarchy construction,
 * - conservative bounds for geometry,
 * - fast rejection before expensive narrow-phase operations.
 */

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <limits>

namespace atlas::spatial {

/**
 * @brief Stores the parametric intersection interval between a ray and an AABB.
 *
 * @details
 * A ray is interpreted as:
 *
 * @f[
 *     \mathbf{r}(t)
 *     =
 *     \mathbf{o}
 *     +
 *     t\mathbf{d},
 * @f]
 *
 * where @f$\mathbf{o}@f$ is the ray origin, @f$\mathbf{d}@f$ is the ray direction,
 * and @f$t@f$ is the ray parameter.
 *
 * When a ray intersects an AABB, the overlap between the ray and the box can be
 * represented as a parametric interval:
 *
 * @f[
 *     t \in [t_{\mathrm{enter}}, t_{\mathrm{exit}}].
 * @f]
 *
 * In this structure:
 *
 * - @ref enter stores @f$t_{\mathrm{enter}}@f$,
 * - @ref exit stores @f$t_{\mathrm{exit}}@f$,
 * - @ref is_intersecting indicates whether such an interval exists.
 *
 * @tparam T Floating-point scalar type used for ray parameters.
 */
template <typename T>
struct AxisAlignedBoundingBoxRayIntersection {
    /**
     * @brief Whether the ray intersects the AABB.
     *
     * @details
     * A value of `true` means that the ray overlaps the AABB for at least one
     * valid parametric interval. A value of `false` means that the ray misses the
     * box.
     */
    bool is_intersecting = false;

    /**
     * @brief Parametric entry distance along the ray.
     *
     * @details
     * If @ref is_intersecting is `true`, this is the first ray parameter at which
     * the ray enters the box:
     *
     * @f[
     *     \mathbf{r}(\mathrm{enter})
     *     =
     *     \mathbf{o}
     *     +
     *     \mathrm{enter}\,\mathbf{d}.
     * @f]
     *
     * If the ray starts inside the box, the implementation sets this value to
     * @f$0@f$.
     */
    T enter = T(0);

    /**
     * @brief Parametric exit distance along the ray.
     *
     * @details
     * If @ref is_intersecting is `true`, this is the last ray parameter at which
     * the ray remains inside the box:
     *
     * @f[
     *     \mathbf{r}(\mathrm{exit})
     *     =
     *     \mathbf{o}
     *     +
     *     \mathrm{exit}\,\mathbf{d}.
     * @f]
     */
    T exit = std::numeric_limits<T>::max();
};

/**
 * @brief Three-dimensional axis-aligned bounding box.
 *
 * @details
 * AxisAlignedBoundingBox represents a rectangular volume aligned with the
 * coordinate axes. The box is stored using a lower corner and an upper corner:
 *
 * @f[
 *     \mathbf{l}
 *     =
 *     \begin{bmatrix}
 *         l_x \\ l_y \\ l_z
 *     \end{bmatrix},
 *     \qquad
 *     \mathbf{u}
 *     =
 *     \begin{bmatrix}
 *         u_x \\ u_y \\ u_z
 *     \end{bmatrix}.
 * @f]
 *
 * The represented closed volume is:
 *
 * @f[
 *     l_x \le x \le u_x,
 *     \qquad
 *     l_y \le y \le u_y,
 *     \qquad
 *     l_z \le z \le u_z.
 * @f]
 *
 * The class supports:
 *
 * - side-length queries,
 * - surface-area computation,
 * - overlap tests against another AABB,
 * - point containment tests,
 * - ray intersection tests,
 * - ray intersection interval tracing,
 * - center, extent, and diagonal queries,
 * - validity and emptiness checks,
 * - merge and expansion operations,
 * - corner extraction,
 * - point clamping.
 *
 * @tparam T Floating-point scalar type used for coordinates, distances, and
 *           geometric measurements.
 *
 * @note The box uses closed intervals for containment and overlap tests.
 * @note The reset state is intentionally inverted so repeated merge() calls can
 *       grow the box from an empty initial state.
 */
template <typename T>
class AxisAlignedBoundingBox final {
public:
    /**
     * @brief Component-wise minimum corner of the box.
     *
     * @details
     * This corner stores:
     *
     * @f[
     *     \mathbf{l}
     *     =
     *     \begin{bmatrix}
     *         l_x \\ l_y \\ l_z
     *     \end{bmatrix}.
     * @f]
     *
     * For a valid non-inverted box, each component must satisfy:
     *
     * @f[
     *     l_i \le u_i.
     * @f]
     */
    Vector3<T> lower_corner;

    /**
     * @brief Component-wise maximum corner of the box.
     *
     * @details
     * This corner stores:
     *
     * @f[
     *     \mathbf{u}
     *     =
     *     \begin{bmatrix}
     *         u_x \\ u_y \\ u_z
     *     \end{bmatrix}.
     * @f]
     *
     * For a valid non-inverted box, each component must satisfy:
     *
     * @f[
     *     l_i \le u_i.
     * @f]
     */
    Vector3<T> upper_corner;

    /**
     * @brief Constructs an empty inverted AABB.
     *
     * @details
     * The default constructor calls reset(). The resulting state is:
     *
     * @f[
     *     \mathbf{l}
     *     =
     *     \begin{bmatrix}
     *         +\infty \\ +\infty \\ +\infty
     *     \end{bmatrix},
     *     \qquad
     *     \mathbf{u}
     *     =
     *     \begin{bmatrix}
     *         -\infty \\ -\infty \\ -\infty
     *     \end{bmatrix}.
     * @f]
     *
     * This sentinel state is useful for incremental construction by merge().
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox() noexcept;

    /**
     * @brief Constructs an AABB from two arbitrary corner points.
     *
     * @details
     * The input points do not need to be ordered. The constructor computes:
     *
     * @f[
     *     \mathbf{l}
     *     =
     *     \min(\mathbf{p}_1, \mathbf{p}_2),
     *     \qquad
     *     \mathbf{u}
     *     =
     *     \max(\mathbf{p}_1, \mathbf{p}_2),
     * @f]
     *
     * where the minimum and maximum are taken component-wise.
     *
     * @param point1 First input corner point.
     * @param point2 Second input corner point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @param other Source bounding box to copy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Returns the surface area of the box.
     *
     * @details
     * For side lengths:
     *
     * @f[
     *     w = u_x - l_x,
     *     \qquad
     *     h = u_y - l_y,
     *     \qquad
     *     d = u_z - l_z,
     * @f]
     *
     * the surface area is:
     *
     * @f[
     *     A
     *     =
     *     2(wh + wd + hd).
     * @f]
     *
     * @return Surface area of the AABB.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    area() const noexcept;

    /**
     * @brief Returns the side length along the x-axis.
     *
     * @details
     * Computes:
     *
     * @f[
     *     w = u_x - l_x.
     * @f]
     *
     * @return Width of the box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    width() const noexcept;

    /**
     * @brief Returns the side length along the y-axis.
     *
     * @details
     * Computes:
     *
     * @f[
     *     h = u_y - l_y.
     * @f]
     *
     * @return Height of the box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    height() const noexcept;

    /**
     * @brief Returns the side length along the z-axis.
     *
     * @details
     * Computes:
     *
     * @f[
     *     d = u_z - l_z.
     * @f]
     *
     * @return Depth of the box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    depth() const noexcept;

    /**
     * @brief Returns the side length along a selected axis.
     *
     * @details
     * Computes:
     *
     * @f[
     *     \mathrm{length}(i)
     *     =
     *     u_i - l_i.
     * @f]
     *
     * @param axis Axis index. The valid indices are `0`, `1`, and `2`.
     * @return Side length along the selected axis.
     *
     * @warning The function assumes @p axis is a valid vector component index.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length(std::size_t axis) const noexcept;

    /**
     * @brief Returns whether this box overlaps another AABB.
     *
     * @details
     * Two closed AABBs overlap if their intervals overlap on every axis:
     *
     * @f[
     *     [l_i, u_i] \cap [l'_i, u'_i] \ne \emptyset
     *     \qquad
     *     \mathrm{for\ all}\ i \in \{0,1,2\}.
     * @f]
     *
     * Equivalently, they do not overlap if there exists an axis on which one box
     * lies completely before the other:
     *
     * @f[
     *     u_i < l'_i
     *     \quad \mathrm{or} \quad
     *     l_i > u'_i.
     * @f]
     *
     * @param other Other bounding box.
     * @return `true` if the two boxes overlap or touch; otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AxisAlignedBoundingBox& other) const noexcept;

    /**
     * @brief Returns whether this box contains a point.
     *
     * @details
     * A point @f$\mathbf{p}@f$ is contained when:
     *
     * @f[
     *     l_i \le p_i \le u_i
     *     \qquad
     *     \mathrm{for\ all}\ i \in \{0,1,2\}.
     * @f]
     *
     * The boundary is included.
     *
     * @param point Query point.
     * @return `true` if the point lies inside or on the boundary of the box;
     *         otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Vector3<T>& point) const noexcept;

    /**
     * @brief Returns whether a ray intersects this box.
     *
     * @details
     * This function performs a slab intersection test. A ray is written as:
     *
     * @f[
     *     \mathbf{r}(t)
     *     =
     *     \mathbf{o}
     *     +
     *     t\mathbf{d},
     *     \qquad
     *     t \ge 0.
     * @f]
     *
     * For each axis, the ray is intersected against the interval
     * @f$[l_i, u_i]@f$. The resulting interval is accumulated as:
     *
     * @f[
     *     t_{\min}
     *     =
     *     \max(t_{\min}, t_{0,i}),
     *     \qquad
     *     t_{\max}
     *     =
     *     \min(t_{\max}, t_{1,i}).
     * @f]
     *
     * If at any point:
     *
     * @f[
     *     t_{\min} > t_{\max},
     * @f]
     *
     * the ray misses the box.
     *
     * @param ray Query ray.
     * @return `true` if the ray intersects the AABB for @f$t \ge 0@f`;
     *         otherwise `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray<T>& ray) const noexcept;

    /**
     * @brief Traces a ray against this box and returns the intersection interval.
     *
     * @details
     * This function uses the same slab method as intersects(), but returns the
     * full parametric overlap interval.
     *
     * For each axis with nonzero direction component:
     *
     * @f[
     *     t_0
     *     =
     *     \frac{l_i - o_i}{d_i},
     *     \qquad
     *     t_1
     *     =
     *     \frac{u_i - o_i}{d_i}.
     * @f]
     *
     * If @f$t_0 > t_1@f$, the two values are swapped. The global interval is:
     *
     * @f[
     *     t_{\mathrm{enter}}
     *     =
     *     \max_i t_{0,i},
     *     \qquad
     *     t_{\mathrm{exit}}
     *     =
     *     \min_i t_{1,i}.
     * @f]
     *
     * If the ray starts inside the box, the entry value is clamped to:
     *
     * @f[
     *     t_{\mathrm{enter}} = 0.
     * @f]
     *
     * @param ray Query ray.
     * @return Ray-box intersection interval result.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBoxRayIntersection<T>
    trace(const Ray<T>& ray) const noexcept;

    /**
     * @brief Returns the center point of the box.
     *
     * @details
     * Computes:
     *
     * @f[
     *     \mathbf{c}
     *     =
     *     \frac{\mathbf{l} + \mathbf{u}}{2}.
     * @f]
     *
     * @return Center point of the box.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    center() const noexcept;

    /**
     * @brief Returns the full side-length vector of the box.
     *
     * @details
     * Computes:
     *
     * @f[
     *     \mathbf{e}
     *     =
     *     \mathbf{u} - \mathbf{l}.
     * @f]
     *
     * This function returns full extents, not half extents.
     *
     * @return Vector containing width, height, and depth.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    extents() const noexcept;

    /**
     * @brief Returns the Euclidean length of the box diagonal.
     *
     * @details
     * The diagonal vector is:
     *
     * @f[
     *     \mathbf{d}
     *     =
     *     \mathbf{u} - \mathbf{l}.
     * @f]
     *
     * The returned value is:
     *
     * @f[
     *     \|\mathbf{d}\|
     *     =
     *     \sqrt{
     *         d_x^2 + d_y^2 + d_z^2
     *     }.
     * @f]
     *
     * @return Length of the box diagonal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length() const noexcept;

    /**
     * @brief Returns the squared Euclidean length of the box diagonal.
     *
     * @details
     * Computes:
     *
     * @f[
     *     \|\mathbf{d}\|^2
     *     =
     *     d_x^2 + d_y^2 + d_z^2,
     *     \qquad
     *     \mathbf{d} = \mathbf{u} - \mathbf{l}.
     * @f]
     *
     * This avoids the square root used by diagonal_length().
     *
     * @return Squared length of the box diagonal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length_squared() const noexcept;

    /**
     * @brief Returns whether the AABB is finite and ordered.
     *
     * @details
     * A box is valid when all corner coordinates are finite and:
     *
     * @f[
     *     l_i \le u_i
     *     \qquad
     *     \mathrm{for\ all}\ i \in \{0,1,2\}.
     * @f]
     *
     * @return `true` if the box is finite and has ordered bounds; otherwise
     *         `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Resets the box to an empty inverted sentinel state.
     *
     * @details
     * After reset, the bounds are:
     *
     * @f[
     *     \mathbf{l}
     *     =
     *     \begin{bmatrix}
     *         +\infty \\ +\infty \\ +\infty
     *     \end{bmatrix},
     *     \qquad
     *     \mathbf{u}
     *     =
     *     \begin{bmatrix}
     *         -\infty \\ -\infty \\ -\infty
     *     \end{bmatrix}.
     * @f]
     *
     * This state is useful for incremental box construction:
     *
     * @code
     * AxisAlignedBoundingBox<T> box;
     * box.merge(point0);
     * box.merge(point1);
     * box.merge(point2);
     * @endcode
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept;

    /**
     * @brief Expands this box so that it contains the given point.
     *
     * @details
     * The updated bounds are computed component-wise:
     *
     * @f[
     *     \mathbf{l}_{\mathrm{new}}
     *     =
     *     \min(\mathbf{l}_{\mathrm{old}}, \mathbf{p}),
     *     \qquad
     *     \mathbf{u}_{\mathrm{new}}
     *     =
     *     \max(\mathbf{u}_{\mathrm{old}}, \mathbf{p}).
     * @f]
     *
     * @param point Point to include in the box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Vector3<T>& point) noexcept;

    /**
     * @brief Expands this box so that it contains another AABB.
     *
     * @details
     * The updated bounds are:
     *
     * @f[
     *     \mathbf{l}_{\mathrm{new}}
     *     =
     *     \min(\mathbf{l}_{\mathrm{old}}, \mathbf{l}_{\mathrm{other}}),
     *     \qquad
     *     \mathbf{u}_{\mathrm{new}}
     *     =
     *     \max(\mathbf{u}_{\mathrm{old}}, \mathbf{u}_{\mathrm{other}}).
     * @f]
     *
     * @param other Box to include in this box.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AxisAlignedBoundingBox& other) noexcept;

    /**
     * @brief Expands the box outward by a uniform scalar amount.
     *
     * @details
     * The lower corner is decreased and the upper corner is increased by
     * @p delta on every axis:
     *
     * @f[
     *     \mathbf{l}_{\mathrm{new}}
     *     =
     *     \mathbf{l}_{\mathrm{old}}
     *     -
     *     \begin{bmatrix}
     *         \delta \\ \delta \\ \delta
     *     \end{bmatrix},
     *     \qquad
     *     \mathbf{u}_{\mathrm{new}}
     *     =
     *     \mathbf{u}_{\mathrm{old}}
     *     +
     *     \begin{bmatrix}
     *         \delta \\ \delta \\ \delta
     *     \end{bmatrix}.
     * @f]
     *
     * A positive @p delta grows the box. A negative @p delta shrinks the box and
     * may invert it if the shrink amount is too large.
     *
     * @param delta Uniform expansion amount.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(T delta) noexcept;

    /**
     * @brief Returns one of the eight corners of the box.
     *
     * @details
     * The corner index uses the lowest three bits of @p idx:
     *
     * - bit 0 selects x: `0` for lower x, `1` for upper x,
     * - bit 1 selects y: `0` for lower y, `1` for upper y,
     * - bit 2 selects z: `0` for lower z, `1` for upper z.
     *
     * Therefore:
     *
     * @f[
     *     x =
     *     \begin{cases}
     *         l_x, & \mathrm{if\ bit\ }0 = 0, \\
     *         u_x, & \mathrm{if\ bit\ }0 = 1,
     *     \end{cases}
     * @f]
     *
     * and similarly for @f$y@f$ and @f$z@f$.
     *
     * @param idx Corner index. Values in `[0,7]` select the eight unique corners.
     *            Higher bits are ignored.
     * @return Selected corner point.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    corner(std::size_t idx) const noexcept;

    /**
     * @brief Clamps a point to the closed box interval.
     *
     * @details
     * Each component is clamped independently:
     *
     * @f[
     *     p'_{i}
     *     =
     *     \min(
     *         \max(p_i, l_i),
     *         u_i
     *     ).
     * @f]
     *
     * The returned point is the closest point inside the AABB under component-wise
     * interval projection.
     *
     * @param point Query point.
     * @return Point clamped to the AABB.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    clamp(const Vector3<T>& point) const noexcept;

    /**
     * @brief Returns whether the box has non-positive length on any axis.
     *
     * @details
     * The implementation considers the box empty when:
     *
     * @f[
     *     u_i \le l_i
     *     \qquad
     *     \mathrm{for\ at\ least\ one\ axis}\ i.
     * @f]
     *
     * This means a zero-thickness box is considered empty by this function.
     *
     * @return `true` if at least one side length is zero or negative; otherwise
     *         `false`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept;
};

/**
 * @brief Constructs a degenerate AABB from a single point.
 *
 * @details
 * The resulting box has:
 *
 * @f[
 *     \mathbf{l}
 *     =
 *     \mathbf{p},
 *     \qquad
 *     \mathbf{u}
 *     =
 *     \mathbf{p}.
 * @f]
 *
 * Therefore, all side lengths are zero.
 *
 * @param p Point used as both lower and upper corner.
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
 * @brief Returns the smallest AABB containing two input AABBs.
 *
 * @details
 * The merged bounds are:
 *
 * @f[
 *     \mathbf{l}_{\mathrm{out}}
 *     =
 *     \min(\mathbf{l}_a, \mathbf{l}_b),
 *     \qquad
 *     \mathbf{u}_{\mathrm{out}}
 *     =
 *     \max(\mathbf{u}_a, \mathbf{u}_b).
 * @f]
 *
 * Component-wise minimum and maximum are used.
 *
 * @param a First bounding box.
 * @param b Second bounding box.
 * @return AABB containing both @p a and @p b.
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
 * @brief Convenience alias for atlas::spatial::AxisAlignedBoundingBox.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using AxisAlignedBoundingBox = atlas::spatial::AxisAlignedBoundingBox<T>;

/**
 * @brief Convenience alias for atlas::spatial::AxisAlignedBoundingBoxRayIntersection.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using HitAABB = atlas::spatial::AxisAlignedBoundingBoxRayIntersection<T>;

/**
 * @brief Short convenience alias for atlas::spatial::AxisAlignedBoundingBox.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using AABB = atlas::spatial::AxisAlignedBoundingBox<T>;

/**
 * @brief Float-specialized axis-aligned bounding box.
 */
using AABBF = spatial::AxisAlignedBoundingBox<float>;

/**
 * @brief Double-specialized axis-aligned bounding box.
 */
using AABBD = spatial::AxisAlignedBoundingBox<double>;

/**
 * @brief Float-specialized AABB ray-intersection result.
 */
using AABBRayInteractionF = spatial::AxisAlignedBoundingBoxRayIntersection<float>;

/**
 * @brief Double-specialized AABB ray-intersection result.
 */
using AABBRayInteractionD = spatial::AxisAlignedBoundingBoxRayIntersection<double>;

} // namespace atlas

#include <atlas/spatial/axis_aligned_bounding_box.hpp>