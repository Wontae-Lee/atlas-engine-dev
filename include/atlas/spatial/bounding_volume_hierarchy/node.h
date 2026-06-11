#pragma once

/**
 * @file node.h
 * @brief Declares the node representation used by bounding volume hierarchies.
 *
 * @details
 * This file defines @ref atlas::spatial::BVHNode, the compact node record used
 * by BVH implementations in Atlas.
 *
 * A BVH is a tree over geometric primitives. Each node stores an axis-aligned
 * bounding box that encloses either:
 *
 * - the bounding boxes of its child nodes, for an internal node,
 * - the primitive range referenced by the leaf, for a leaf node.
 *
 * A node is interpreted according to @ref atlas::spatial::BVHNode::is_leaf:
 *
 * - if `is_leaf == true`, the node is a leaf and uses `start` and `count`,
 * - if `is_leaf == false`, the node is an internal node and uses `left` and `right`.
 *
 * For a leaf node, the primitive range is a half-open interval:
 *
 * @f[
 *     [\mathrm{start},\ \mathrm{start} + \mathrm{count})
 * @f]
 *
 * into an external primitive-index buffer.
 *
 * For an internal node, the children are referenced by integer node indices:
 *
 * @f[
 *     \mathrm{left} \ge 0,
 *     \qquad
 *     \mathrm{right} \ge 0.
 * @f]
 *
 * The structure intentionally stores indices rather than pointers so that node
 * arrays can be copied between host and device memory and traversed using
 * contiguous buffer storage.
 */

#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::spatial {

/**
 * @brief Node record used in a bounding volume hierarchy.
 *
 * @details
 * BVHNode represents one node in a binary bounding volume hierarchy. The same
 * structure is used for both internal nodes and leaf nodes.
 *
 * ## Internal node layout
 *
 * For an internal node:
 *
 * - @ref is_leaf is `false`,
 * - @ref left is the index of the left child node,
 * - @ref right is the index of the right child node,
 * - @ref start is unused and normally set to `-1`,
 * - @ref count is normally set to `0`,
 * - @ref bounds encloses the bounds of both children.
 *
 * The internal-node bound is usually:
 *
 * @f[
 *     B_{\mathrm{node}}
 *     =
 *     B_{\mathrm{left}}
 *     \cup
 *     B_{\mathrm{right}}.
 * @f]
 *
 * ## Leaf node layout
 *
 * For a leaf node:
 *
 * - @ref is_leaf is `true`,
 * - @ref left and @ref right are normally set to `-1`,
 * - @ref start is the first entry in the primitive-index buffer,
 * - @ref count is the number of primitives referenced by the leaf,
 * - @ref bounds encloses all referenced primitives.
 *
 * The primitive interval is:
 *
 * @f[
 *     i \in
 *     [\mathrm{start},\ \mathrm{start} + \mathrm{count}).
 * @f]
 *
 * If the external primitive-index buffer is called `indices`, then the primitive
 * IDs referenced by the leaf are:
 *
 * @code
 * indices[start],
 * indices[start + 1],
 * ...
 * indices[start + count - 1]
 * @endcode
 *
 * @tparam T Floating-point scalar type used by the node bounding box.
 *
 * @note The node does not own primitive data. Leaf nodes only reference a range
 *       in an external primitive-index buffer.
 * @note Child references are stored as integer indices into an external node
 *       buffer, not as pointers.
 */
template <typename T>
struct BVHNode {

    /**
     * @brief Axis-aligned bounding box associated with this node.
     *
     * @details
     * For an internal node, this box encloses both child nodes:
     *
     * @f[
     *     B_{\mathrm{node}}
     *     =
     *     B_{\mathrm{left}}
     *     \cup
     *     B_{\mathrm{right}}.
     * @f]
     *
     * For a leaf node, this box encloses all primitives referenced by the
     * half-open primitive range:
     *
     * @f[
     *     [\mathrm{start},\ \mathrm{start} + \mathrm{count}).
     * @f]
     */
    AABB<T> bounds;

    /**
     * @brief Area-weighted centroid sum used by fast winding approximation.
     *
     * @details
     * This stores \f$\sum_i A_i c_i\f$ over the triangles covered by the node,
     * where @f$A_i@f$ is triangle area and @f$c_i@f$ is triangle centroid. The
     * aggregate center is `solid_angle_moment / solid_angle_area`.
     */
    atlas::math::Vector<T, 3> solid_angle_moment { T(0), T(0), T(0) };

    /**
     * @brief Sum of oriented triangle area vectors used by fast winding.
     *
     * @details
     * Each triangle contributes @f$\frac{1}{2} ((b-a) \times (c-a))@f$.
     */
    atlas::math::Vector<T, 3> solid_angle_normal_area { T(0), T(0), T(0) };

    /**
     * @brief Sum of unsigned triangle areas covered by this node.
     */
    T solid_angle_area = T(0);

    /**
     * @brief Left child node index for internal nodes.
     *
     * @details
     * This value is valid only when @ref is_leaf is `false`.
     *
     * For a valid internal node:
     *
     * @f[
     *     \mathrm{left} \ge 0.
     * @f]
     *
     * For a leaf node, this value is normally `-1`.
     */
    int left = -1;

    /**
     * @brief Right child node index for internal nodes.
     *
     * @details
     * This value is valid only when @ref is_leaf is `false`.
     *
     * For a valid internal node:
     *
     * @f[
     *     \mathrm{right} \ge 0.
     * @f]
     *
     * For a leaf node, this value is normally `-1`.
     */
    int right = -1;

    /**
     * @brief Start offset into the primitive-index buffer for leaf nodes.
     *
     * @details
     * This value is valid only when @ref is_leaf is `true`.
     *
     * The referenced primitive interval is:
     *
     * @f[
     *     [\mathrm{start},\ \mathrm{start} + \mathrm{count}).
     * @f]
     *
     * For a valid leaf node:
     *
     * @f[
     *     \mathrm{start} \ge 0.
     * @f]
     *
     * For an internal node, this value is normally `-1`.
     */
    int start = -1;

    /**
     * @brief Number of primitives referenced by a leaf node.
     *
     * @details
     * This value is valid only when @ref is_leaf is `true`.
     *
     * For a valid leaf node:
     *
     * @f[
     *     \mathrm{count} > 0.
     * @f]
     *
     * For an internal node, this value is normally `0`.
     */
    int count = 0;

    /**
     * @brief Indicates whether this node is a leaf node.
     *
     * @details
     * This flag determines how the remaining index fields are interpreted:
     *
     * - `true`: use @ref start and @ref count as a primitive range,
     * - `false`: use @ref left and @ref right as child node indices.
     */
    bool is_leaf = false;
};

} // namespace atlas::spatial
