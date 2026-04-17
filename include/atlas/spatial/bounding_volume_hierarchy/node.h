#pragma once

/**
 * @file node.h
 * @brief Declares the node representation used in bounding volume hierarchies (BVH).
 *
 * @details
 * This file defines @ref atlas::spatial::BVHNode, a compact structure used to
 * represent nodes in a bounding volume hierarchy.
 *
 * ## Node types
 * A node may be either:
 * - **internal node**
 *   - contains child indices (`left`, `right`)
 *   - encloses a spatial region via @ref bounds
 * - **leaf node**
 *   - contains a contiguous primitive range (`start`, `count`)
 *   - has no valid children
 *
 * ## Usage
 * BVHNode is shared across multiple BVH implementations (e.g., LBVH) and is
 * designed to be:
 * - trivially copyable,
 * - contiguous in memory,
 * - suitable for both host and device usage.
 *
 * ## Invariants
 * - If @ref is_leaf is `true`:
 *   - `start >= 0`
 *   - `count > 0`
 *   - `left == -1`, `right == -1`
 * - If @ref is_leaf is `false`:
 *   - `left >= 0`, `right >= 0`
 *   - `count == 0`
 *
 * ---
 *
 * @tparam T Floating-point scalar used for bounding volumes.
 */

#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::spatial {

/**
 * @brief Single node in a bounding volume hierarchy.
 *
 * @details
 * Represents either an internal node with two children or a leaf node
 * containing a contiguous range of primitives.
 *
 * The node stores:
 * - an axis-aligned bounding box covering its subtree or primitives,
 * - child indices for traversal (internal nodes),
 * - primitive range information (leaf nodes).
 *
 * ---
 *
 * @tparam T Floating-point scalar used for bounding boxes.
 */
template <typename T>
struct BVHNode {

    /**
     * @brief Bounding volume enclosing this node.
     *
     * @details
     * For internal nodes, this encloses both children.
     * For leaf nodes, this encloses all referenced primitives.
     */
    AABB<T> bounds;

    /**
     * @brief Left child node index.
     *
     * @details
     * Valid only when @ref is_leaf is `false`.
     * Otherwise set to `-1`.
     */
    int left = -1;

    /**
     * @brief Right child node index.
     *
     * @details
     * Valid only when @ref is_leaf is `false`.
     * Otherwise set to `-1`.
     */
    int right = -1;

    /**
     * @brief Start index of primitives in leaf nodes.
     *
     * @details
     * Refers to an index into a primitive index buffer.
     * Valid only when @ref is_leaf is `true`.
     */
    int start = -1;

    /**
     * @brief Number of primitives stored in this leaf.
     *
     * @details
     * Must be positive when @ref is_leaf is `true`.
     * Must be zero for internal nodes.
     */
    int count = 0;

    /**
     * @brief Indicates whether this node is a leaf.
     *
     * @details
     * - `true`  → leaf node (uses @ref start and @ref count)
     * - `false` → internal node (uses @ref left and @ref right)
     */
    bool is_leaf = false;
};

} // namespace atlas::spatial