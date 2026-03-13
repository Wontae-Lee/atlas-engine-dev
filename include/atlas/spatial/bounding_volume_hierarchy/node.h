#pragma once
#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::spatial {

/**
 * @brief Node stored inside a bounding volume hierarchy.
 *
 * @details
 * Each node represents either:
 * - an internal partition node whose children cover two disjoint subsets, or
 * - a leaf node that stores a contiguous range of primitive indices.
 *
 * The geometric invariant is:
 * \f[
 *   \mathrm{bounds}(node) \supseteq \bigcup_{k \in \mathrm{primitives}(node)} B_k
 * \f]
 * where \f$B_k\f$ is the primitive AABB. Traversal uses that conservative bound
 * to reject large subsets before testing individual triangles.
 *
 * @tparam T Floating-point scalar type used by the bounding box.
 */
template <typename T>
struct BVHNode {
    /// Conservative bounding box for the entire subtree rooted at this node.
    AABB<T> bounds;

    /// Index of the left child for internal nodes, `-1` for leaves.
    int left = -1;

    /// Index of the right child for internal nodes, `-1` for leaves.
    int right = -1;

    /// Offset into the primitive index array for leaf nodes.
    int start = -1;

    /// Number of primitives referenced by a leaf node.
    int count = 0;

    /// Tagged-state flag distinguishing internal nodes from leaves.
    bool is_leaf = false;
};

} // namespace atlas::spatial
