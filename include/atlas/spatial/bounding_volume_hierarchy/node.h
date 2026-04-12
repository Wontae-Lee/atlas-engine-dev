#pragma once

/**
 * @file node.h
 * @brief Declares the node type shared by BVH implementations.
 */
#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas::spatial {

/**
 * @brief Single node in a bounding volume hierarchy.
 */
template <typename T>
struct BVHNode {
    AABB<T> bounds; ///< Bounding box covering this subtree or leaf range.
    int left = -1; ///< Left child node index for internal nodes.
    int right = -1; ///< Right child node index for internal nodes.
    int start = -1; ///< Primitive start index for leaf nodes.
    int count = 0; ///< Primitive count for leaf nodes.
    bool is_leaf = false; ///< Whether this node stores primitives directly.
};

}
