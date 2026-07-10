#pragma once

#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas {

/**
 * @brief One node of a bounding-volume hierarchy, laid out for flat array storage.
 *
 * Both LBVH and SAHBVH populate arrays of this struct on the host and upload
 * them verbatim to the device, so the layout is a POD with plain integer child
 * links rather than pointers. A node is either an internal node (children in
 * @ref left / @ref right, @ref is_leaf false) or a leaf referencing a
 * contiguous run of primitives (@ref start / @ref count, @ref is_leaf true).
 * The solid-angle moments are accumulated bottom-up so an interior node
 * summarizes the emissive geometry of its whole subtree, letting a traversal
 * approximate a distant subtree by its aggregate instead of descending into it.
 *
 * @note Leaf primitive indices are indices into the BVH's reordered `indices`
 *       array, not directly into the original triangle array.
 */
struct BVHNode {

    /// World-space bound enclosing this node's subtree (or leaf primitives).
    AABB bounds;

    /// Area-weighted sum of triangle centroids in the subtree (first moment);
    /// dividing by @ref solid_angle_area recovers the area-weighted centroid.
    Float3 solid_angle_moment = Float3(0.0f, 0.0f, 0.0f);

    /// Sum of triangle area-scaled normals (each `0.5 * cross(b-a, c-a)`) in the
    /// subtree; its magnitude and direction summarize net orientation and area.
    Float3 solid_angle_normal_area = Float3(0.0f, 0.0f, 0.0f);

    /// Total triangle surface area accumulated in the subtree.
    float solid_angle_area = 0.0f;

    /// Index of the left child node, or -1 when this is a leaf.
    int left = -1;

    /// Index of the right child node, or -1 when this is a leaf.
    int right = -1;

    /// First primitive offset into the reordered index array; -1 for interior nodes.
    int start = -1;

    /// Number of primitives referenced by this leaf; 0 for interior nodes.
    int count = 0;

    /// True for a leaf node, false for an interior node.
    bool is_leaf = false;
};

}