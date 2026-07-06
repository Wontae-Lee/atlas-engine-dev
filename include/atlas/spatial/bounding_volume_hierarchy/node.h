#pragma once

#include <atlas/spatial/axis_aligned_bounding_box.h>

/**
 * @file node.h
 * @brief One node of a triangle-mesh bounding volume hierarchy (built
 *        by `LBVH`/`SAHBVH`): a bounding box for fast ray/point
 *        traversal, plus precomputed low-order moments enabling a fast
 *        *approximate* generalized winding number evaluation for
 *        point-in-mesh tests.
 *
 * @details
 * ### Background — fast winding numbers via a BVH
 * `TriangleMeshGeometryOperator::is_inside` needs a robust
 * "is point `p` inside this mesh" test that works even for meshes that
 * aren't watertight/manifold (open boundaries, self-intersections,
 * multiple components) — the generalized winding number `w(p) =
 * (1/4*pi) * sum_over_triangles(solid_angle subtended by that triangle
 * at p)` (Jacobson, Kavan & Sorkine-Hornung, 2013) is exactly such a
 * test: `w(p) ~= 1` for points well inside, `~= 0` well outside,
 * regardless of mesh defects (`is_inside` checks `|w(p)| > 0.5`).
 * Computed exactly, this is `O(triangle_count)` per query — too slow for
 * per-particle-per-step evaluation. Barill, Dickson, Schmidt, Levin &
 * Jacobson's "Fast Winding Numbers for Soups and Clouds" (SIGGRAPH
 * 2018) makes it hierarchical: each BVH node precomputes a low-order
 * multipole approximation of its subtree's aggregate solid-angle
 * contribution — `solid_angle_area` (total triangle area, the 0th
 * moment), `solid_angle_moment` (area-weighted centroid sum, `sum(area_i
 * * centroid_i)`, related to the 1st moment), and
 * `solid_angle_normal_area` (area-weighted normal sum,
 * `sum(cross(b-a,c-a)/2)`, needed to approximate the subtree's aggregate
 * signed solid angle without summing every triangle's exact
 * contribution). A traversal that is "far enough" from a node (relative
 * to that node's spatial extent) can use this cheap moment-based
 * approximation for the whole subtree instead of descending into every
 * leaf, giving `O(log n)`-ish query cost in practice — see
 * `TriangleMeshGeometryOperator::fast_winding_number_bvh` for the
 * traversal itself.
 *
 * Interior nodes merge their two children's moments by simple addition
 * (`LBVH::merge_solid_angle_moment`/`SAHBVH::merge_solid_angle_moment`):
 * since each moment is itself a sum over the subtree's triangles, the
 * union of two disjoint triangle sets' sums is just the sum of their
 * sums.
 *
 * ### References
 * - A. Jacobson, L. Kavan, and O. Sorkine-Hornung, "Robust Inside-Outside
 *   Segmentation using Generalized Winding Numbers," ACM Transactions
 *   on Graphics 32(4), 2013 (SIGGRAPH). (the generalized winding number
 *   itself)
 * - G. Barill, N. Dickson, R. Schmidt, D. I. W. Levin, and A. Jacobson,
 *   "Fast Winding Numbers for Soups and Clouds," ACM Transactions on
 *   Graphics 37(4), 2018 (SIGGRAPH). (the BVH-hierarchical
 *   moment-approximation evaluation this node stores data for)
 */

namespace atlas {

/**
 * @brief One BVH node: bounding box for spatial traversal, plus the
 *        precomputed winding-number moments for its subtree. See this
 *        file's top-of-file documentation.
 */
struct BVHNode {

    /** World-space bound of every primitive in this node's subtree. */
    AABB bounds;

    /** Area-weighted centroid sum over this subtree's triangles
     *  (`sum(area_i * centroid_i)`), the 1st-moment term of the
     *  hierarchical winding-number approximation. */
    Vector3 solid_angle_moment = Vector3(0.0f, 0.0f, 0.0f);

    /** Area-weighted (unnormalized) normal sum over this subtree's
     *  triangles, needed to approximate the subtree's aggregate signed
     *  solid angle at a distant query point. */
    Vector3 solid_angle_normal_area = Vector3(0.0f, 0.0f, 0.0f);

    /** Total triangle area in this subtree (the 0th-moment term). */
    float solid_angle_area = 0.0f;

    /** Left child node index, or `-1` for a leaf. */
    int left = -1;

    /** Right child node index, or `-1` for a leaf. */
    int right = -1;

    /** Index into the primitive-index array where this leaf's
     *  primitives begin; `-1` for an interior node. */
    int start = -1;

    /** Number of primitives in this leaf; `0` for an interior node. */
    int count = 0;

    bool is_leaf = false;
};

}
