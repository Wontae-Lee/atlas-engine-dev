#pragma once
#include <algorithm>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel.h>
#include <cmath>
#include <limits>

namespace atlas::spatial {

template <typename T>
BvhGeometryOperator<T>
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::make_geometry_operator() const {
    /**
     * @brief Create a lightweight BVH query operator from the current SAH BVH.
     *
     * @return A @ref BvhGeometryOperator that references the current device-side
     *         BVH node array, primitive index array, triangle array, and root index.
     *
     * @details
     * The returned operator is a non-owning view over the internal device buffers.
     * It is intended for fast traversal/query code and therefore only exports:
     * - `bvh_nodes`   : pointer to the flat BVH node array
     * - `bvh_indices` : pointer to the primitive index indirection array
     * - `bvh_tris`    : pointer to the packed triangle array
     * - `bvh_root`    : index of the root node in the flat node array
     *
     * The validity of the returned operator depends on:
     * - this BVH object staying alive,
     * - the underlying device buffers not being reallocated or cleared.
     */
    BvhGeometryOperator<T> op;

    /// Export the flat BVH node storage.
    op.bvh_nodes = atlas::raw_pointer_cast(d_nodes.data());

    /// Export the primitive index storage.
    op.bvh_indices = atlas::raw_pointer_cast(d_indices.data());

    /// Export the packed triangle payload storage.
    op.bvh_tris = atlas::raw_pointer_cast(d_triangles.data());

    /// Export the root-node index used as the traversal entry point.
    op.bvh_root = _root;

    return op;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_leaf_size(const int leaf_size) noexcept {
    /**
     * @brief Set the preferred maximum number of primitives per leaf.
     *
     * @param leaf_size Requested leaf size.
     *
     * @details
     * The value is clamped to a minimum of `1` so every leaf remains meaningful.
     * A smaller leaf size generally produces:
     * - deeper trees,
     * - more internal nodes,
     * - potentially tighter traversal pruning.
     *
     * A larger leaf size generally produces:
     * - shallower trees,
     * - fewer internal nodes,
     * - more primitive tests per leaf traversal.
     */
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_num_of_bins(int num_bins) noexcept {
    /**
     * @brief Set the number of SAH bins used during splitting.
     *
     * @param num_bins Requested bin count.
     *
     * @details
     * The value is clamped to the implementation-supported range:
     * - minimum : `4`
     * - maximum : `256`
     *
     * More bins provide finer split resolution at the cost of more work
     * during BVH construction.
     */
    if (num_bins < 4) num_bins = 4;
    if (num_bins > 256) num_bins = 256;
    _num_of_bins = num_bins;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::leaf_size() const noexcept {
    /**
     * @brief Return the configured leaf size threshold.
     *
     * @return Maximum preferred primitive count per leaf.
     */
    return _leaf_size;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::num_of_bins() const noexcept {
    /**
     * @brief Return the configured SAH bin count.
     *
     * @return Number of bins used for split evaluation.
     */
    return _num_of_bins;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::root() const noexcept {
    /**
     * @brief Return the root node index of the current BVH.
     *
     * @return Root node index, or `-1` if no valid BVH is currently built.
     */
    return _root;
}

template <typename T>
const HostBuffer<BVHNode<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::nodes() const noexcept {
    /**
     * @brief Return the host-side flat BVH node array.
     *
     * @return Const reference to the host node buffer.
     */
    return h_nodes;
}

template <typename T>
const HostBuffer<int>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::indices() const noexcept {
    /**
     * @brief Return the host-side primitive index ordering.
     *
     * @return Const reference to the host primitive index buffer.
     *
     * @details
     * This array stores the primitive permutation produced during recursive
     * partitioning. Leaf nodes reference contiguous subranges of this array.
     */
    return h_indices;
}

template <typename T>
const HostBuffer<AABB<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::bounds() const noexcept {
    /**
     * @brief Return cached host-side primitive bounding boxes.
     *
     * @return Const reference to the primitive AABB buffer.
     */
    return h_prim_bounds;
}

template <typename T>
const HostBuffer<Vector3<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::centroids() const noexcept {
    /**
     * @brief Return cached host-side primitive centroids.
     *
     * @return Const reference to the primitive centroid buffer.
     */
    return h_centroids;
}

template <typename T>
const DeviceBuffer<BVHNode<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_nodes() const noexcept {
    /**
     * @brief Return the device-side flat BVH node array.
     *
     * @return Const reference to the device node buffer.
     */
    return d_nodes;
}

template <typename T>
const DeviceBuffer<int>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_indices() const noexcept {
    /**
     * @brief Return the device-side primitive index array.
     *
     * @return Const reference to the device primitive index buffer.
     */
    return d_indices;
}

template <typename T>
const DeviceBuffer<TriangleContainer4<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_triangles() const noexcept {
    /**
     * @brief Return the device-side packed triangle payload array.
     *
     * @return Const reference to the device triangle buffer.
     */
    return d_triangles;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build(const HostBuffer<TriangleContainer4<T>>& triangles) {
    /**
     * @brief Build a surface-area-heuristic BVH from a host-side triangle array.
     *
     * @param triangles Input triangle array.
     *
     * @details
     * The build pipeline is:
     * 1. Reset any previous BVH state.
     * 2. Cache one primitive AABB and one centroid per triangle.
     * 3. Initialize the primitive index array to the identity ordering.
     * 4. Allocate enough host-side nodes for the worst-case binary tree:
     *    `2 * n - 1`.
     * 5. Recursively partition the primitive range using a binned SAH split rule.
     * 6. Shrink the host node array to the actual number of created nodes.
     * 7. Upload nodes, indices, and triangle payloads to device buffers.
     *
     * Leaf nodes store:
     * - `start` : first primitive slot in `h_indices`
     * - `count` : number of primitives in that contiguous leaf range
     *
     * Internal nodes store:
     * - `left`  : left child node index
     * - `right` : right child node index
     *
     * The resulting tree is stored in a flat array, and `_root` records
     * the root node index.
     */
    const int n = static_cast<int>(triangles.size());
    ///< Number of input primitives.

    reset();
    ///< Discard any previously built tree and cached data.

    if (n <= 0) return;
    ///< Empty input yields an empty BVH.

    h_prim_bounds.resize(n);
    ///< Allocate one AABB per primitive.

    h_centroids.resize(n);
    ///< Allocate one centroid per primitive.

    h_indices.resize(n);
    ///< Allocate primitive permutation / indexing storage.

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            geometry::TriangleGeometryOperator<T> tri_op;
            ///< Temporary triangle query operator for primitive `i`.

            tri_op.a = &triangles[i].a();
            tri_op.b = &triangles[i].b();
            tri_op.c = &triangles[i].c();
            tri_op.n = &triangles[i].d();
            ///< Bind the operator to the i-th packed triangle.

            const AABB<T> bounds = tri_op.bound();
            ///< Compute the primitive AABB.

            h_prim_bounds[i] = bounds;
            ///< Cache primitive bounds.

            h_centroids[i] = tri_op.centroid();
            ///< Cache primitive centroid.

            h_indices[i] = i;
            ///< Initialize primitive ordering to identity.
        });

    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());
    /**
     * @brief Reserve worst-case binary-tree node count.
     *
     * @details
     * A full binary tree over `n` leaves has at most `2n - 1` nodes.
     * This preallocation avoids repeated growth during recursive construction.
     */

    int next_node = 0;
    ///< Monotonic counter assigning flat-array node indices during recursion.

    _root = build_recursive(0, n, next_node);
    ///< Build the whole tree over the primitive range `[0, n)`.

    h_nodes.resize(next_node);
    ///< Shrink to the actual number of nodes created.

    d_nodes.resize(n);
    d_indices.resize(n);
    d_triangles.resize(n);
    /**
     * @brief Resize device buffers before upload.
     *
     * @note
     * This code resizes `d_nodes` to `n` before assignment from `h_nodes`.
     * The subsequent assignment is assumed to replace/resize appropriately
     * according to the device-buffer implementation.
     */

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
    ///< Upload final BVH and primitive data to device storage.
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build_recursive(int start, const int end, int& node_count) {
    /**
     * @brief Recursively build one BVH subtree over a primitive subrange.
     *
     * @param start First primitive slot in the current range.
     * @param end One-past-the-last primitive slot in the current range.
     * @param[in,out] node_count Running count used to allocate flat node indices.
     * @return Flat-array index of the subtree root node.
     *
     * @details
     * This routine:
     * 1. Creates a node slot for the current range.
     * 2. Computes:
     *    - the union of primitive bounds,
     *    - the bounds of primitive centroids.
     * 3. Creates a leaf if:
     *    - the range size is below the leaf threshold, or
     *    - the centroid bounds are degenerate.
     * 4. Otherwise:
     *    - chooses the longest centroid-extent axis,
     *    - bins primitives along that axis,
     *    - evaluates all split candidates with a binned SAH cost,
     *    - partitions the range with `std::stable_partition`,
     *    - recurses into left and right children,
     *    - stores child links and merged bounds in the internal node.
     */
    const int node_index = node_count++;
    ///< Assign one flat-array node index to this subtree root.

    if (node_index >= static_cast<int>(h_nodes.size())) {
        h_nodes.resize(node_index + 1);
    }
    ///< Ensure storage exists even if the preallocation was insufficient.

    AABB<T> node_bounds;
    ///< Union of primitive bounds for the current range.

    AABB<T> centroid_bounds;
    ///< Union of primitive centroids for the current range.

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];
        ///< Primitive id stored at the current range slot.

        node_bounds.merge(h_prim_bounds[pid]);
        ///< Expand the node AABB with the primitive bounds.

        centroid_bounds.merge(h_centroids[pid]);
        ///< Expand centroid bounds with the primitive centroid.
    }

    const Vector3<T> ext = centroid_bounds.extents();
    ///< Extent of the centroid bounding box.

    const bool degenerate = (ext.x <= eps) && (ext.y <= eps) && (ext.z <= eps);
    /**
     * @brief Detect whether centroid spread is effectively zero on all axes.
     *
     * @details
     * In this case, further spatial partitioning is not meaningful, so the
     * current range is emitted as a leaf.
     */

    const int count = end - start;
    ///< Number of primitives in the current range.

    if (count <= _leaf_size || degenerate) {
        BVHNode<T>& leaf = h_nodes[node_index];
        ///< Node storage for the leaf.

        leaf.is_leaf = true;
        ///< Mark this node as a leaf.

        leaf.left = leaf.right = -1;
        ///< Leaves have no child links.

        leaf.bounds = node_bounds;
        ///< Leaf AABB is the union of the enclosed primitives.

        leaf.start = start;
        ///< First primitive slot covered by the leaf.

        leaf.count = count;
        ///< Number of contiguous primitive slots covered by the leaf.

        return node_index;
    }

    int axis = ext.major_axis();
    ///< Split axis chosen as the axis of maximum centroid extent.

    const T cmin = centroid_bounds.lower_corner.at(axis);
    ///< Minimum centroid coordinate along the chosen split axis.

    const T cmax = centroid_bounds.upper_corner.at(axis);
    ///< Maximum centroid coordinate along the chosen split axis.

    const T den = cmax - cmin;
    ///< Denominator used to normalize centroids into [0, 1] bin space.

    if (den <= T(0)) {
        /**
         * @brief Fallback leaf when the selected split axis has no usable spread.
         *
         * @details
         * Even if the full centroid bounds were not globally marked degenerate,
         * numerical collapse along the chosen axis prevents safe binning.
         */
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    HostBuffer<sah::Bin<T>> bins(_num_of_bins);
    ///< SAH bins used to accumulate counts and bounds along the chosen axis.

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];
        ///< Primitive id currently being binned.

        const T t = (h_centroids[pid].at(axis) - cmin) / den;
        ///< Normalized centroid position in [approximately] [0, 1].

        const int b = std::clamp(
            static_cast<int>(std::floor(t * _num_of_bins)),
            0,
            _num_of_bins - 1);
        ///< Discrete bin index for the primitive centroid.

        if (bins[b].count == 0) {
            bins[b].bounds = h_prim_bounds[pid];
            ///< First primitive in this bin initializes the bin AABB.
        } else {
            bins[b].bounds.merge(h_prim_bounds[pid]);
            ///< Subsequent primitives grow the bin AABB.
        }

        ++bins[b].count;
        ///< Count one more primitive in this bin.
    }

    HostBuffer<AABB<T>> prefix_bounds(_num_of_bins), suffix_bounds(_num_of_bins);
    ///< Prefix/suffix unions of bin AABBs.

    HostBuffer<int> prefix_counts(_num_of_bins, 0), suffix_counts(_num_of_bins, 0);
    ///< Prefix/suffix primitive counts for split evaluation.

    AABB<T> acc_bounds;
    ///< Running AABB accumulator for prefix/suffix scans.

    int acc_count = 0;
    ///< Running primitive-count accumulator for prefix/suffix scans.

    for (int i = 0; i < _num_of_bins; ++i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
            ///< Prefix union absorbs the current non-empty bin bounds.
        }

        acc_count += bins[i].count;
        ///< Prefix count absorbs the current bin count.

        prefix_bounds[i] = acc_bounds;
        ///< Store prefix AABB up to and including bin i.

        prefix_counts[i] = acc_count;
        ///< Store prefix primitive count up to and including bin i.
    }

    acc_bounds = AABB<T>();
    ///< Reset AABB accumulator for suffix pass.

    acc_count = 0;
    ///< Reset count accumulator for suffix pass.

    for (int i = _num_of_bins - 1; i >= 0; --i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
            ///< Suffix union absorbs the current non-empty bin bounds.
        }

        acc_count += bins[i].count;
        ///< Suffix count absorbs the current bin count.

        suffix_bounds[i] = acc_bounds;
        ///< Store suffix AABB from bin i through the last bin.

        suffix_counts[i] = acc_count;
        ///< Store suffix primitive count from bin i through the last bin.
    }

    const T parent_area = node_bounds.area();
    ///< Surface area of the current node bounds used for SAH normalization.

    T best_cost = std::numeric_limits<T>::max();
    ///< Best SAH cost found so far.

    int best_split = -1;
    ///< Best bin split position found so far.

    for (int s = 0; s < _num_of_bins - 1; ++s) {
        const int lc = prefix_counts[s];
        ///< Primitive count on the left side of split s.

        const int rc = suffix_counts[s + 1];
        ///< Primitive count on the right side of split s.

        if (lc == 0 || rc == 0) continue;
        ///< Invalid split: one side would be empty.

        const T cost = T(1) + (prefix_bounds[s].area() * T(lc) + suffix_bounds[s + 1].area() * T(rc)) / parent_area;
        /**
         * @brief Standard binned SAH cost model.
         *
         * @details
         * The constant `1` is the traversal cost term, and the remaining term
         * approximates expected primitive-intersection work after splitting.
         */

        if (cost < best_cost) {
            best_cost  = cost;
            best_split = s;
            ///< Keep the lowest-cost split.
        }
    }

    if (best_split < 0) {
        /**
         * @brief Fallback leaf when no valid SAH split exists.
         *
         * @details
         * This happens when every candidate split would leave one side empty.
         */
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    auto first = h_indices.begin() + start;
    ///< Iterator to the first primitive slot in the current range.

    auto last = h_indices.begin() + end;
    ///< Iterator one past the last primitive slot in the current range.

    auto mid_it = std::stable_partition(
        first,
        last,
        [&](int pid) {
            const T t = (h_centroids[pid].at(axis) - cmin) / den;
            ///< Normalized centroid position of the candidate primitive.

            const int b = std::clamp(
                static_cast<int>(std::floor(t * _num_of_bins)),
                0,
                _num_of_bins - 1);
            ///< Candidate primitive bin.

            return b <= best_split;
            ///< Left partition contains bins `[0, best_split]`.
        });
    /**
     * @brief Partition the primitive range according to the chosen bin split.
     *
     * @details
     * `std::stable_partition` preserves the relative order within the left and
     * right subsets, which can help keep construction deterministic.
     */

    const int left_count = static_cast<int>(mid_it - first);
    ///< Number of primitives assigned to the left child.

    if (left_count <= 0 || left_count >= count) {
        /**
         * @brief Fallback leaf when partitioning produced a degenerate split.
         *
         * @details
         * Even a valid bin split can collapse after classification if every
         * primitive lands on the same side.
         */
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    const int left_child = build_recursive(start, start + left_count, node_count);
    ///< Recursively build the left subtree.

    const int right_child = build_recursive(start + left_count, end, node_count);
    ///< Recursively build the right subtree.

    BVHNode<T>& node = h_nodes[node_index];
    ///< Current internal node storage.

    node.is_leaf = false;
    ///< Mark this node as internal.

    node.left = left_child;
    ///< Store left child node index.

    node.right = right_child;
    ///< Store right child node index.

    node.start = -1;
    ///< Internal nodes do not directly own primitive ranges.

    node.count = 0;
    ///< Internal nodes store no leaf primitive count.

    node.bounds = h_nodes[left_child].bounds;
    ///< Initialize internal bounds from the left child.

    node.bounds.merge(h_nodes[right_child].bounds);
    ///< Grow internal bounds to include the right child.

    return node_index;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::reset() {
    /**
     * @brief Clear all host-side and device-side BVH data.
     *
     * @details
     * After reset:
     * - host node/index/centroid/bounds buffers are empty,
     * - device node/index/triangle buffers are empty,
     * - `_root` is set to `-1` to indicate an invalid / absent tree.
     */
    h_nodes.clear();
    ///< Clear host-side flat node storage.

    h_indices.clear();
    ///< Clear host-side primitive ordering.

    h_centroids.clear();
    ///< Clear cached primitive centroids.

    h_prim_bounds.clear();
    ///< Clear cached primitive bounds.

    d_nodes.clear();
    ///< Clear device-side node storage.

    d_indices.clear();
    ///< Clear device-side primitive ordering.

    d_triangles.clear();
    ///< Clear device-side triangle payload storage.

    _root = -1;
    ///< Mark the BVH as empty / invalid.
}
} // namespace atlas::spatial