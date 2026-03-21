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
    // Like the LBVH version, this returns a non-owning view over BVH storage.
    BvhGeometryOperator<T> op;

    op.bvh_nodes   = atlas::raw_pointer_cast(d_nodes.data());
    op.bvh_indices = atlas::raw_pointer_cast(d_indices.data());
    op.bvh_tris    = atlas::raw_pointer_cast(d_triangles.data());

    op.bvh_root = _root;

    return op;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_leaf_size(const int leaf_size) noexcept {
    // Leaves must contain at least one primitive.
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_num_of_bins(int num_bins) noexcept {
    // Very small histograms give unstable SAH estimates; very large histograms
    // increase build cost with diminishing returns.
    if (num_bins < 4) num_bins = 4;
    if (num_bins > 256) num_bins = 256;
    _num_of_bins = num_bins;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::leaf_size() const noexcept {

    return _leaf_size;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::num_of_bins() const noexcept {

    return _num_of_bins;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::root() const noexcept {

    return _root;
}

template <typename T>
const HostBuffer<BVHNode<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::nodes() const noexcept {

    return h_nodes;
}

template <typename T>
const HostBuffer<int>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::indices() const noexcept {

    return h_indices;
}

template <typename T>
const HostBuffer<AABB<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::bounds() const noexcept {

    return h_prim_bounds;
}

template <typename T>
const HostBuffer<Vector3<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::centroids() const noexcept {

    return h_centroids;
}

template <typename T>
const DeviceBuffer<BVHNode<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_nodes() const noexcept {

    return d_nodes;
}

template <typename T>
const DeviceBuffer<int>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_indices() const noexcept {

    return d_indices;
}

template <typename T>
const DeviceBuffer<TriangleContainer4<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_triangles() const noexcept {

    return d_triangles;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build(const HostBuffer<TriangleContainer4<T>>& triangles) {
    // Build primitive metadata once; recursive partitioning reorders indices
    // rather than copying triangle geometry.
    const int n = static_cast<int>(triangles.size());

    reset();

    if (n <= 0) return;

    h_prim_bounds.resize(n);
    h_centroids.resize(n);
    h_indices.resize(n);

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            geometry::TriangleGeometryOperator<T> tri_op;
            // TriangleContainer4 exposes one primitive through four packed slots;
            // the query operator consumes raw addresses and computes geometry
            // properties without materializing another triangle object.
            tri_op.a             = &triangles[i].a();
            tri_op.b             = &triangles[i].b();
            tri_op.c             = &triangles[i].c();
            tri_op.n             = &triangles[i].d();
            const AABB<T> bounds = tri_op.bound();
            // The builder keeps all heavy geometry in place and only shuffles
            // these metadata arrays plus the primitive index permutation.
            h_prim_bounds[i]     = bounds;
            h_centroids[i]       = tri_op.centroid();
            h_indices[i]         = i;
        });

    // Preallocate the worst-case full binary tree size.
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());

    // build_recursive writes bvh_nodes into h_nodes linearly. next_node is the
    // monotonic allocator cursor for that array and starts at the future bvh_root.
    int next_node = 0;

    // The bvh_root covers the full primitive permutation range [0, n). Recursive
    // calls will keep partitioning that index interval into smaller subranges.
    _root = build_recursive(0, n, next_node);

    // Worst-case space was reserved up front; shrink back to the number of
    // bvh_nodes actually emitted by the recursive builder.
    h_nodes.resize(next_node);

    // Device buffers mirror the compacted host arrays after construction.
    d_nodes.resize(n);
    d_indices.resize(n);
    d_triangles.resize(n);

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}
template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build_recursive(int start, const int end, int& node_count) {
    // ------------------------------------------------------------------
    // Recursively build one BVH node over the half-open primitive range:
    //   [start, end)
    //
    // Important convention:
    // - The range is expressed in terms of `h_indices`, not the original triangle array.
    // - `h_indices` is a permutation of primitive ids, so this range selects a
    //   contiguous subset of the CURRENT working order.
    //
    // Return value:
    // - index of the node created in `h_nodes`
    //
    // node_count:
    // - acts as an append cursor into `h_nodes`
    // - incremented each time a new node is allocated
    // ------------------------------------------------------------------

    // Reserve one node slot for the current subtree bvh_root.
    // This node may later become either:
    // - a leaf, or
    // - an internal node with two children.
    const int node_index = node_count++;

    // Ensure host node buffer is large enough to hold the newly allocated node.
    // Recursive construction may discover bvh_nodes in depth-first order, so we grow
    // on demand rather than precomputing the exact size here.
    if (node_index >= static_cast<int>(h_nodes.size())) {
        h_nodes.resize(node_index + 1);
    }

    // node_bounds:
    // - union of primitive AABBs in this range
    // - becomes the traversal bound of the current node
    //
    // centroid_bounds:
    // - union of primitive centroids in this range
    // - used only for split-axis selection and bin normalization
    AABB<T> node_bounds;
    AABB<T> centroid_bounds;

    // Accumulate geometric bounds and centroid bounds for the current subset.
    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];

        // h_indices stores the current primitive permutation.
        // Therefore [start, end) defines a subset in permuted order, not a slice
        // of the original triangle array.
        node_bounds.merge(h_prim_bounds[pid]);
        centroid_bounds.merge(h_centroids[pid]);
    }

    // Extents of centroid distribution.
    // These tell us how much centroid spread exists in x/y/z.
    const Vector3<T> ext = centroid_bounds.extents();

    // If all centroid extents are extremely small, the primitives are effectively
    // collapsed to one point in centroid space. In that case, spatial splitting
    // becomes numerically meaningless and usually unstable.
    const bool degenerate = (ext.x <= eps) && (ext.y <= eps) && (ext.z <= eps);

    // Number of primitives represented by this subtree.
    const int count = end - start;

    // ------------------------------------------------------------------
    // Leaf termination conditions
    // ------------------------------------------------------------------
    //
    // Stop recursion when:
    // 1) the range is already small enough according to `_leaf_size`, or
    // 2) the centroid distribution is degenerate, so no useful split exists
    //
    // In both cases, emit a leaf node that directly references the primitive
    // range [start, end) through `h_indices`.
    if (count <= _leaf_size || degenerate) {
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;

        // Leaves have no children.
        leaf.left = leaf.right = -1;

        // Leaf bounds are the union of all primitives in this range.
        leaf.bounds = node_bounds;

        // Leaves refer back into the index permutation:
        // traversal can later read primitive ids from
        //   h_indices[start ... start + count - 1]
        leaf.start = start;
        leaf.count = count;

        return node_index;
    }

    // Choose the axis with the largest centroid extent.
    //
    // Heuristic:
    // - The longest centroid axis often provides the strongest spatial separation.
    // - SAH binning below is then performed only along this 1D axis.
    int axis = ext.major_axis();

    // The centroid coordinate interval [cmin, cmax] defines the normalization
    // range for mapping centroids into histogram bins.
    const T cmin = centroid_bounds.lower_corner.at(axis);
    const T cmax = centroid_bounds.upper_corner.at(axis);
    const T den  = cmax - cmin;

    // If the chosen axis has zero span, all centroids share the same coordinate
    // along that axis. That means no valid 1D partition can be formed here.
    //
    // Fallback: emit a leaf instead of forcing an invalid split.
    if (den <= T(0)) {
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    // ------------------------------------------------------------------
    // SAH binning setup
    // ------------------------------------------------------------------
    //
    // Surface Area Heuristic (SAH) is approximated by histogram binning:
    // - primitives are grouped into `_num_of_bins` buckets along the chosen axis
    // - candidate split positions are tested only between neighboring bins
    //
    // This reduces evaluation cost from O(n^2) over all primitive split positions
    // to roughly O(n + B), where B is the number of bins.
    HostBuffer<sah::Bin<T>> bins(_num_of_bins);

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];

        // Normalize centroid coordinate into [0, 1].
        // This projects the current centroid interval [cmin, cmax] onto unit space.
        const T t = (h_centroids[pid].at(axis) - cmin) / den;

        // Convert normalized coordinate into a bin index.
        //
        // floor(t * _num_of_bins) gives the nominal bin,
        // then std::clamp keeps it inside [0, _num_of_bins - 1].
        //
        // The clamp is especially important for centroids at the upper boundary
        // (t == 1), ensuring they land in the last bin instead of overflowing.
        const int b = std::clamp(
            static_cast<int>(std::floor(t * _num_of_bins)),
            0,
            _num_of_bins - 1);

        // Each bin stores:
        // - count  : how many primitives landed in that bin
        // - bounds : union AABB of those primitives
        //
        // This is exactly the information needed later to estimate SAH cost.
        if (bins[b].count == 0) {
            bins[b].bounds = h_prim_bounds[pid];
        } else {
            bins[b].bounds.merge(h_prim_bounds[pid]);
        }

        ++bins[b].count;
    }

    // Prefix/suffix arrays are used to evaluate all split positions efficiently.
    //
    // prefix_bounds[s] = union of bins [0, s]
    // prefix_counts[s] = primitive count in bins [0, s]
    //
    // suffix_bounds[s] = union of bins [s, B-1]
    // suffix_counts[s] = primitive count in bins [s, B-1]
    HostBuffer<AABB<T>> prefix_bounds(_num_of_bins), suffix_bounds(_num_of_bins);
    HostBuffer<int> prefix_counts(_num_of_bins, 0), suffix_counts(_num_of_bins, 0);

    // ------------------------------------------------------------------
    // Prefix scan over bins
    // ------------------------------------------------------------------
    //
    // After this loop:
    // - prefix_bounds[i] is the union of all primitive AABBs from bins 0..i
    // - prefix_counts[i] is the total primitive count in bins 0..i
    AABB<T> acc_bounds;
    int acc_count = 0;

    for (int i = 0; i < _num_of_bins; ++i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
        }

        acc_count += bins[i].count;

        prefix_bounds[i] = acc_bounds;
        prefix_counts[i] = acc_count;
    }

    // ------------------------------------------------------------------
    // Suffix scan over bins
    // ------------------------------------------------------------------
    //
    // Same idea as prefix scan, but in reverse:
    // - suffix_bounds[i] is the union of bins i..B-1
    // - suffix_counts[i] is the primitive count in bins i..B-1
    acc_bounds = AABB<T>();
    acc_count  = 0;

    for (int i = _num_of_bins - 1; i >= 0; --i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
        }

        acc_count += bins[i].count;

        suffix_bounds[i] = acc_bounds;
        suffix_counts[i] = acc_count;
    }

    // ------------------------------------------------------------------
    // Evaluate SAH split candidates
    // ------------------------------------------------------------------
    //
    // parent_area is used to normalize child areas into approximate hit
    // probabilities:
    //
    //   P(child) ≈ area(child) / area(parent)
    //
    // Classical SAH:
    //   C = C_traversal + P(left) * N_left + P(right) * N_right
    //
    // Here:
    // - traversal cost is approximated as 1
    // - left/right work is approximated by primitive counts
    const T parent_area = node_bounds.area();

    T best_cost    = std::numeric_limits<T>::max();
    int best_split = -1;

    // A split after bin s means:
    // - left  child gets bins [0, s]
    // - right child gets bins [s+1, B-1]
    //
    // Therefore we only test s in [0, B-2].
    for (int s = 0; s < _num_of_bins - 1; ++s) {
        const int lc = prefix_counts[s];
        const int rc = suffix_counts[s + 1];

        // Legal split must leave at least one primitive on each side.
        if (lc == 0 || rc == 0) continue;

        // SAH estimate for this candidate split.
        const T cost = T(1) + (prefix_bounds[s].area() * T(lc) + suffix_bounds[s + 1].area() * T(rc)) / parent_area;

        if (cost < best_cost) {
            best_cost  = cost;
            best_split = s;
        }
    }

    // If no split survived, all primitives effectively fell to one side for every
    // candidate split. In that case, recursion cannot make progress safely.
    //
    // Fallback: emit a leaf.
    if (best_split < 0) {
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    auto first = h_indices.begin() + start;
    auto last  = h_indices.begin() + end;

    // ------------------------------------------------------------------
    // Partition the primitive permutation by best bin threshold
    // ------------------------------------------------------------------
    //
    // Primitives whose centroid bin satisfies:
    //   b <= best_split
    // go to the left subset.
    //
    // Others go to the right subset.
    //
    // std::stable_partition is intentionally used so that:
    // - relative order inside the left subset is preserved
    // - relative order inside the right subset is preserved
    //
    // This makes the build deterministic across runs and platforms.
    auto mid_it = std::stable_partition(
        first,
        last,
        [&](int pid) {
            const T t = (h_centroids[pid].at(axis) - cmin) / den;

            const int b = std::clamp(
                static_cast<int>(std::floor(t * _num_of_bins)),
                0,
                _num_of_bins - 1);

            // Left side = bins [0, best_split]
            // Right side = bins [best_split + 1, ...]
            return b <= best_split;
        });

    // Number of primitives routed into the left child.
    //
    // After partition:
    // - h_indices[start, start + left_count)      => left subset
    // - h_indices[start + left_count, end)        => right subset
    const int left_count = static_cast<int>(mid_it - first);

    // Guard against pathological partitions:
    // - left_count == 0   => everything went right
    // - left_count == count => everything went left
    //
    // Either case would cause infinite recursion if accepted.
    if (left_count <= 0 || left_count >= count) {
        BVHNode<T>& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;

        return node_index;
    }

    // ------------------------------------------------------------------
    // Recursive descent
    // ------------------------------------------------------------------
    //
    // Recursively build:
    // - left  subtree over [start, start + left_count)
    // - right subtree over [start + left_count, end)
    const int left_child  = build_recursive(start, start + left_count, node_count);
    const int right_child = build_recursive(start + left_count, end, node_count);

    // Fill the current node as an internal node.
    BVHNode<T>& node = h_nodes[node_index];
    node.is_leaf     = false;

    // Children are node indices returned by recursion.
    node.left  = left_child;
    node.right = right_child;

    // Internal bvh_nodes do not directly reference a primitive range.
    node.start = -1;
    node.count = 0;

    // Internal node bound is the union of its child subtree bounds.
    node.bounds = h_nodes[left_child].bounds;
    node.bounds.merge(h_nodes[right_child].bounds);

    return node_index;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::reset() {
    // ------------------------------------------------------------------
    // Reset the BVH to an empty, unbuilt state.
    //
    // This removes all cached products of a previous build on both:
    // - host side
    // - device side
    //
    // After reset():
    // - there are no bvh_nodes
    // - there is no primitive permutation
    // - there are no cached centroids/bounds
    // - the device mirrors are empty
    // - bvh_root index is invalid
    // ------------------------------------------------------------------

    // Host-side build products.
    h_nodes.clear();       // full BVH topology + bounds
    h_indices.clear();     // primitive permutation used by leaves
    h_centroids.clear();   // cached primitive centroids
    h_prim_bounds.clear(); // cached primitive AABBs

    // Device-side mirrors used by traversal/intersection kernels.
    d_nodes.clear();
    d_indices.clear();
    d_triangles.clear();

    // No valid bvh_root exists in the reset state.
    _root = -1;
}
}
