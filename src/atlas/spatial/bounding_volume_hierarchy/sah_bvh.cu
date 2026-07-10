#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace atlas {

BvhView
SAHBVH::view() const {
    BvhView view {};

    view.bvh_nodes   = atlas::raw_pointer_cast(d_nodes.data());
    view.bvh_indices = atlas::raw_pointer_cast(d_indices.data());
    view.bvh_tris    = atlas::raw_pointer_cast(d_triangles.data());
    view.bvh_root    = _root;

    return view;
}

void
SAHBVH::build(
    const HostBuffer<TriangleContainer4>& triangles) {
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
            Triangle tri_op;

            tri_op.a = triangles[i].a();
            tri_op.b = triangles[i].b();
            tri_op.c = triangles[i].c();
            tri_op.n = triangles[i].d();

            const AABB bounds = tri_op.bound();
            h_prim_bounds[i]  = bounds;
            h_centroids[i]    = tri_op.centroid();
            h_indices[i]      = i;
        });

    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode());

    int next_node = 0;
    _root         = build_recursive(0, n, next_node, triangles);

    h_nodes.resize(next_node);

    d_nodes.resize(n);
    d_indices.resize(n);
    d_triangles.resize(n);

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

void
SAHBVH::assign_solid_angle_moment(
    BVHNode& node,
    const int start,
    const int end,
    const HostBuffer<TriangleContainer4>& triangles) const noexcept {
    node.solid_angle_moment      = Float3(0.0f, 0.0f, 0.0f);
    node.solid_angle_normal_area = Float3(0.0f, 0.0f, 0.0f);
    node.solid_angle_area        = 0.0f;

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];

        const Float3& a = triangles[pid].a();
        const Float3& b = triangles[pid].b();
        const Float3& c = triangles[pid].c();

        const Float3 normal_area = atlas::cross(b - a, c - a) * 0.5f;
        const float area         = normal_area.length();

        if (!(area > 0.0f)) {
            continue;
        }

        node.solid_angle_moment += (a + b + c) * (area / 3.0f);
        node.solid_angle_normal_area += normal_area;
        node.solid_angle_area += area;
    }
}

void
SAHBVH::merge_solid_angle_moment(
    BVHNode& node,
    const BVHNode& left,
    const BVHNode& right) noexcept {
    node.solid_angle_moment      = left.solid_angle_moment + right.solid_angle_moment;
    node.solid_angle_normal_area = left.solid_angle_normal_area + right.solid_angle_normal_area;
    node.solid_angle_area        = left.solid_angle_area + right.solid_angle_area;
}

int
SAHBVH::build_recursive(
    int start,
    const int end,
    int& node_count,
    const HostBuffer<TriangleContainer4>& triangles) {

    const int node_index = node_count++;

    if (node_index >= static_cast<int>(h_nodes.size())) {
        h_nodes.resize(node_index + 1);
    }

    AABB node_bounds;
    AABB centroid_bounds;

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];

        node_bounds.merge(h_prim_bounds[pid]);
        centroid_bounds.merge(h_centroids[pid]);
    }

    const Float3 ext      = centroid_bounds.extents();
    // All centroids coincide (or nearly so) on every axis: there is no
    // meaningful spatial spread left to split on, so stop recursing rather
    // than attempt (and fail) to find a useful SAH split.
    const bool degenerate = atlas::all(ext <= Float3(eps));
    const int count       = end - start;

    if (count <= _leaf_size || degenerate) {
        BVHNode& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        assign_solid_angle_moment(leaf, start, end, triangles);

        return node_index;
    }

    int axis         = ext.major_axis();
    const float cmin = centroid_bounds.lower_corner.at(axis);
    const float cmax = centroid_bounds.upper_corner.at(axis);
    const float den  = cmax - cmin;

    if (den <= 0.0f) {
        BVHNode& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        assign_solid_angle_moment(leaf, start, end, triangles);

        return node_index;
    }

    // Bin every primitive's centroid into _bin_count equal-width buckets
    // along the chosen (longest-centroid-spread) axis — O(count) instead of
    // sorting exactly, at the cost of approximating rather than exactly
    // evaluating the SAH cost at every possible split (see sah_bvh.h's
    // top-of-file documentation for the binned-SAH derivation).
    HostBuffer<Bin> bins(_bin_count);

    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];
        const float t = (h_centroids[pid].at(axis) - cmin) / den;
        const int b   = std::clamp(
            static_cast<int>(std::floor(t * _bin_count)),
            0,
            _bin_count - 1);

        if (bins[b].count == 0) {
            bins[b].bounds = h_prim_bounds[pid];
        } else {
            bins[b].bounds.merge(h_prim_bounds[pid]);
        }

        ++bins[b].count;
    }

    // Prefix/suffix sweeps turn per-bin bounds/counts into "bounds and
    // count of everything left of boundary s" / "... right of boundary
    // s+1" in O(bin_count) total, so the cost loop below can evaluate
    // every candidate split boundary in O(1) each instead of re-merging
    // bins per candidate.
    HostBuffer<AABB> prefix_bounds(_bin_count), suffix_bounds(_bin_count);
    HostBuffer<int> prefix_counts(_bin_count, 0), suffix_counts(_bin_count, 0);

    AABB acc_bounds;
    int acc_count = 0;

    for (int i = 0; i < _bin_count; ++i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
        }

        acc_count += bins[i].count;
        prefix_bounds[i] = acc_bounds;
        prefix_counts[i] = acc_count;
    }

    acc_bounds = AABB();
    acc_count  = 0;

    for (int i = _bin_count - 1; i >= 0; --i) {
        if (bins[i].count > 0) {
            acc_bounds.merge(bins[i].bounds);
        }

        acc_count += bins[i].count;
        suffix_bounds[i] = acc_bounds;
        suffix_counts[i] = acc_count;
    }

    // Binned SAH cost at boundary s: C = C_trav + (A_left/A_parent)*N_left
    // + (A_right/A_parent)*N_right, with C_trav folded into the constant
    // "1.0f" term and C_isect implicitly 1 per primitive (see sah_bvh.h's
    // top-of-file documentation for the cost-model derivation). A boundary
    // with all primitives on one side (lc or rc == 0) is skipped — it isn't
    // a real split.
    const float parent_area = node_bounds.area();
    float best_cost         = std::numeric_limits<float>::max();
    int best_split          = -1;

    for (int s = 0; s < _bin_count - 1; ++s) {
        const int lc = prefix_counts[s];
        const int rc = suffix_counts[s + 1];

        if (lc == 0 || rc == 0) continue;

        const float cost = 1.0f
            + (prefix_bounds[s].area() * static_cast<float>(lc)
               + suffix_bounds[s + 1].area() * static_cast<float>(rc))
                / parent_area;

        if (cost < best_cost) {
            best_cost  = cost;
            best_split = s;
        }
    }

    if (best_split < 0) {
        BVHNode& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        assign_solid_angle_moment(leaf, start, end, triangles);

        return node_index;
    }

    // Physically reorder h_indices[start, end) so every primitive whose bin
    // is <= best_split comes first — turns the abstract "split at bin
    // boundary s" decision into a concrete contiguous [start, start+left_count)
    // / [start+left_count, end) partition the two recursive calls below can
    // operate on directly. stable_partition (not a full sort) keeps this
    // O(count).
    auto first = h_indices.begin() + start;
    auto last  = h_indices.begin() + end;

    auto mid_it = std::stable_partition(
        first,
        last,
        [&](int pid) {
            const float t = (h_centroids[pid].at(axis) - cmin) / den;
            const int b   = std::clamp(
                static_cast<int>(std::floor(t * _bin_count)),
                0,
                _bin_count - 1);

            return b <= best_split;
        });

    const int left_count = static_cast<int>(mid_it - first);

    // Defensive: the binned cost estimate chose a boundary with primitives
    // on both sides, but binning is only an approximation of centroid
    // position — a degenerate distribution could still partition to an
    // empty side. Fall back to a leaf rather than recurse into a
    // zero-size child.
    if (left_count <= 0 || left_count >= count) {
        BVHNode& leaf = h_nodes[node_index];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        assign_solid_angle_moment(leaf, start, end, triangles);

        return node_index;
    }

    const int left_child  = build_recursive(start, start + left_count, node_count, triangles);
    const int right_child = build_recursive(start + left_count, end, node_count, triangles);

    BVHNode& node = h_nodes[node_index];

    node.is_leaf = false;
    node.left    = left_child;
    node.right   = right_child;
    node.start   = -1;
    node.count   = 0;
    node.bounds  = h_nodes[left_child].bounds;
    node.bounds.merge(h_nodes[right_child].bounds);
    merge_solid_angle_moment(node, h_nodes[left_child], h_nodes[right_child]);

    return node_index;
}

void
SAHBVH::reset() {

    h_nodes.clear();
    h_indices.clear();
    h_centroids.clear();
    h_prim_bounds.clear();

    d_nodes.clear();
    d_indices.clear();
    d_triangles.clear();

    _root = -1;
}

}
