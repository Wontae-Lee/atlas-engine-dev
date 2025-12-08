#pragma once
#include <algorithm>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel.h>
#include <cmath>
#include <limits>

namespace atlas::spatial {
template <typename T>
BvhTraceOperator<T>
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::make_trace_operator() const {
    BvhTraceOperator<T> op;
    op.nodes   = atlas::raw_pointer_cast(d_nodes.data());
    op.indices = atlas::raw_pointer_cast(d_indices.data());
    op.tris    = atlas::raw_pointer_cast(d_triangles.data());
    op.root    = _root;
    return op;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_leaf_size(int leaf_size) noexcept {
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::set_num_of_bins(int num_bins) noexcept {
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
const DeviceBuffer<Triangle<T>>&
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::device_triangles() const noexcept {
    return d_triangles;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build(const HostBuffer<Triangle<T>>& triangles) {
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
            const AABB<T> bounds = triangles[i].bound();
            h_prim_bounds[i]     = bounds;
            h_centroids[i]       = bounds.center();
            h_indices[i]         = i;
        });
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());
    int next_node = 0;
    _root         = build_recursive(0, n, next_node);
    h_nodes.resize(next_node);
    d_nodes.resize(n);
    d_indices.resize(n);
    d_triangles.resize(n);
    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

template <typename T>
int
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::build_recursive(int start, int end, int& node_count) {
    constexpr T k_eps    = T(1e-8);
    const int node_index = node_count++;
    if (node_index >= static_cast<int>(h_nodes.size()))
        h_nodes.resize(node_index + 1);
    AABB<T> node_bounds;
    AABB<T> centroid_bounds;
    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];
        node_bounds.merge(h_prim_bounds[pid]);
        centroid_bounds.merge(h_centroids[pid]);
    }
    const Vector3<T> ext  = centroid_bounds.extents();
    const bool degenerate = (ext.x <= k_eps) && (ext.y <= k_eps) && (ext.z <= k_eps);
    const int count       = end - start;
    if (count <= _leaf_size || degenerate) {
        BVHNode<T>& leaf = h_nodes[node_index];
        leaf.is_leaf     = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        return node_index;
    }
    int axis     = ext.major_axis();
    const T cmin = centroid_bounds.lower_corner.at(axis);
    const T cmax = centroid_bounds.upper_corner.at(axis);
    const T den  = cmax - cmin;
    if (den <= T(0)) {
        BVHNode<T>& leaf = h_nodes[node_index];
        leaf.is_leaf     = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        return node_index;
    }
    HostBuffer<sah::Bin<T>> bins(_num_of_bins);
    for (int i = start; i < end; ++i) {
        const int pid = h_indices[i];
        const T t     = (h_centroids[pid].at(axis) - cmin) / den;
        const int b   = std::clamp(static_cast<int>(std::floor(t * _num_of_bins)), 0, _num_of_bins - 1);
        if (bins[b].count == 0)
            bins[b].bounds = h_prim_bounds[pid];
        else
            bins[b].bounds.merge(h_prim_bounds[pid]);
        ++bins[b].count;
    }
    HostBuffer<AABB<T>> prefix_bounds(_num_of_bins), suffix_bounds(_num_of_bins);
    HostBuffer<int> prefix_counts(_num_of_bins, 0), suffix_counts(_num_of_bins, 0);
    AABB<T> acc_bounds;
    int acc_count = 0;
    for (int i = 0; i < _num_of_bins; ++i) {
        if (bins[i].count > 0)
            acc_bounds.merge(bins[i].bounds);
        acc_count += bins[i].count;
        prefix_bounds[i] = acc_bounds;
        prefix_counts[i] = acc_count;
    }
    acc_bounds = AABB<T>();
    acc_count  = 0;
    for (int i = _num_of_bins - 1; i >= 0; --i) {
        if (bins[i].count > 0)
            acc_bounds.merge(bins[i].bounds);
        acc_count += bins[i].count;
        suffix_bounds[i] = acc_bounds;
        suffix_counts[i] = acc_count;
    }
    const T parent_area = node_bounds.area();
    T best_cost         = std::numeric_limits<T>::max();
    int best_split      = -1;
    for (int s = 0; s < _num_of_bins - 1; ++s) {
        const int lc = prefix_counts[s];
        const int rc = suffix_counts[s + 1];
        if (lc == 0 || rc == 0) continue;
        const T cost = T(1) + (prefix_bounds[s].area() * T(lc) + suffix_bounds[s + 1].area() * T(rc)) / parent_area;
        if (cost < best_cost) {
            best_cost  = cost;
            best_split = s;
        }
    }
    if (best_split < 0) {
        BVHNode<T>& leaf = h_nodes[node_index];
        leaf.is_leaf     = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        return node_index;
    }
    auto first           = h_indices.begin() + start;
    auto last            = h_indices.begin() + end;
    auto mid_it          = std::stable_partition(first, last, [&](int pid) {
        const T t   = (h_centroids[pid].at(axis) - cmin) / den;
        const int b = std::clamp(static_cast<int>(std::floor(t * _num_of_bins)), 0, _num_of_bins - 1);
        return b <= best_split;
    });
    const int left_count = static_cast<int>(mid_it - first);
    if (left_count <= 0 || left_count >= count) {
        BVHNode<T>& leaf = h_nodes[node_index];
        leaf.is_leaf     = true;
        leaf.left = leaf.right = -1;
        leaf.bounds            = node_bounds;
        leaf.start             = start;
        leaf.count             = count;
        return node_index;
    }
    const int left_child  = build_recursive(start, start + left_count, node_count);
    const int right_child = build_recursive(start + left_count, end, node_count);
    BVHNode<T>& node      = h_nodes[node_index];
    node.is_leaf          = false;
    node.left             = left_child;
    node.right            = right_child;
    node.start            = -1;
    node.count            = 0;
    node.bounds           = h_nodes[left_child].bounds;
    node.bounds.merge(h_nodes[right_child].bounds);
    return node_index;
}

template <typename T>
void
SurfaceAreaHeuristicBoundingVolumeHierachy<T>::reset() {
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