#pragma once
#include <algorithm>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel.h>

namespace atlas::spatial {
template <typename T>
BvhTraceOperator<T>
LinearBoundingVolumeHierachy<T>::make_trace_operator() const {
    BvhTraceOperator<T> op;
    op.nodes   = atlas::raw_pointer_cast(d_nodes.data());
    op.indices = atlas::raw_pointer_cast(d_indices.data());
    op.tris    = atlas::raw_pointer_cast(d_triangles.data());
    op.root    = _root;
    return op;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_leaf_size(int leaf_size) noexcept {
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_morton_bits(int morton_bits) noexcept {
    if (morton_bits < 1) morton_bits = 1;
    if (morton_bits > 10) morton_bits = 10;
    _morton_bits = morton_bits;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_size() const noexcept {
    return _leaf_size;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::morton_bits() const noexcept {
    return _morton_bits;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::root() const noexcept {
    return _root;
}

template <typename T>
const HostBuffer<BVHNode<T>>&
LinearBoundingVolumeHierachy<T>::nodes() const noexcept {
    return h_nodes;
}

template <typename T>
const HostBuffer<int>&
LinearBoundingVolumeHierachy<T>::indices() const noexcept {
    return h_indices;
}

template <typename T>
const HostBuffer<AABB<T>>&
LinearBoundingVolumeHierachy<T>::bounds() const noexcept {
    return h_prim_bounds;
}

template <typename T>
const HostBuffer<Vector3<T>>&
LinearBoundingVolumeHierachy<T>::centroids() const noexcept {
    return h_centroids;
}

template <typename T>
const DeviceBuffer<BVHNode<T>>&
LinearBoundingVolumeHierachy<T>::device_nodes() const noexcept {
    return d_nodes;
}

template <typename T>
const DeviceBuffer<int>&
LinearBoundingVolumeHierachy<T>::device_indices() const noexcept {
    return d_indices;
}

template <typename T>
const DeviceBuffer<Triangle<T>>&
LinearBoundingVolumeHierachy<T>::device_triangles() const noexcept {
    return d_triangles;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_node_index(int k, int n) noexcept {
    return (n - 1) + k;
}

template <typename T>
unsigned
LinearBoundingVolumeHierachy<T>::expand_bits(unsigned v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::clz32(uint32_t x) noexcept {
    if (x == 0u) return 32;
    int n      = 0;
    uint32_t m = 1u << 31;
    while ((x & m) == 0u) {
        ++n;
        m >>= 1;
    }
    return n;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::clz64(uint64_t x) noexcept {
    if (x == 0u) return 64;
    int n      = 0;
    uint64_t m = 1ull << 63;
    while ((x & m) == 0u) {
        ++n;
        m >>= 1;
    }
    return n;
}

template <typename T>
uint32_t
LinearBoundingVolumeHierachy<T>::morton3(const Vector3<T>& p, const AABB<T>& cb, int bits) const noexcept {
    const Vector3<T>& minp = cb.lower_corner;
    const Vector3<T>& maxp = cb.upper_corner;
    const Vector3<T> ext { maxp.x - minp.x, maxp.y - minp.y, maxp.z - minp.z };
    T nx = (ext.x > T(0)) ? (p.x - minp.x) / ext.x : T(0);
    T ny = (ext.y > T(0)) ? (p.y - minp.y) / ext.y : T(0);
    T nz = (ext.z > T(0)) ? (p.z - minp.z) / ext.z : T(0);
    if (nx < T(0)) nx = T(0);
    if (nx > T(1)) nx = T(1);
    if (ny < T(0)) ny = T(0);
    if (ny > T(1)) ny = T(1);
    if (nz < T(0)) nz = T(0);
    if (nz > T(1)) nz = T(1);
    const unsigned maxq = (1u << bits) - 1u;
    const unsigned ix   = static_cast<unsigned>(nx * maxq + T(0.5));
    const unsigned iy   = static_cast<unsigned>(ny * maxq + T(0.5));
    const unsigned iz   = static_cast<unsigned>(nz * maxq + T(0.5));
    const unsigned xx   = expand_bits(ix);
    const unsigned yy   = expand_bits(iy);
    const unsigned zz   = expand_bits(iz);
    return (xx << 2) | (yy << 1) | (zz << 0);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept {
    if (j < 0 || j >= n) return -1;
    const uint64_t a = keys[i];
    const uint64_t b = keys[j];
    if (a == b) return 64;
    return clz64(a ^ b);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept {
    const uint32_t first_code = codes[first];
    const uint32_t last_code  = codes[last];
    if (first_code == last_code) return (first + last) >> 1;
    const int common_prefix = clz32(first_code ^ last_code);
    int split               = first;
    int step                = last - first;
    do {
        step                = (step + 1) >> 1;
        const int new_split = split + step;
        if (new_split < last) {
            const int split_prefix = clz32(first_code ^ codes[new_split]);
            if (split_prefix > common_prefix) split = new_split;
        }
    } while (step > 1);
    return split;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::build(const HostBuffer<Triangle<T>>& triangles) {
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
            const AABB<T> b  = triangles[i].bound();
            h_prim_bounds[i] = b;
            h_centroids[i]   = b.center();
            h_indices[i]     = i;
        });
    AABB<T> centroid_bounds;
    for (int i = 0; i < n; ++i) centroid_bounds.merge(h_centroids[i]);
    HostBuffer<uint32_t> morton(n);
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
        });
    HostBuffer<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (morton[a] != morton[b]) return morton[a] < morton[b];
        return a < b;
    });
    HostBuffer<uint32_t> morton_sorted(n);
    HostBuffer<uint64_t> keys_sorted(n);
    HostBuffer<int> indices_sorted(n);
    for (int i = 0; i < n; ++i) {
        const int oi      = order[i];
        morton_sorted[i]  = morton[oi];
        keys_sorted[i]    = (static_cast<uint64_t>(morton_sorted[i]) << 32) | static_cast<uint32_t>(i);
        indices_sorted[i] = h_indices[oi];
    }
    h_indices = indices_sorted;
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());
    for (int k = 0; k < n; ++k) {
        const int ni     = leaf_node_index(k, n);
        const int pid    = h_indices[k];
        BVHNode<T>& leaf = h_nodes[ni];
        leaf.is_leaf     = true;
        leaf.left = leaf.right = -1;
        leaf.start             = k;
        leaf.count             = 1;
        leaf.bounds            = h_prim_bounds[pid];
    }
    if (n == 1) {
        _root       = leaf_node_index(0, n);
        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;
        return;
    }
    for (int i = 0; i < n - 1; ++i) {
        const int dl        = delta_lcp(keys_sorted, n, i, i - 1);
        const int dr        = delta_lcp(keys_sorted, n, i, i + 1);
        const int d         = (dr > dl) ? 1 : -1;
        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);
        int lmax            = 2;
        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) { lmax <<= 1; }
        int t    = 0;
        int step = lmax;
        do {
            step = (step + 1) >> 1;
            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) t += step;
        } while (step > 1);
        const int j             = i + t * d;
        const int first         = std::min(i, j);
        const int last          = std::max(i, j);
        const int split         = find_split(morton_sorted, first, last);
        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;
        const int left_child    = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child   = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);
        BVHNode<T>& in          = h_nodes[i];
        in.is_leaf              = false;
        in.left                 = left_child;
        in.right                = right_child;
        in.start                = -1;
        in.count                = 0;
    }
    for (int i = n - 2; i >= 0; --i) {
        BVHNode<T>& in      = h_nodes[i];
        const BVHNode<T>& L = h_nodes[in.left];
        const BVHNode<T>& R = h_nodes[in.right];
        in.bounds           = L.bounds;
        in.bounds.merge(R.bounds);
    }
    _root       = 0;
    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::reset() {
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