#pragma once
#include <algorithm>
#include <atlas/geometry/query_operator.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::spatial {

template <typename T>
BvhTraceOperator<T>
LinearBoundingVolumeHierachy<T>::make_trace_operator() const {
    // Expose raw pointers so the traversal operator becomes a light-weight view
    // over the already-built buffers. Ownership stays in the BVH object.
    BvhTraceOperator<T> op;

    op.nodes   = atlas::raw_pointer_cast(d_nodes.data());
    op.indices = atlas::raw_pointer_cast(d_indices.data());
    op.tris    = atlas::raw_pointer_cast(d_triangles.data());

    op.root = _root;
    return op;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_leaf_size(const int leaf_size) noexcept {
    // Enforce the invariant that each leaf stores at least one primitive.
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_morton_bits(int morton_bits) noexcept {
    // The 30-bit Morton code implementation dilates at most 10 bits per axis.
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
const DeviceBuffer<TriangleContainer4<T>>&
LinearBoundingVolumeHierachy<T>::device_triangles() const noexcept {

    return d_triangles;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_node_index(const int k, const int n) noexcept {
    // Node layout:
    //   [0, n-2]     -> internal nodes
    //   [n-1, 2n-2]  -> leaves in Morton order
    return (n - 1) + k;
}

template <typename T>
unsigned
LinearBoundingVolumeHierachy<T>::expand_bits(unsigned v) noexcept {
    // Successive masking and multiplication spreads each bit so that the final
    // pattern occupies every third bit position:
    //   abcdefghij -> a00b00c00...j00
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::clz32(const uint32_t x) noexcept {
    // Count-leading-zeros approximates the depth at which two radix-tree paths
    // diverge: more shared leading bits means stronger spatial locality.
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
LinearBoundingVolumeHierachy<T>::clz64(const uint64_t x) noexcept {
    // 64-bit variant used because the implementation appends the sorted index to
    // the Morton code, making ties deterministic even for identical centroids.
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
LinearBoundingVolumeHierachy<T>::morton3(const Vector3<T>& p, const AABB<T>& cb, const int bits) const noexcept {
    // Normalize the centroid into the unit cube:
    //   n_i = (p_i - min_i) / (max_i - min_i).
    // Degenerate axes collapse to zero to avoid division by zero.
    const Vector3<T>& minp = cb.lower_corner;
    const Vector3<T>& maxp = cb.upper_corner;
    const Vector3<T> ext { maxp.x - minp.x, maxp.y - minp.y, maxp.z - minp.z };

    T nx = (ext.x > T(0)) ? (p.x - minp.x) / ext.x : T(0);
    T ny = (ext.y > T(0)) ? (p.y - minp.y) / ext.y : T(0);
    T nz = (ext.z > T(0)) ? (p.z - minp.z) / ext.z : T(0);

    // Clamp into [0, 1] so numerically stray centroids still map inside the
    // quantization domain.
    if (nx < T(0)) nx = T(0);
    if (nx > T(1)) nx = T(1);
    if (ny < T(0)) ny = T(0);
    if (ny > T(1)) ny = T(1);
    if (nz < T(0)) nz = T(0);
    if (nz > T(1)) nz = T(1);

    // Uniform scalar quantization maps [0, 1] to integers in [0, 2^bits - 1].
    const unsigned maxq = (1u << bits) - 1u;
    const auto ix       = static_cast<unsigned>(nx * maxq + T(0.5));
    const auto iy       = static_cast<unsigned>(ny * maxq + T(0.5));
    const auto iz       = static_cast<unsigned>(nz * maxq + T(0.5));

    const unsigned xx = expand_bits(ix);
    const unsigned yy = expand_bits(iy);
    const unsigned zz = expand_bits(iz);

    // Interleave the expanded coordinate bits as x2 y2 z2 x1 y1 z1 ...
    return (xx << 2) | (yy << 1) | (zz << 0);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::delta_lcp(const HostBuffer<uint64_t>& keys, const int n, const int i, const int j) noexcept {
    // Out-of-range comparisons behave like negative infinity so the direction
    // search naturally stops at array boundaries.
    if (j < 0 || j >= n) return -1;
    const uint64_t a = keys[i];
    const uint64_t b = keys[j];
    // Identical keys share the maximum possible prefix length.
    if (a == b) return 64;
    return clz64(a ^ b);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::find_split(const HostBuffer<uint32_t>& codes, const int first, const int last) noexcept {
    // The split is the last index whose code shares a longer prefix with
    // first_code than the full interval [first, last] does.
    const uint32_t first_code = codes[first];
    const uint32_t last_code  = codes[last];

    if (first_code == last_code) return (first + last) >> 1;

    const int common_prefix = clz32(first_code ^ last_code);

    // Binary search on prefix length avoids scanning the whole interval.
    int split = first;
    int step  = last - first;
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
LinearBoundingVolumeHierachy<T>::build(const HostBuffer<TriangleContainer4<T>>& triangles) {
    // n primitives yield n leaves and n - 1 internal nodes in a full binary BVH.
    const int n = static_cast<int>(triangles.size());

    reset();
    if (n <= 0) return;

    h_prim_bounds.resize(n);
    h_centroids.resize(n);
    h_indices.resize(n);

    // Precompute primitive bounds and centroids. The centroid acts as the point
    // representative used by the Morton mapping.
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            geometry::TriangleQueryOperator<T> tri_op;
            tri_op.a         = &triangles[i].a();
            tri_op.b         = &triangles[i].b();
            tri_op.c         = &triangles[i].c();
            tri_op.n         = &triangles[i].d();
            const AABB<T> b  = tri_op.bound();
            h_prim_bounds[i] = b;
            h_centroids[i]   = tri_op.centroid();
            h_indices[i]     = i;
        });

    // The global centroid bounds define the affine map from world space into the
    // Morton unit cube.
    AABB<T> centroid_bounds;
    for (int i = 0; i < n; ++i) centroid_bounds.merge(h_centroids[i]);

    HostBuffer<uint32_t> morton(n);
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
        });

    // Sort by Morton code so nearby centroids in space become nearby keys in the
    // linear array, which approximates a depth-first spatial clustering.
    HostBuffer<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;

    std::stable_sort(order.begin(),
                     order.end(),
                     [&](const int a, const int b) {
                         if (morton[a] != morton[b]) return morton[a] < morton[b];

                         return a < b;
                     });

    HostBuffer<uint32_t> morton_sorted(n);
    HostBuffer<uint64_t> keys_sorted(n);
    HostBuffer<int> indices_sorted(n);

    // Attach the stable sorted index to each Morton code. This breaks ties while
    // preserving deterministic ordering for duplicate codes.
    for (int i = 0; i < n; ++i) {
        const int oi      = order[i];
        morton_sorted[i]  = morton[oi];
        keys_sorted[i]    = (static_cast<uint64_t>(morton_sorted[i]) << 32) | static_cast<uint32_t>(i);
        indices_sorted[i] = h_indices[oi];
    }

    h_indices = indices_sorted;

    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());

    // Initialize all leaves first; internal nodes are filled afterwards.
    for (int k = 0; k < n; ++k) {
        const int ni     = leaf_node_index(k, n);
        const int pid    = h_indices[k];
        BVHNode<T>& leaf = h_nodes[ni];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.start             = k;
        leaf.count             = 1;
        leaf.bounds            = h_prim_bounds[pid];
    }

    if (n == 1) {
        _root = leaf_node_index(0, n);

        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;
        return;
    }

    // Karras-style radix tree construction:
    // choose the build direction by comparing LCP with neighbors, determine the
    // maximal range sharing that prefix, then split inside that range.
    for (int i = 0; i < n - 1; ++i) {
        const int dl = delta_lcp(keys_sorted, n, i, i - 1);
        const int dr = delta_lcp(keys_sorted, n, i, i + 1);
        const int d  = (dr > dl) ? 1 : -1;

        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);

        int lmax = 2;
        // Exponential search brackets the range length, then binary refinement
        // locates the exact interval endpoint.
        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) { lmax <<= 1; }

        int t    = 0;
        int step = lmax;
        do {
            step = (step + 1) >> 1;
            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) t += step;
        } while (step > 1);

        const int j     = i + t * d;
        const int first = std::min(i, j);
        const int last  = std::max(i, j);

        const int split = find_split(morton_sorted, first, last);

        // Child ranges of length one correspond directly to leaves.
        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;

        const int left_child  = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);

        BVHNode<T>& in = h_nodes[i];
        in.is_leaf     = false;
        in.left        = left_child;
        in.right       = right_child;
        in.start       = -1;
        in.count       = 0;
    }

    // Bottom-up union of child boxes computes conservative bounds for each
    // internal node after the topology is fixed.
    for (int i = n - 2; i >= 0; --i) {
        BVHNode<T>& in      = h_nodes[i];
        const BVHNode<T>& L = h_nodes[in.left];
        const BVHNode<T>& R = h_nodes[in.right];
        in.bounds           = L.bounds;
        in.bounds.merge(R.bounds);
    }

    _root = 0;

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::reset() {
    // Clear both host and device mirrors so a later build starts from a known
    // empty state and stale traversal operators become obviously invalid.
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
