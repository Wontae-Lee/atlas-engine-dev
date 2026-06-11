#pragma once

#include <algorithm>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::spatial {

template <typename T>
BvhGeometryOperator<T>
LinearBoundingVolumeHierachy<T>::make_geometry_operator() const {
    BvhGeometryOperator<T> op;

    // Expose device-side BVH buffers through raw pointers.
    op.bvh_nodes   = atlas::raw_pointer_cast(d_nodes.data());
    op.bvh_indices = atlas::raw_pointer_cast(d_indices.data());
    op.bvh_tris    = atlas::raw_pointer_cast(d_triangles.data());
    op.bvh_root    = _root;

    return op;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_leaf_size(const int leaf_size) noexcept {
    // Keep at least one primitive per leaf.
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_morton_bits(int morton_bits) noexcept {
    // Clamp Morton quantization bits to the supported 30-bit Morton layout.
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
void
LinearBoundingVolumeHierachy<T>::assign_solid_angle_moment(
    BVHNode<T>& node,
    const TriangleContainer4<T>& triangle) noexcept {
    const Vector3<T>& a = triangle.a();
    const Vector3<T>& b = triangle.b();
    const Vector3<T>& c = triangle.c();

    const Vector3<T> normal_area = atlas::math::cross(b - a, c - a) * T(0.5);
    const T area                 = normal_area.length();

    node.solid_angle_moment      = Vector3<T>(T(0), T(0), T(0));
    node.solid_angle_normal_area = Vector3<T>(T(0), T(0), T(0));
    node.solid_angle_area        = T(0);

    if (!(area > T(0))) {
        return;
    }

    node.solid_angle_moment      = (a + b + c) * (area / T(3));
    node.solid_angle_normal_area = normal_area;
    node.solid_angle_area        = area;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::merge_solid_angle_moment(
    BVHNode<T>& node,
    const BVHNode<T>& left,
    const BVHNode<T>& right) noexcept {
    node.solid_angle_moment      = left.solid_angle_moment + right.solid_angle_moment;
    node.solid_angle_normal_area = left.solid_angle_normal_area + right.solid_angle_normal_area;
    node.solid_angle_area        = left.solid_angle_area + right.solid_angle_area;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_node_index(const int k, const int n) noexcept {
    // In a 2n-1 node LBVH layout, leaves are stored after the n-1 internal nodes.
    return (n - 1) + k;
}

template <typename T>
unsigned
LinearBoundingVolumeHierachy<T>::expand_bits(unsigned v) noexcept {
    // Interleave lower 10 bits with two zero bits between each original bit.
    v = (v * atlas::seed::MORTON_EXPAND_BITS_FIRST_MULTIPLIER)
        & atlas::seed::MORTON_EXPAND_BITS_FIRST_MASK;
    v = (v * atlas::seed::MORTON_EXPAND_BITS_SECOND_MULTIPLIER)
        & atlas::seed::MORTON_EXPAND_BITS_SECOND_MASK;
    v = (v * atlas::seed::MORTON_EXPAND_BITS_THIRD_MULTIPLIER)
        & atlas::seed::MORTON_EXPAND_BITS_THIRD_MASK;
    v = (v * atlas::seed::MORTON_EXPAND_BITS_FINAL_MULTIPLIER)
        & atlas::seed::MORTON_EXPAND_BITS_FINAL_MASK;

    return v;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::clz32(const uint32_t x) noexcept {
    // Count leading zero bits in a 32-bit integer.
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
    // Count leading zero bits in a 64-bit integer.
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
LinearBoundingVolumeHierachy<T>::morton3(
    const Vector3<T>& p,
    const AABB<T>& cb,
    const int bits) const noexcept {
    const Vector3<T>& minp = cb.lower_corner;
    const Vector3<T>& maxp = cb.upper_corner;

    // Compute centroid bounds extent.
    const Vector3<T> ext {
        maxp.x - minp.x,
        maxp.y - minp.y,
        maxp.z - minp.z
    };

    // Normalize the point into [0, 1]^3 inside the centroid bounds.
    T nx = (ext.x > T(0)) ? (p.x - minp.x) / ext.x : T(0);
    T ny = (ext.y > T(0)) ? (p.y - minp.y) / ext.y : T(0);
    T nz = (ext.z > T(0)) ? (p.z - minp.z) / ext.z : T(0);

    // Clamp normalized coordinates to the valid Morton domain.
    if (nx < T(0)) nx = T(0);
    if (nx > T(1)) nx = T(1);
    if (ny < T(0)) ny = T(0);
    if (ny > T(1)) ny = T(1);
    if (nz < T(0)) nz = T(0);
    if (nz > T(1)) nz = T(1);

    // Quantize normalized coordinates to the requested bit resolution.
    const unsigned maxq = (1u << bits) - 1u;
    const auto ix       = static_cast<unsigned>(nx * maxq + T(0.5));
    const auto iy       = static_cast<unsigned>(ny * maxq + T(0.5));
    const auto iz       = static_cast<unsigned>(nz * maxq + T(0.5));

    // Expand bits and interleave x, y, and z into a 30-bit Morton code.
    const unsigned xx = expand_bits(ix);
    const unsigned yy = expand_bits(iy);
    const unsigned zz = expand_bits(iz);

    return (xx << 2) | (yy << 1) | (zz << 0);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::delta_lcp(
    const HostBuffer<uint64_t>& keys,
    const int n,
    const int i,
    const int j) noexcept {
    // Out-of-range neighbors are treated as invalid.
    if (j < 0 || j >= n) return -1;

    const uint64_t a = keys[i];
    const uint64_t b = keys[j];

    // Identical keys have the maximum common prefix length.
    if (a == b) return 64;

    // The leading zeros of xor(a, b) give the common prefix length.
    return clz64(a ^ b);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::find_split(
    const HostBuffer<uint32_t>& codes,
    const int first,
    const int last) noexcept {
    const uint32_t first_code = codes[first];
    const uint32_t last_code  = codes[last];

    // Equal Morton codes cannot be separated by prefix length, so split midway.
    if (first_code == last_code) return (first + last) >> 1;

    const int common_prefix = clz32(first_code ^ last_code);
    int split               = first;
    int step                = last - first;

    // Binary search for the last index with a longer prefix than the range prefix.
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
    const int n = static_cast<int>(triangles.size());

    // Start from a clean tree before rebuilding.
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

            // Build a lightweight triangle operator over the input triangle storage.
            tri_op.a = &triangles[i].a();
            tri_op.b = &triangles[i].b();
            tri_op.c = &triangles[i].c();
            tri_op.n = &triangles[i].d();

            // Cache primitive bounds, centroids, and original primitive indices.
            const AABB<T> b  = tri_op.bound();
            h_prim_bounds[i] = b;
            h_centroids[i]   = tri_op.centroid();
            h_indices[i]     = i;
        });

    AABB<T> centroid_bounds;

    // Compute bounds over all primitive centroids for Morton normalization.
    for (int i = 0; i < n; ++i) {
        centroid_bounds.merge(h_centroids[i]);
    }

    HostBuffer<uint32_t> morton(n);

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            // Compute Morton code from the normalized primitive centroid.
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
        });

    HostBuffer<int> order(n);

    // Initialize sortable primitive order.
    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }

    std::stable_sort(
        order.begin(),
        order.end(),
        [&](const int a, const int b) {
            // Sort by Morton code, then by primitive id for deterministic ties.
            if (morton[a] != morton[b]) return morton[a] < morton[b];

            return a < b;
        });

    HostBuffer<uint32_t> morton_sorted(n);
    HostBuffer<uint64_t> keys_sorted(n);
    HostBuffer<int> indices_sorted(n);

    for (int i = 0; i < n; ++i) {
        const int oi = order[i];

        // Store sorted Morton codes and unique 64-bit keys for LCP tests.
        morton_sorted[i] = morton[oi];
        keys_sorted[i]   =
            (static_cast<uint64_t>(morton_sorted[i]) << 32)
            | static_cast<uint32_t>(i);

        indices_sorted[i] = h_indices[oi];
    }

    // Replace primitive order with Morton-sorted primitive indices.
    h_indices = indices_sorted;

    // LBVH uses n - 1 internal nodes and n leaf nodes.
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());

    for (int k = 0; k < n; ++k) {
        const int ni     = leaf_node_index(k, n);
        const int pid    = h_indices[k];
        BVHNode<T>& leaf = h_nodes[ni];

        // Initialize one primitive per leaf in Morton order.
        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.start = k;
        leaf.count = 1;
        leaf.bounds = h_prim_bounds[pid];
        assign_solid_angle_moment(leaf, triangles[pid]);
    }

    if (n == 1) {
        // A single primitive tree consists only of one leaf root.
        _root       = leaf_node_index(0, n);
        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;

        return;
    }

    for (int i = 0; i < n - 1; ++i) {
        // Determine range direction by comparing common prefixes with neighbors.
        const int dl = delta_lcp(keys_sorted, n, i, i - 1);
        const int dr = delta_lcp(keys_sorted, n, i, i + 1);
        const int d  = (dr > dl) ? 1 : -1;

        // Find the minimum prefix length that defines the current range boundary.
        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);

        int lmax = 2;

        // Exponentially grow the candidate range while prefix length remains valid.
        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) {
            lmax <<= 1;
        }

        int t    = 0;
        int step = lmax;

        // Binary search the exact range length.
        do {
            step = (step + 1) >> 1;

            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) {
                t += step;
            }
        } while (step > 1);

        const int j     = i + t * d;
        const int first = std::min(i, j);
        const int last  = std::max(i, j);

        // Find the split position inside the Morton-code range.
        const int split = find_split(morton_sorted, first, last);

        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;

        // Convert split positions to either leaf-node indices or internal-node indices.
        const int left_child  = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);

        BVHNode<T>& in = h_nodes[i];

        // Store internal node topology.
        in.is_leaf = false;
        in.left    = left_child;
        in.right   = right_child;
        in.start   = -1;
        in.count   = 0;
    }

    // Build internal node bounds bottom-up after topology is known.
    for (int i = n - 2; i >= 0; --i) {
        BVHNode<T>& in      = h_nodes[i];
        const BVHNode<T>& L = h_nodes[in.left];
        const BVHNode<T>& R = h_nodes[in.right];

        in.bounds = L.bounds;
        in.bounds.merge(R.bounds);
        merge_solid_angle_moment(in, L, R);
    }

    // Internal node 0 is the LBVH root for n > 1.
    _root = 0;

    // Upload completed hierarchy and primitives to device buffers.
    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::reset() {
    // Clear host and device data before rebuilding.
    h_nodes.clear();
    h_indices.clear();
    h_centroids.clear();
    h_prim_bounds.clear();

    d_nodes.clear();
    d_indices.clear();
    d_triangles.clear();

    _root = -1;
}

} // namespace atlas::spatial
