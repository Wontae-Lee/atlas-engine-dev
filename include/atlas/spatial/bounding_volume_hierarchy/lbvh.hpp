#pragma once
#include <algorithm>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::spatial {

template <typename T>
BvhGeometryOperator<T>
LinearBoundingVolumeHierachy<T>::make_geometry_operator() const {
    // Export a lightweight BVH query operator that references the current
    // device-side node/index/triangle storage.
    //
    // The returned operator is a non-owning view and therefore depends on the
    // lifetime and stability of this BVH object's internal buffers.
    BvhGeometryOperator<T> op;

    // Expose raw pointers to the packed BVH node array.
    op.bvh_nodes = atlas::raw_pointer_cast(d_nodes.data());

    // Expose raw pointers to the primitive index indirection array.
    op.bvh_indices = atlas::raw_pointer_cast(d_indices.data());

    // Expose raw pointers to the packed triangle payload array.
    op.bvh_tris = atlas::raw_pointer_cast(d_triangles.data());

    // Export the root node index so traversal can start at the BVH entry point.
    op.bvh_root = _root;
    return op;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_leaf_size(const int leaf_size) noexcept {
    // Clamp the requested leaf size to a minimum of 1.
    //
    // A BVH leaf must contain at least one primitive to remain meaningful.
    _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::set_morton_bits(int morton_bits) noexcept {
    // Clamp the Morton resolution to the supported range.
    //
    // Current implementation assumptions:
    // - minimum supported bit depth : 1
    // - maximum supported bit depth : 10
    //
    // The upper bound keeps the expanded Morton code within the intended
    // 30-bit 3D interleaving scheme used by `expand_bits()`.
    if (morton_bits < 1) morton_bits = 1;
    if (morton_bits > 10) morton_bits = 10;
    _morton_bits = morton_bits;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_size() const noexcept {
    // Return the configured target leaf size.
    return _leaf_size;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::morton_bits() const noexcept {
    // Return the configured Morton quantization depth per axis.
    return _morton_bits;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::root() const noexcept {
    // Return the index of the BVH root node in the flat node array.
    //
    // A value of -1 indicates that no valid tree is currently built.
    return _root;
}

template <typename T>
const HostBuffer<BVHNode<T>>&
LinearBoundingVolumeHierachy<T>::nodes() const noexcept {
    // Expose the host-side flat BVH node array.
    return h_nodes;
}

template <typename T>
const HostBuffer<int>&
LinearBoundingVolumeHierachy<T>::indices() const noexcept {
    // Expose the host-side primitive index ordering used by the BVH.
    return h_indices;
}

template <typename T>
const HostBuffer<AABB<T>>&
LinearBoundingVolumeHierachy<T>::bounds() const noexcept {
    // Expose host-side primitive bounds cached during BVH construction.
    return h_prim_bounds;
}

template <typename T>
const HostBuffer<Vector3<T>>&
LinearBoundingVolumeHierachy<T>::centroids() const noexcept {
    // Expose host-side primitive centroids cached during BVH construction.
    return h_centroids;
}

template <typename T>
const DeviceBuffer<BVHNode<T>>&
LinearBoundingVolumeHierachy<T>::device_nodes() const noexcept {
    // Expose the device-side flat BVH node array.
    return d_nodes;
}

template <typename T>
const DeviceBuffer<int>&
LinearBoundingVolumeHierachy<T>::device_indices() const noexcept {
    // Expose the device-side primitive index array.
    return d_indices;
}

template <typename T>
const DeviceBuffer<TriangleContainer4<T>>&
LinearBoundingVolumeHierachy<T>::device_triangles() const noexcept {
    // Expose the device-side packed triangle payload array.
    return d_triangles;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::leaf_node_index(const int k, const int n) noexcept {
    // Map the k-th leaf to its position in the flat BVH node array.
    //
    // Layout convention used by this LBVH:
    // - internal nodes occupy indices [0, n - 2]
    // - leaf nodes occupy     indices [n - 1, 2n - 2]
    //
    // Therefore leaf k begins at:
    //     (n - 1) + k
    return (n - 1) + k;
}

template <typename T>
unsigned
LinearBoundingVolumeHierachy<T>::expand_bits(unsigned v) noexcept {
    // Expand up to 10 low-order bits so that zeros are inserted between them.
    //
    // This is the classic Morton bit-interleaving helper. For an input:
    //     b9 b8 ... b0
    // it produces:
    //     00b9 00b8 ... 00b0
    //
    // The magic constants implement staged bit spreading using masks.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::clz32(const uint32_t x) noexcept {
    // Count leading zero bits in a 32-bit unsigned integer.
    //
    // This software fallback is used to measure common-prefix length in
    // Morton codes when building the LBVH hierarchy.
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
    // Count leading zero bits in a 64-bit unsigned integer.
    //
    // Used when comparing full 64-bit sort keys that combine:
    // - the Morton code in the high 32 bits
    // - a unique index in the low 32 bits
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
    // Compute a 3D Morton code for point `p` within centroid bounds `cb`.
    //
    // Steps:
    // 1. Normalize p into [0, 1]^3 relative to the centroid bounding box.
    // 2. Clamp each coordinate to [0, 1].
    // 3. Quantize each coordinate to `bits` bits.
    // 4. Interleave the quantized x/y/z bits into one 30-bit Morton code.

    const Vector3<T>& minp = cb.lower_corner;
    const Vector3<T>& maxp = cb.upper_corner;
    const Vector3<T> ext { maxp.x - minp.x, maxp.y - minp.y, maxp.z - minp.z };
    // Bounding-box extent along each axis.

    // Normalize into [0, 1]. If an axis is degenerate, place the coordinate at 0.
    T nx = (ext.x > T(0)) ? (p.x - minp.x) / ext.x : T(0);
    T ny = (ext.y > T(0)) ? (p.y - minp.y) / ext.y : T(0);
    T nz = (ext.z > T(0)) ? (p.z - minp.z) / ext.z : T(0);

    // Clamp to the valid Morton quantization cube.
    if (nx < T(0)) nx = T(0);
    if (nx > T(1)) nx = T(1);
    if (ny < T(0)) ny = T(0);
    if (ny > T(1)) ny = T(1);
    if (nz < T(0)) nz = T(0);
    if (nz > T(1)) nz = T(1);

    // Maximum quantized integer coordinate representable with `bits` bits.
    const unsigned maxq = (1u << bits) - 1u;

    // Round normalized coordinates to the nearest quantized integer.
    const auto ix = static_cast<unsigned>(nx * maxq + T(0.5));
    const auto iy = static_cast<unsigned>(ny * maxq + T(0.5));
    const auto iz = static_cast<unsigned>(nz * maxq + T(0.5));

    // Spread bits so they can be interleaved.
    const unsigned xx = expand_bits(ix);
    const unsigned yy = expand_bits(iy);
    const unsigned zz = expand_bits(iz);

    // Interleave x/y/z in Morton order.
    return (xx << 2) | (yy << 1) | (zz << 0);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::delta_lcp(const HostBuffer<uint64_t>& keys, const int n, const int i, const int j) noexcept {
    // Compute the longest common prefix length between keys[i] and keys[j].
    //
    // If j is out of bounds, return -1 so callers can treat it as a sentinel
    // that always loses against any valid in-range comparison.
    if (j < 0 || j >= n) return -1;

    const uint64_t a = keys[i];
    const uint64_t b = keys[j];

    // If the full 64-bit keys are identical, treat them as having the maximum
    // possible common prefix length in this representation.
    if (a == b) return 64;

    // Otherwise measure the prefix length via leading zeros of XOR.
    return clz64(a ^ b);
}

template <typename T>
int
LinearBoundingVolumeHierachy<T>::find_split(const HostBuffer<uint32_t>& codes, const int first, const int last) noexcept {
    // Find the split point that partitions the Morton-code range [first, last]
    // into two child ranges for one internal LBVH node.
    //
    // This follows the common LBVH strategy:
    // - determine the common prefix of the endpoints
    // - binary search for the last index that shares a strictly longer prefix
    //   with the first code

    const uint32_t first_code = codes[first];
    const uint32_t last_code  = codes[last];

    // If all codes in the range collapse to the same Morton value,
    // split the range in the middle.
    if (first_code == last_code) return (first + last) >> 1;

    // Prefix length shared by the whole range.
    const int common_prefix = clz32(first_code ^ last_code);

    int split = first;
    int step  = last - first;

    do {
        // Binary-search-like stepping toward the rightmost valid split.
        step                = (step + 1) >> 1;
        const int new_split = split + step;

        if (new_split < last) {
            const int split_prefix = clz32(first_code ^ codes[new_split]);

            // Keep moving right while the candidate still belongs to the
            // left subgroup with a longer prefix.
            if (split_prefix > common_prefix) split = new_split;
        }
    } while (step > 1);

    return split;
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::build(const HostBuffer<TriangleContainer4<T>>& triangles) {
    // Build a linear BVH from a host-side triangle array.
    //
    // High-level pipeline:
    // 1. Reset any previous BVH state.
    // 2. Compute primitive bounds and centroids.
    // 3. Compute Morton codes of centroids.
    // 4. Stable-sort primitives by Morton code.
    // 5. Create leaf nodes.
    // 6. Build internal topology using LBVH prefix rules.
    // 7. Refit internal bounds bottom-up.
    // 8. Upload the final flat data to device buffers.

    const int n = static_cast<int>(triangles.size());
    // Number of input primitives.

    reset();
    // Discard any previous tree/caches before rebuilding.

    if (n <= 0) return;
    // Empty input produces an empty BVH.

    h_prim_bounds.resize(n);
    h_centroids.resize(n);
    h_indices.resize(n);
    // Allocate host-side primitive metadata arrays.

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            geometry::TriangleGeometryOperator<T> tri_op;
            // Temporary triangle query operator used to derive bound and centroid.

            tri_op.a = &triangles[i].a();
            tri_op.b = &triangles[i].b();
            tri_op.c = &triangles[i].c();
            tri_op.n = &triangles[i].d();
            // Bind the operator to the i-th packed triangle.

            const AABB<T> b = tri_op.bound();
            // Primitive axis-aligned bounding box.

            h_prim_bounds[i] = b;
            h_centroids[i]   = tri_op.centroid();
            // Cache primitive spatial metadata used for Morton sorting.

            h_indices[i] = i;
            // Initial primitive ordering is identity.
        });

    AABB<T> centroid_bounds;
    // Bounds over primitive centroids, used to normalize Morton coordinates.

    for (int i = 0; i < n; ++i) {
        centroid_bounds.merge(h_centroids[i]);
    }

    HostBuffer<uint32_t> morton(n);
    // Raw Morton code per primitive.

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
            // Quantize centroid i into a Morton code.
        });

    HostBuffer<int> order(n);
    // Permutation array used to sort primitives by Morton code.
    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }

    std::stable_sort(order.begin(),
                     order.end(),
                     [&](const int a, const int b) {
                         // Primary key  : Morton code
                         // Secondary key: original primitive index
                         //
                         // The secondary key guarantees deterministic ordering
                         // among equal Morton codes.
                         if (morton[a] != morton[b]) return morton[a] < morton[b];

                         return a < b;
                     });

    HostBuffer<uint32_t> morton_sorted(n);
    // Sorted Morton codes.

    HostBuffer<uint64_t> keys_sorted(n);
    // Full 64-bit sort keys:
    // - high 32 bits: Morton code
    // - low  32 bits: sorted position to ensure uniqueness

    HostBuffer<int> indices_sorted(n);
    // Primitive indices reordered into Morton order.

    for (int i = 0; i < n; ++i) {
        const int oi = order[i];
        // Original primitive index now placed at sorted slot i.

        morton_sorted[i] = morton[oi];

        keys_sorted[i] = (static_cast<uint64_t>(morton_sorted[i]) << 32)
            | static_cast<uint32_t>(i);
        // Promote Morton code to a unique 64-bit key so duplicate Morton codes
        // still have a strict ordering in prefix comparisons.

        indices_sorted[i] = h_indices[oi];
    }

    h_indices = indices_sorted;
    // Replace the primitive index array with Morton order.

    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());
    // Allocate the full flat node array:
    // - n leaves
    // - n - 1 internal nodes
    // => 2n - 1 total nodes

    for (int k = 0; k < n; ++k) {
        const int ni  = leaf_node_index(k, n);
        const int pid = h_indices[k];
        // ni  : node-array slot of leaf k
        // pid : primitive index referenced by this leaf

        BVHNode<T>& leaf = h_nodes[ni];

        leaf.is_leaf = true;
        // Mark node as a leaf.

        leaf.left = leaf.right = -1;
        // Leaves have no children.

        leaf.start = k;
        leaf.count = 1;
        // Leaf covers exactly one primitive stored at sorted position k.

        leaf.bounds = h_prim_bounds[pid];
        // Leaf bounds are exactly the primitive bounds.
    }

    if (n == 1) {
        // Degenerate one-primitive case:
        // the only leaf is also the root.
        _root = leaf_node_index(0, n);

        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;
        return;
    }

    for (int i = 0; i < n - 1; ++i) {
        // Construct internal node i using the LBVH range-direction logic.

        const int dl = delta_lcp(keys_sorted, n, i, i - 1);
        // Longest common prefix with left neighbor.

        const int dr = delta_lcp(keys_sorted, n, i, i + 1);
        // Longest common prefix with right neighbor.

        const int d = (dr > dl) ? 1 : -1;
        // Range growth direction:
        // +1 if the right side shares a longer prefix
        // -1 otherwise

        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);
        // Minimum prefix threshold that defines the containing range.

        int lmax = 2;
        // Exponential search upper bound for the range length.

        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) {
            lmax <<= 1;
        }

        int t    = 0;
        int step = lmax;

        do {
            // Binary search within the discovered range extent.
            step = (step + 1) >> 1;
            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) {
                t += step;
            }
        } while (step > 1);

        const int j = i + t * d;
        // Opposite end of the full range represented by internal node i.

        const int first = std::min(i, j);
        const int last  = std::max(i, j);
        // Normalize range orientation to ascending order.

        const int split = find_split(morton_sorted, first, last);
        // Split point that separates the left and right child ranges.

        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;
        // Detect whether each child is a single primitive leaf range.

        const int left_child  = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);
        // Map child ranges either to:
        // - a leaf node slot, or
        // - an internal node slot

        BVHNode<T>& in = h_nodes[i];
        in.is_leaf     = false;
        // Mark node i as an internal node.

        in.left  = left_child;
        in.right = right_child;
        // Store child links.

        in.start = -1;
        in.count = 0;
        // Internal nodes do not directly own primitive spans in this layout.
    }

    for (int i = n - 2; i >= 0; --i) {
        // Bottom-up refit:
        // combine child AABBs to compute each internal node bound.
        BVHNode<T>& in      = h_nodes[i];
        const BVHNode<T>& L = h_nodes[in.left];
        const BVHNode<T>& R = h_nodes[in.right];

        in.bounds = L.bounds;
        in.bounds.merge(R.bounds);
    }

    _root = 0;
    // In this flat LBVH layout, the topmost internal node is stored at index 0.

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
    // Upload the final flat BVH and primitive data to device buffers.
}

template <typename T>
void
LinearBoundingVolumeHierachy<T>::reset() {
    // Clear all host-side BVH construction data.
    h_nodes.clear();
    h_indices.clear();
    h_centroids.clear();
    h_prim_bounds.clear();

    // Clear all device-side cached data.
    d_nodes.clear();
    d_indices.clear();
    d_triangles.clear();

    // Mark the tree as invalid / empty.
    _root = -1;
}

} // namespace atlas::spatial