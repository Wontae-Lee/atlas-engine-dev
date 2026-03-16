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
    // A full binary BVH built over n primitives has:
    // - n leaf nodes   (one leaf per primitive)
    // - n - 1 internal nodes
    //
    // Therefore the total node count is:
    //   2 * n - 1
    //
    // We keep `n` as int because the build logic below uses signed neighbor/range
    // arithmetic (i-1, i+1, direction = ±1, etc.), which is much easier and safer
    // to express with signed integers than with std::size_t.
    const int n = static_cast<int>(triangles.size());

    // Clear any previous BVH state before starting a fresh build.
    //
    // This ensures:
    // - stale host/device buffers do not survive between builds
    // - root index is reset consistently
    // - partial old topology cannot leak into the new structure
    reset();

    // Empty input => empty BVH.
    //
    // There is nothing to build, so leave the object in the reset state.
    if (n <= 0) return;

    // Allocate per-primitive temporary arrays on the host:
    //
    // h_prim_bounds[i]:
    //   AABB of primitive i in the ORIGINAL input order
    //
    // h_centroids[i]:
    //   centroid of primitive i in the ORIGINAL input order
    //
    // h_indices[i]:
    //   primitive id mapping. Initially identity (i -> i), later replaced by a
    //   Morton-sorted permutation back to the original primitive array.
    h_prim_bounds.resize(n);
    h_centroids.resize(n);
    h_indices.resize(n);

    // ------------------------------------------------------------------
    // Step 1: Precompute primitive bounds and centroids
    // ------------------------------------------------------------------
    //
    // Each TriangleContainer4 stores:
    // - a(), b(), c() : triangle vertices
    // - d()           : stored normal
    //
    // We wrap each primitive in TriangleQueryOperator so we can reuse:
    // - bound()    for primitive AABB
    // - centroid() for Morton mapping
    //
    // Important:
    // - At this stage, everything is still in the ORIGINAL input order.
    // - h_indices[i] = i records that identity mapping explicitly.
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            geometry::TriangleQueryOperator<T> tri_op;

            // TriangleQueryOperator expects raw addresses to triangle data.
            // TriangleContainer4 exposes those as references through a()/b()/c()/d().
            tri_op.a = &triangles[i].a();
            tri_op.b = &triangles[i].b();
            tri_op.c = &triangles[i].c();
            tri_op.n = &triangles[i].d();

            // Compute a tight AABB for this single triangle.
            const AABB<T> b = tri_op.bound();

            // Store all primitive-local information in original order.
            h_prim_bounds[i] = b;
            h_centroids[i]   = tri_op.centroid();

            // Initially the primitive id is just the original array index.
            h_indices[i] = i;
        });

    // ------------------------------------------------------------------
    // Step 2: Compute global centroid bounds
    // ------------------------------------------------------------------
    //
    // Morton encoding needs a scene-wide coordinate frame so that all primitive
    // centroids can be mapped into the same normalized unit cube.
    //
    // centroid_bounds encloses all primitive centroids and defines that affine map.
    AABB<T> centroid_bounds;
    for (int i = 0; i < n; ++i) {
        centroid_bounds.merge(h_centroids[i]);
    }

    // One Morton code per primitive.
    HostBuffer<uint32_t> morton(n);

    // ------------------------------------------------------------------
    // Step 3: Encode centroids into Morton codes
    // ------------------------------------------------------------------
    //
    // morton3():
    // - normalizes centroid against centroid_bounds
    // - quantizes coordinates
    // - interleaves x/y/z bits into a 3D Morton code
    //
    // Nearby centroids in space tend to get nearby Morton keys, which is the
    // core approximation used by LBVH to cluster spatially close primitives.
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
        });

    // ------------------------------------------------------------------
    // Step 4: Sort primitive references by Morton code
    // ------------------------------------------------------------------
    //
    // We do not sort the primitive data buffers directly here.
    // Instead we sort an auxiliary `order` array containing original indices.
    //
    // order[k] = original primitive index occupying sorted position k.
    HostBuffer<int> order(n);
    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }

    std::stable_sort(order.begin(),
                     order.end(),
                     [&](const int a, const int b) {
                         // Primary key: Morton code
                         //
                         // This is the spatial ordering criterion.
                         if (morton[a] != morton[b]) return morton[a] < morton[b];

                         // Secondary key: original primitive order
                         //
                         // Stable, deterministic tie-breaking matters because many
                         // primitives can quantize to the same Morton code.
                         // Without this, equal-code ordering could vary across
                         // platforms / standard library implementations.
                         return a < b;
                     });

    // These arrays are the sorted views used by the radix-tree construction.
    HostBuffer<uint32_t> morton_sorted(n);
    HostBuffer<uint64_t> keys_sorted(n);
    HostBuffer<int> indices_sorted(n);

    // ------------------------------------------------------------------
    // Step 5: Materialize sorted Morton stream and strict-order keys
    // ------------------------------------------------------------------
    //
    // morton_sorted[k]:
    //   Morton code at sorted position k
    //
    // keys_sorted[k]:
    //   64-bit key used for LCP/radix logic
    //   high 32 bits = Morton code
    //   low  32 bits = sorted position k
    //
    // Why add the sorted position?
    // - Duplicate Morton codes are common after quantization.
    // - The low bits inject a unique tie-breaker, giving a strict total order.
    // - This is crucial for well-defined longest-common-prefix comparisons.
    //
    // indices_sorted[k]:
    //   original primitive id for sorted slot k
    for (int i = 0; i < n; ++i) {
        const int oi = order[i];

        // Store contiguous sorted Morton stream.
        morton_sorted[i] = morton[oi];

        // Build strict-order radix key:
        // - spatial locality lives in upper bits
        // - uniqueness / determinism lives in lower bits
        keys_sorted[i] = (static_cast<uint64_t>(morton_sorted[i]) << 32)
            | static_cast<uint32_t>(i);

        // Map sorted slot i back to original primitive id.
        indices_sorted[i] = h_indices[oi];
    }

    // Replace h_indices with the Morton-sorted primitive permutation.
    //
    // From this point on:
    // - sorted slot k corresponds to original primitive id h_indices[k]
    h_indices = indices_sorted;

    // ------------------------------------------------------------------
    // Step 6: Allocate node array
    // ------------------------------------------------------------------
    //
    // Node layout convention:
    // - internal nodes occupy indices [0, n - 2]
    // - leaf nodes occupy     indices [n - 1, 2n - 2]
    //
    // For safety, std::max(1, 2*n - 1) keeps at least one slot allocated even
    // if some edge case slips through, though n > 0 here already.
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode<T>());

    // ------------------------------------------------------------------
    // Step 7: Initialize leaves
    // ------------------------------------------------------------------
    //
    // There is exactly one leaf per primitive in Morton-sorted order.
    //
    // Leaf k represents sorted primitive slot k, which maps back to original
    // primitive id `pid = h_indices[k]`.
    for (int k = 0; k < n; ++k) {
        const int ni  = leaf_node_index(k, n);
        const int pid = h_indices[k];

        BVHNode<T>& leaf = h_nodes[ni];

        leaf.is_leaf = true;

        // Leaves do not have children.
        leaf.left = leaf.right = -1;

        // A leaf covers exactly one sorted primitive slot.
        leaf.start = k;
        leaf.count = 1;

        // Its bounds come from the ORIGINAL primitive selected by pid.
        //
        // This distinction matters:
        // - leaf position in the array is Morton-sorted
        // - primitive geometry is still stored in original primitive buffers
        leaf.bounds = h_prim_bounds[pid];
    }

    // ------------------------------------------------------------------
    // Step 8: Handle single-primitive special case
    // ------------------------------------------------------------------
    //
    // With one primitive:
    // - there are no internal nodes
    // - the root is that one leaf
    if (n == 1) {
        _root = leaf_node_index(0, n);

        // Mirror host-side state to device buffers used by traversal.
        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;
        return;
    }

    // ------------------------------------------------------------------
    // Step 9: Build internal node topology (Karras LBVH radix tree)
    // ------------------------------------------------------------------
    //
    // Each internal node i in [0, n-2] owns one maximal interval [first, last]
    // of the Morton-sorted primitive stream.
    //
    // Construction idea:
    // 1) Compare LCP (longest common prefix) with left and right neighbors
    // 2) Decide growth direction d = ±1
    // 3) Find maximal range sharing a longer prefix than delta_min
    // 4) Split that range into left/right child subranges
    for (int i = 0; i < n - 1; ++i) {
        // LCP of node i with left neighbor.
        const int dl = delta_lcp(keys_sorted, n, i, i - 1);

        // LCP of node i with right neighbor.
        const int dr = delta_lcp(keys_sorted, n, i, i + 1);

        // Choose the direction toward the neighbor with the larger LCP.
        //
        // Intuition:
        // - That side shares more Morton-prefix bits with i
        // - Therefore it belongs to the same radix-tree branch
        const int d = (dr > dl) ? 1 : -1;

        // Minimum prefix length that defines the boundary of i's owned range.
        //
        // Anything inside the range must share MORE than this prefix length.
        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);

        int lmax = 2;

        // Exponential search:
        // Grow outward along direction d until the prefix condition fails.
        //
        // This cheaply brackets the maximal interval size.
        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) {
            lmax <<= 1;
        }

        int t    = 0;
        int step = lmax;

        // Binary refinement:
        // Shrink the bracket to find the exact endpoint of the maximal interval.
        do {
            step = (step + 1) >> 1;
            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) {
                t += step;
            }
        } while (step > 1);

        // j is the other endpoint of the maximal interval owned by internal node i.
        const int j = i + t * d;

        // Normalize interval ordering so first <= last regardless of direction.
        const int first = std::min(i, j);
        const int last  = std::max(i, j);

        // Choose where to split [first, last] into left and right child ranges.
        //
        // find_split() typically finds the highest position where the common prefix
        // changes in a way that best matches the radix-tree subdivision.
        const int split = find_split(morton_sorted, first, last);

        // Child interval length 1 => child is a leaf.
        // Otherwise child is another internal node.
        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;

        const int left_child  = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);

        BVHNode<T>& in = h_nodes[i];
        in.is_leaf     = false;

        // Children are stored by direct node indices into h_nodes.
        //
        // Depending on interval length:
        // - left/right may point to internal nodes [0, n-2]
        // - or to leaves                     [n-1, 2n-2]
        in.left  = left_child;
        in.right = right_child;

        // Internal nodes do not directly own primitive ranges in leaf terms here.
        in.start = -1;
        in.count = 0;
    }

    // ------------------------------------------------------------------
    // Step 10: Bottom-up bound propagation
    // ------------------------------------------------------------------
    //
    // Internal node topology is now fixed.
    // Next, compute each internal node's AABB as the union of its two children.
    //
    // Because of the chosen node layout, iterating internal nodes backwards from
    // n-2 down to 0 guarantees child bounds are already initialized.
    for (int i = n - 2; i >= 0; --i) {
        BVHNode<T>& in      = h_nodes[i];
        const BVHNode<T>& L = h_nodes[in.left];
        const BVHNode<T>& R = h_nodes[in.right];

        // Start from left child box and merge right child box into it.
        in.bounds = L.bounds;
        in.bounds.merge(R.bounds);
    }

    // ------------------------------------------------------------------
    // Step 11: Finalize root and mirror to device
    // ------------------------------------------------------------------
    //
    // In this LBVH layout, internal node 0 spans the full sorted primitive range
    // and therefore acts as the root.
    _root = 0;

    // Copy finalized host-side BVH data to device buffers used during traversal.
    //
    // d_nodes:
    //   final BVH topology + bounds
    //
    // d_indices:
    //   Morton-sorted mapping from leaf slot -> original primitive id
    //
    // d_triangles:
    //   original primitive storage, still in original order
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
