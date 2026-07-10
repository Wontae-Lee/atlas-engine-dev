#include <atlas/geometry/geometry.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/random/seed.h>
#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>

#include <algorithm>

namespace atlas {

BvhView
LBVH::view() const {
    BvhView view {};

    // Strip the device_vector wrappers down to raw device pointers so the view
    // is trivially copyable and a device lambda can capture it by value.
    view.bvh_nodes   = atlas::raw_pointer_cast(d_nodes.data());
    view.bvh_indices = atlas::raw_pointer_cast(d_indices.data());
    view.bvh_tris    = atlas::raw_pointer_cast(d_triangles.data());
    view.bvh_root    = _root;

    return view;
}

void
LBVH::assign_solid_angle_moment(
    BVHNode& node,
    const TriangleContainer4& triangle) noexcept {
    const Float3& a = triangle.a();
    const Float3& b = triangle.b();
    const Float3& c = triangle.c();

    const Float3 normal_area = atlas::cross(b - a, c - a) * 0.5f;
    const float area         = normal_area.length();

    node.solid_angle_moment      = Float3(0.0f, 0.0f, 0.0f);
    node.solid_angle_normal_area = Float3(0.0f, 0.0f, 0.0f);
    node.solid_angle_area        = 0.0f;

    // Degenerate (zero-area) triangle contributes nothing; the NaN-safe form
    // !(area > 0) also rejects a NaN area.
    if (!(area > 0.0f)) {
        return;
    }

    // Area-weighted centroid uses the triangle centroid (a+b+c)/3 scaled by area.
    node.solid_angle_moment      = (a + b + c) * (area / 3.0f);
    node.solid_angle_normal_area = normal_area;
    node.solid_angle_area        = area;
}

void
LBVH::merge_solid_angle_moment(
    BVHNode& node,
    const BVHNode& left,
    const BVHNode& right) noexcept {
    node.solid_angle_moment      = left.solid_angle_moment + right.solid_angle_moment;
    node.solid_angle_normal_area = left.solid_angle_normal_area + right.solid_angle_normal_area;
    node.solid_angle_area        = left.solid_angle_area + right.solid_angle_area;
}

int
LBVH::leaf_node_index(const int k, const int n) noexcept {

    return (n - 1) + k;
}

unsigned
LBVH::expand_bits(unsigned v) noexcept {

    v = (v * atlas::MORTON_EXPAND_BITS_FIRST_MULTIPLIER)
        & atlas::MORTON_EXPAND_BITS_FIRST_MASK;
    v = (v * atlas::MORTON_EXPAND_BITS_SECOND_MULTIPLIER)
        & atlas::MORTON_EXPAND_BITS_SECOND_MASK;
    v = (v * atlas::MORTON_EXPAND_BITS_THIRD_MULTIPLIER)
        & atlas::MORTON_EXPAND_BITS_THIRD_MASK;
    v = (v * atlas::MORTON_EXPAND_BITS_FINAL_MULTIPLIER)
        & atlas::MORTON_EXPAND_BITS_FINAL_MASK;

    return v;
}

int
LBVH::clz32(const uint32_t x) noexcept {

    if (x == 0u) return 32;

    int n      = 0;
    uint32_t m = 1u << 31;

    while ((x & m) == 0u) {
        ++n;
        m >>= 1;
    }

    return n;
}

int
LBVH::clz64(const uint64_t x) noexcept {

    if (x == 0u) return 64;

    int n      = 0;
    uint64_t m = 1ull << 63;

    while ((x & m) == 0u) {
        ++n;
        m >>= 1;
    }

    return n;
}

uint32_t
LBVH::morton3(
    const Float3& p,
    const AABB& cb,
    const int bits) const noexcept {
    const Float3& minp = cb.lower_corner;
    const Float3& maxp = cb.upper_corner;

    const Float3 ext(
        maxp.x - minp.x,
        maxp.y - minp.y,
        maxp.z - minp.z);

    float nx = (ext.x > 0.0f) ? (p.x - minp.x) / ext.x : 0.0f;
    float ny = (ext.y > 0.0f) ? (p.y - minp.y) / ext.y : 0.0f;
    float nz = (ext.z > 0.0f) ? (p.z - minp.z) / ext.z : 0.0f;

    if (nx < 0.0f) nx = 0.0f;
    if (nx > 1.0f) nx = 1.0f;
    if (ny < 0.0f) ny = 0.0f;
    if (ny > 1.0f) ny = 1.0f;
    if (nz < 0.0f) nz = 0.0f;
    if (nz > 1.0f) nz = 1.0f;

    const unsigned maxq = (1u << bits) - 1u;
    const auto ix       = static_cast<unsigned>(nx * maxq + 0.5f);
    const auto iy       = static_cast<unsigned>(ny * maxq + 0.5f);
    const auto iz       = static_cast<unsigned>(nz * maxq + 0.5f);

    const unsigned xx = expand_bits(ix);
    const unsigned yy = expand_bits(iy);
    const unsigned zz = expand_bits(iz);

    return (xx << 2) | (yy << 1) | (zz << 0);
}

// Longest-common-prefix length between two sorted leaves' *combined* keys
// (see the keys_sorted packing below), via the leading-zero-count-of-XOR
// trick (equal keys share all bits, XOR is 0, clz64 saturates to 64 —
// treated as the maximum possible LCP). j out of [0, n) returns -1, an
// LCP shorter than any real pair, so range searches naturally treat an
// out-of-range neighbor as "definitely not part of this node's range."
int
LBVH::delta_lcp(
    const HostBuffer<uint64_t>& keys,
    const int n,
    const int i,
    const int j) noexcept {

    if (j < 0 || j >= n) return -1;

    const uint64_t a = keys[i];
    const uint64_t b = keys[j];

    if (a == b) return 64;

    return clz64(a ^ b);
}

// Binary search for the LCP boundary within [first, last]: the split point
// where the common prefix shared by codes[first] and everything up to
// `split` is *longer* than the prefix shared by codes[first] and
// codes[last] as a whole — i.e. the point where the Morton-code-implied
// spatial hierarchy actually diverges. Exponentially-decreasing `step`
// makes this O(log(last-first)) instead of an O(last-first) linear scan.
int
LBVH::find_split(
    const HostBuffer<uint32_t>& codes,
    const int first,
    const int last) noexcept {
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

void
LBVH::build(const HostBuffer<TriangleContainer4>& triangles) {
    const int n = static_cast<int>(triangles.size());

    reset();

    if (n <= 0) return;

    h_prim_bounds.resize(n);
    h_centroids.resize(n);
    h_indices.resize(n);

    // Phase 1: per-primitive bounds and centroids (independent across i).
    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &triangles](int i) {
            Triangle tri_op;

            tri_op.a = triangles[i].a();
            tri_op.b = triangles[i].b();
            tri_op.c = triangles[i].c();
            tri_op.n = triangles[i].d();

            const AABB b     = tri_op.bound();
            h_prim_bounds[i] = b;
            h_centroids[i]   = tri_op.centroid();
            h_indices[i]     = i;
        });

    // Morton codes are quantized against the bound of all centroids, so the
    // full [0, 1] code range is used regardless of the scene's world extent.
    AABB centroid_bounds;

    for (int i = 0; i < n; ++i) {
        centroid_bounds.merge(h_centroids[i]);
    }

    HostBuffer<uint32_t> morton(n);

    atlas::parallel_for<ExecutionPolicy::host>(
        0,
        n,
        [this, &morton, &centroid_bounds](int i) {
            morton[i] = morton3(h_centroids[i], centroid_bounds, _morton_bits);
        });

    // Sort primitives by Morton code (original index as a stable tiebreak) so
    // spatially near triangles become array-adjacent for the Karras build.
    HostBuffer<int> order(n);

    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }

    std::stable_sort(
        order.begin(),
        order.end(),
        [&](const int a, const int b) {
            if (morton[a] != morton[b]) return morton[a] < morton[b];

            return a < b;
        });

    HostBuffer<uint32_t> morton_sorted(n);
    HostBuffer<uint64_t> keys_sorted(n);
    HostBuffer<int> indices_sorted(n);

    for (int i = 0; i < n; ++i) {
        const int oi = order[i];

        morton_sorted[i] = morton[oi];
        // Pack (morton_code, sorted_index) into one 64-bit key: two
        // triangles with identical Morton codes (a real possibility at
        // finite quantization) would otherwise make delta_lcp return the
        // same LCP for every pair among them, breaking Karras' algorithm's
        // assumption that distinct leaves have distinct keys. Appending the
        // index as a tiebreaker guarantees strictly increasing keys along
        // the sorted array, at the cost of only affecting the LCP once the
        // full 32-bit Morton code already matches exactly.
        keys_sorted[i]   = (static_cast<uint64_t>(morton_sorted[i]) << 32)
            | static_cast<uint32_t>(i);

        indices_sorted[i] = h_indices[oi];
    }

    h_indices = indices_sorted;

    // Exactly 2n-1 nodes: n leaves plus n-1 internal nodes (max(1, ...) keeps a
    // single-primitive tree from requesting a zero-length allocation).
    h_nodes.resize(std::max(1, 2 * n - 1), BVHNode());

    // Initialize every leaf node from its sorted primitive before the internal
    // nodes are wired up: leaf k lives at node index (n-1)+k.
    for (int k = 0; k < n; ++k) {
        const int ni  = leaf_node_index(k, n);
        const int pid = h_indices[k];
        BVHNode& leaf = h_nodes[ni];

        leaf.is_leaf = true;
        leaf.left = leaf.right = -1;
        leaf.start             = k;
        leaf.count             = 1;
        leaf.bounds            = h_prim_bounds[pid];
        assign_solid_angle_moment(leaf, triangles[pid]);
    }

    // Single primitive: there are no internal nodes to build, so the lone leaf
    // is the root and the Karras loops below are skipped entirely.
    if (n == 1) {

        _root       = leaf_node_index(0, n);
        d_nodes     = h_nodes;
        d_indices   = h_indices;
        d_triangles = triangles;

        return;
    }

    // Parent links, filled as the ranges are derived. The refit below walks upward from the
    // leaves and cannot be replaced by a descending index sweep: an internal node whose range
    // *ends* at its own index (direction d == -1) owns a child stored at a smaller index, so
    // a parent can precede its child in the array.
    HostBuffer<int> parent(static_cast<std::size_t>(2 * n - 1), -1);

    // Karras' (2012) core insight: internal node i's owned leaf range can
    // be computed independently of every other internal node — no
    // recursion, no dependency chain — purely from index i and the global
    // sorted-key array. This loop derives that range for every internal
    // node in parallel-friendly fashion (this .cu is host-only, but the
    // computation has no cross-iteration dependency and would parallelize
    // directly on device).
    for (int i = 0; i < n - 1; ++i) {

        // Direction d: whether node i's range extends toward increasing or
        // decreasing index, chosen by whichever neighbor shares a longer
        // prefix with i (the range should grow toward the "more similar"
        // side).
        const int dl = delta_lcp(keys_sorted, n, i, i - 1);
        const int dr = delta_lcp(keys_sorted, n, i, i + 1);
        const int d  = (dr > dl) ? 1 : -1;

        // delta_min: the LCP length any leaf must have with i to belong to
        // i's range at all — anything sharing a shorter prefix than the
        // "wrong" (non-d) direction neighbor is outside this range.
        const int delta_min = delta_lcp(keys_sorted, n, i, i - d);

        // Exponential (doubling) search for the range's far end: keep
        // doubling lmax until stepping that far in direction d would leave
        // the required delta_min prefix — O(log(range length)) instead of
        // a linear scan to find the range boundary.
        int lmax = 2;

        while (delta_lcp(keys_sorted, n, i, i + lmax * d) > delta_min) {
            lmax <<= 1;
        }

        // Binary search within [0, lmax] to pin down the exact range
        // extent t (same halving-step pattern as find_split above).
        int t    = 0;
        int step = lmax;

        do {
            step = (step + 1) >> 1;

            if (delta_lcp(keys_sorted, n, i, i + (t + step) * d) > delta_min) {
                t += step;
            }
        } while (step > 1);

        const int j     = i + t * d;
        const int first = std::min(i, j);
        const int last  = std::max(i, j);

        // Within this node's now-known [first, last] leaf range, find_split
        // locates where the subtree actually divides into left/right
        // children (a second, independent LCP-boundary search, this time
        // over the range instead of finding the range itself).
        const int split = find_split(morton_sorted, first, last);

        const int left_is_leaf  = (split == first) ? 1 : 0;
        const int right_is_leaf = (split + 1 == last) ? 1 : 0;

        const int left_child  = left_is_leaf ? leaf_node_index(split, n) : split;
        const int right_child = right_is_leaf ? leaf_node_index(split + 1, n) : (split + 1);

        BVHNode& in = h_nodes[i];

        in.is_leaf = false;
        in.left    = left_child;
        in.right   = right_child;
        in.start   = -1;
        in.count   = 0;

        parent[left_child]  = i;
        parent[right_child] = i;
    }

    // Bottom-up refit of internal-node bounds and solid-angle moments. Leaves (indices
    // >= n-1) were finalized above; each internal node is refitted by whichever of its two
    // children reaches it second, which guarantees both children are final. The counter is a
    // plain increment here because the build is host-serial; on device it becomes an atomic
    // and the same walk parallelizes over the leaves.
    HostBuffer<int> visits(static_cast<std::size_t>(n - 1), 0);

    for (int k = 0; k < n; ++k) {
        int node = parent[leaf_node_index(k, n)];

        while (node != -1 && ++visits[node] == 2) {
            BVHNode& in      = h_nodes[node];
            const BVHNode& L = h_nodes[in.left];
            const BVHNode& R = h_nodes[in.right];

            in.bounds = L.bounds;
            in.bounds.merge(R.bounds);
            merge_solid_angle_moment(in, L, R);

            node = parent[node];
        }
    }

    // Karras' layout always roots the tree at internal node 0.
    _root = 0;

    d_nodes     = h_nodes;
    d_indices   = h_indices;
    d_triangles = triangles;
}

void
LBVH::reset() {

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
