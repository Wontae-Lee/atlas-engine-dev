#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <cstdint>

namespace atlas {

/**
 * @brief Linear BVH built by Karras' parallel Morton-code algorithm.
 *
 * Sorts primitives by the Morton code of their centroids so spatially near
 * triangles become array-adjacent, then derives each internal node's leaf range
 * directly from the sorted keys — with no recursion or inter-node dependency —
 * following Karras (2012). The tree has exactly `2n - 1` nodes for `n`
 * primitives: leaves occupy indices `[n-1, 2n-1)` and internal nodes `[0, n-1)`,
 * with node 0 as the root. The whole build here runs on the host, but the
 * per-node range derivation is deliberately free of cross-iteration
 * dependencies so it maps directly onto a device parallel pass.
 *
 * This produces a hierarchy fast and cheaply; it does not minimize a surface-
 * area-heuristic cost the way @ref SAHBVH does, trading build-time SAH quality
 * for speed — in line with the engine's easy/fast, large-domain priorities.
 *
 * @note Leaf capacity is not configurable here. Karras' layout stores exactly one
 *       primitive per leaf: the `2n - 1` node count, the `(n - 1) + k` leaf indexing,
 *       and the delta metric's assumption that each leaf owns one distinct sorted key
 *       all depend on it. Packing several primitives into a leaf requires a separate
 *       subtree-collapse pass over the built tree, which this class does not perform.
 *       @ref SAHBVH, whose top-down build recurses until a leaf is small enough,
 *       exposes a `leaf_size` instead.
 */
class LBVH final : public BVH {
public:
    /// Construct an empty hierarchy; call @ref build before querying.
    LBVH() = default;

    /// Destructor (defaulted); releases host and device buffers.
    ~LBVH() override = default;

    /**
     * @brief Build the linear BVH from a triangle soup and upload it (host).
     *
     * Computes per-triangle bounds and centroids, Morton-sorts them, constructs
     * the Karras tree, refits bounds and solid-angle moments bottom-up, and
     * copies nodes/indices/triangles to the device. Any prior state is cleared
     * first; an empty input leaves the hierarchy empty (root = -1).
     *
     * @param triangles Triangles to index.
     */
    ATLAS_HOST void
    build(const HostBuffer<TriangleContainer4>& triangles) override;

    /**
     * @brief Gather the built device buffers into a capturable view.
     * @return A @ref BvhView of raw device pointers and the root index.
     */
    ATLAS_HOST BvhView
    view() const override;

    /**
     * @brief Clear all host and device buffers and mark the hierarchy empty.
     *
     * Called at the start of @ref build; leaves @ref root at -1.
     */
    ATLAS_HOST void
    reset();

    /**
     * @brief Set the per-axis Morton quantization resolution, clamped to [1, 10].
     *
     * Ten bits per axis is the ceiling because three 10-bit axes pack into the
     * 30 usable bits of a 32-bit Morton code. More bits distinguish nearby
     * centroids more finely at no extra storage cost.
     *
     * @param morton_bits Desired bits per axis; clamped into [1, 10].
     */
    void
    set_morton_bits(int morton_bits) noexcept {
        if (morton_bits < 1) morton_bits = 1;
        if (morton_bits > 10) morton_bits = 10;

        _morton_bits = morton_bits;
    }

    /**
     * @brief Current Morton bits per axis.
     * @return The configured quantization resolution.
     */
    ATLAS_NODISCARD int
    morton_bits() const noexcept {
        return _morton_bits;
    }

    /**
     * @brief Index of the root node in the node array.
     * @return The root index, or -1 if nothing is built.
     */
    ATLAS_NODISCARD int
    root() const noexcept {
        return _root;
    }

    /**
     * @brief Host-side node array (`2n - 1` entries after a build).
     * @return Const reference to the host nodes.
     */
    ATLAS_NODISCARD const HostBuffer<BVHNode>&
    nodes() const noexcept {
        return h_nodes;
    }

    /**
     * @brief Host-side primitive index array, reordered into leaf-run order.
     * @return Const reference to the reordered indices.
     */
    ATLAS_NODISCARD const HostBuffer<int>&
    indices() const noexcept {
        return h_indices;
    }

    /**
     * @brief Host-side per-primitive bounds, indexed by original triangle id.
     * @return Const reference to the primitive bounds.
     */
    ATLAS_NODISCARD const HostBuffer<AABB>&
    bounds() const noexcept {
        return h_prim_bounds;
    }

    /**
     * @brief Host-side per-primitive centroids, indexed by original triangle id.
     * @return Const reference to the centroids.
     */
    ATLAS_NODISCARD const HostBuffer<Float3>&
    centroids() const noexcept {
        return h_centroids;
    }

    /**
     * @brief Device copy of the node array.
     * @return Const reference to the device nodes.
     */
    ATLAS_NODISCARD const DeviceBuffer<BVHNode>&
    device_nodes() const noexcept {
        return d_nodes;
    }

    /**
     * @brief Device copy of the reordered primitive indices.
     * @return Const reference to the device indices.
     */
    ATLAS_NODISCARD const DeviceBuffer<int>&
    device_indices() const noexcept {
        return d_indices;
    }

    /**
     * @brief Device copy of the triangle soup used for the build.
     * @return Const reference to the device triangles.
     */
    ATLAS_NODISCARD const DeviceBuffer<TriangleContainer4>&
    device_triangles() const noexcept {
        return d_triangles;
    }

private:
    /// Host node array; leaves at [n-1, 2n-1), internal nodes at [0, n-1).
    HostBuffer<BVHNode> h_nodes;

    /// Primitive ids reordered so each leaf covers a contiguous run.
    HostBuffer<int> h_indices;

    /// Per-primitive AABBs, indexed by original triangle id.
    HostBuffer<AABB> h_prim_bounds;

    /// Per-primitive centroids, indexed by original triangle id.
    HostBuffer<Float3> h_centroids;

    /// Root node index, -1 until a successful build.
    int _root = -1;

    /// Morton quantization bits per axis (in [1, 10]).
    int _morton_bits = 10;

    /// Device copy of @ref h_nodes.
    DeviceBuffer<BVHNode> d_nodes;

    /// Device copy of @ref h_indices.
    DeviceBuffer<int> d_indices;

    /// Device copy of the source triangles.
    DeviceBuffer<TriangleContainer4> d_triangles;

private:
    /**
     * @brief Seed a leaf node's solid-angle moments from a single triangle.
     *
     * Computes the triangle's area-weighted centroid, area-scaled normal, and
     * area, storing them on @p node. A degenerate (zero-area) triangle leaves
     * all three moments at zero.
     *
     * @param node     Leaf node to populate.
     * @param triangle Source triangle.
     */
    static void
    assign_solid_angle_moment(BVHNode& node,
                              const TriangleContainer4& triangle) noexcept;

    /**
     * @brief Combine two children's solid-angle moments into their parent.
     * @param node  Parent node to fill.
     * @param left  Left child.
     * @param right Right child.
     */
    static void
    merge_solid_angle_moment(BVHNode& node,
                             const BVHNode& left,
                             const BVHNode& right) noexcept;

    /**
     * @brief Map leaf ordinal @p k to its node-array index for an `n`-leaf tree.
     * @param k Leaf ordinal in [0, n).
     * @param n Primitive count.
     * @return Node index `(n - 1) + k`, i.e. leaves live after the internal nodes.
     */
    ATLAS_NODISCARD static int
    leaf_node_index(int k, int n) noexcept;

    /**
     * @brief Spread 10 low bits of @p v so bit i lands at bit 3i (Morton dilate).
     *
     * The bit-manipulation constants come from `atlas/random/seed.h`. Interleaving
     * three dilated coordinates yields a 30-bit Morton code.
     *
     * @param v Value whose low 10 bits are dilated.
     * @return The dilated value with two zero bits inserted between each input bit.
     */
    ATLAS_NODISCARD static unsigned
    expand_bits(unsigned v) noexcept;

    /**
     * @brief Compute the Morton code of a point within a bounding box.
     *
     * Normalizes @p p into [0, 1] per axis using @p cb, quantizes to
     * `_morton_bits` per axis, and interleaves via @ref expand_bits. A zero-
     * width axis maps to 0 on that axis.
     *
     * @param p    Point to encode (a primitive centroid).
     * @param cb   Bounding box of all centroids, defining the normalization range.
     * @param bits Bits per axis (typically @ref morton_bits).
     * @return The interleaved Morton code.
     */
    ATLAS_NODISCARD uint32_t
    morton3(const Float3& p, const AABB& cb, int bits) const noexcept;

    /**
     * @brief Count leading zero bits of a 32-bit value (32 for zero).
     * @param x Value to inspect.
     * @return Number of leading zero bits.
     */
    ATLAS_NODISCARD static int
    clz32(uint32_t x) noexcept;

    /**
     * @brief Count leading zero bits of a 64-bit value (64 for zero).
     * @param x Value to inspect.
     * @return Number of leading zero bits.
     */
    ATLAS_NODISCARD static int
    clz64(uint64_t x) noexcept;

    /**
     * @brief Longest-common-prefix length between two leaves' combined keys.
     *
     * The delta metric of Karras' algorithm: `clz64(keys[i] ^ keys[j])`, with
     * equal keys saturating to 64. An out-of-range @p j returns -1 so range
     * searches treat a non-existent neighbor as "not part of this range."
     *
     * @param keys Packed (Morton, index) keys in sorted order.
     * @param n    Number of leaves.
     * @param i    First leaf index (assumed in range).
     * @param j    Second leaf index; if outside [0, n) the result is -1.
     * @return The LCP length, or -1 when @p j is out of range.
     */
    ATLAS_NODISCARD static int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    /**
     * @brief Locate the split position within an internal node's leaf range.
     *
     * Binary-searches [first, last] for the last index whose Morton code shares
     * a longer prefix with `codes[first]` than `codes[first]` shares with
     * `codes[last]` — the point where the spatial hierarchy diverges. When the
     * endpoints' codes are equal it falls back to the midpoint.
     *
     * @param codes Sorted Morton codes.
     * @param first Range start (inclusive).
     * @param last  Range end (inclusive).
     * @return The split index; the left child covers [first, split], the right
     *         child [split+1, last].
     */
    ATLAS_NODISCARD static int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};

}