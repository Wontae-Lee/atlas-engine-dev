#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <cstdint>

/**
 * @file lbvh.h
 * @brief Linear BVH: builds a hierarchy in `O(n log n)` (dominated by
 *        one sort) directly from Morton-coded triangle centroids,
 *        without the top-down recursive splitting `SAHBVH` uses —
 *        Karras' (2012) parallel-friendly construction algorithm.
 *
 * @details
 * ### Background — Morton codes and the LBVH construction
 * A Morton (Z-order) code interleaves the bits of a point's quantized
 * `(x, y, z)` coordinates into one integer such that spatially nearby
 * points tend to have numerically nearby codes — `morton3`/`expand_bits`
 * compute this by quantizing each centroid into the node's bounding box
 * (`cb`) at `bits` bits of precision per axis and interleaving
 * (`expand_bits` spreads one axis's bits out to make room for the other
 * two, a standard bit-twiddling trick). Sorting triangles by Morton code
 * (`keys_sorted`) thus produces an ordering that is already a
 * reasonable spatial hierarchy in disguise: any *contiguous range* of
 * the sorted array corresponds to a roughly spatially-coherent cluster.
 *
 * Karras' algorithm (2012) exploits this directly: for `n` sorted
 * leaves there are exactly `n-1` internal nodes, and internal node `i`
 * can be assigned, *without recursion*, to own the contiguous leaf range
 * whose boundary is where the longest-common-prefix (LCP) of adjacent
 * Morton codes drops — `delta_lcp(i, j)` measures the LCP length between
 * codes `i` and `j` (`clz32`/`clz64` count leading zero bits of the XOR,
 * the standard LCP-via-XOR trick), and each internal node `i` determines
 * its range's direction (`d`) and extent (`lmax`, found by exponential/
 * doubling search rather than a linear scan — `delta_lcp(i, i+lmax*d) >
 * delta_min` doubles `lmax` until the range no longer shares the
 * required prefix depth) and its split point (`find_split`, a binary
 * search for the LCP boundary within the range) independently of every
 * other internal node. This is what makes LBVH construction trivially
 * parallel (each node's topology is computed from its index and the
 * global sorted key array alone, no dependency on other nodes'
 * results) — the cost of that parallelism-friendliness is tree quality:
 * splits follow Morton-code boundaries rather than an
 * area/cost-optimizing criterion, so LBVH trees are typically somewhat
 * less tight than an `SAHBVH` tree built from the same primitives, in
 * exchange for a much cheaper build.
 *
 * The same winding-number moment computation described in `node.h`
 * (`assign_solid_angle_moment`/`merge_solid_angle_moment`) runs as a
 * bottom-up pass once the tree topology is fixed.
 *
 * ### References
 * - T. Karras, "Maximizing Parallelism in the Construction of BVHs,
 *   Octrees, and k-d Trees," High-Performance Graphics, 2012. (the
 *   LCP-range parallel construction algorithm this class implements)
 */

namespace atlas {

/**
 * @brief Morton-code-based linear BVH builder. See this file's
 *        top-of-file documentation for the LCP-range parallel
 *        construction algorithm.
 */
class LBVH final : public BVH {
public:
    LBVH() = default;

    ~LBVH() override = default;

    /** @brief Assigns Morton codes to `triangles`' centroids, sorts by
     *  code, and constructs the tree via Karras' LCP-range algorithm;
     *  see this file's top-of-file documentation. */
    ATLAS_HOST void
    build(const HostBuffer<TriangleContainer4>& triangles) override;

    ATLAS_HOST BvhGeometryOperator
    make_geometry_operator() const override;

    /** @brief Clears the built tree back to an empty state. */
    ATLAS_HOST void
    reset();

    /** @brief Maximum primitives per leaf (clamped to `>= 1`); smaller
     *  leaves mean deeper trees with tighter bounds but more nodes. */
    void
    set_leaf_size(const int leaf_size) noexcept {
        _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
    }

    /** @brief Quantization precision (bits per axis, clamped to
     *  `[1, 10]`) used by `morton3` when computing centroid Morton
     *  codes; more bits distinguish closer centroids at the cost of a
     *  wider integer key. */
    void
    set_morton_bits(int morton_bits) noexcept {
        if (morton_bits < 1) morton_bits = 1;
        if (morton_bits > 10) morton_bits = 10;

        _morton_bits = morton_bits;
    }

    ATLAS_NODISCARD int
    leaf_size() const noexcept {
        return _leaf_size;
    }

    ATLAS_NODISCARD int
    morton_bits() const noexcept {
        return _morton_bits;
    }

    ATLAS_NODISCARD int
    root() const noexcept {
        return _root;
    }

    ATLAS_NODISCARD const HostBuffer<BVHNode>&
    nodes() const noexcept {
        return h_nodes;
    }

    ATLAS_NODISCARD const HostBuffer<int>&
    indices() const noexcept {
        return h_indices;
    }

    ATLAS_NODISCARD const HostBuffer<AABB>&
    bounds() const noexcept {
        return h_prim_bounds;
    }

    ATLAS_NODISCARD const HostBuffer<Vector3>&
    centroids() const noexcept {
        return h_centroids;
    }

    ATLAS_NODISCARD const DeviceBuffer<BVHNode>&
    device_nodes() const noexcept {
        return d_nodes;
    }

    ATLAS_NODISCARD const DeviceBuffer<int>&
    device_indices() const noexcept {
        return d_indices;
    }

    ATLAS_NODISCARD const DeviceBuffer<TriangleContainer4>&
    device_triangles() const noexcept {
        return d_triangles;
    }

private:
    HostBuffer<BVHNode> h_nodes;

    HostBuffer<int> h_indices;

    HostBuffer<AABB> h_prim_bounds;

    HostBuffer<Vector3> h_centroids;

    int _root = -1;

    int _leaf_size = 1;

    int _morton_bits = 10;

    DeviceBuffer<BVHNode> d_nodes;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<TriangleContainer4> d_triangles;

private:
    static void
    assign_solid_angle_moment(BVHNode& node,
                              const TriangleContainer4& triangle) noexcept;

    static void
    merge_solid_angle_moment(BVHNode& node,
                             const BVHNode& left,
                             const BVHNode& right) noexcept;

    ATLAS_NODISCARD static int
    leaf_node_index(int k, int n) noexcept;

    ATLAS_NODISCARD static unsigned
    expand_bits(unsigned v) noexcept;

    ATLAS_NODISCARD uint32_t
    morton3(const Vector3& p, const AABB& cb, int bits) const noexcept;

    ATLAS_NODISCARD static int
    clz32(uint32_t x) noexcept;

    ATLAS_NODISCARD static int
    clz64(uint64_t x) noexcept;

    ATLAS_NODISCARD static int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    ATLAS_NODISCARD static int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};


}
