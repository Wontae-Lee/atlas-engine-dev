#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

/**
 * @file sah_bvh.h
 * @brief Top-down BVH construction minimizing the Surface Area
 *        Heuristic (SAH) cost at every split, via Wald's (2007) binned
 *        approximation — slower to build than `LBVH` but produces
 *        tighter trees (fewer, more targeted ray/query traversal steps).
 *
 * @details
 * ### Background — the surface area heuristic
 * The expected cost of a ray query through a BVH node with children
 * `L`, `R` is modeled as
 * `C = C_trav + (A(L)/A(N)) * count(L) * C_isect + (A(R)/A(N)) *
 * count(R) * C_isect`, where `A(.)` is a box's surface area, `C_trav`
 * the fixed cost of descending a node, and `C_isect` the cost of an
 * exact primitive intersection test — the probability a ray passing
 * through parent box `N` also passes through child box `L`/`R` is
 * approximated by the ratio of surface areas (valid under a uniform
 * random ray direction assumption), so minimizing this expression at
 * every split greedily minimizes expected traversal cost (Goldsmith &
 * Salmon, 1987; the standard cost model behind "SAH" BVH construction).
 * Exhaustively evaluating this cost at every possible split position
 * (every primitive boundary, sorted) is `O(n log n)` per node; Wald's
 * binned approximation instead partitions the split axis into
 * `bin_count` equal-width buckets (`Bin`, holding each bucket's merged
 * bounds and primitive count) and only evaluates the cost at the
 * `bin_count - 1` bucket boundaries — `O(n)` per node (one pass to bin,
 * one to sweep bucket prefix sums for the cost) at a small accuracy
 * cost relative to the exact evaluation.
 *
 * ### Operating principle
 * `build_recursive` (per node): `compute_range_bounds` computes both
 * the node's own AABB and the *centroid* AABB (`RangeBounds` — the
 * spread of primitive centroids, used to choose which axis and where to
 * bin, independent of primitive extent); `choose_sah_split` bins
 * centroids into `bin_count` buckets along the longest centroid-spread
 * axis, sweeps prefix/suffix bucket-bound merges to evaluate the binned
 * SAH cost at each boundary, and returns the cheapest (`SplitChoice`);
 * `partition_sah_split` then partitions the primitive range in place
 * around that boundary (a Hoare/quicksort-style partition, not a full
 * sort); `make_leaf`/`make_internal` finalize whichever the chosen
 * split (or a degenerate/too-small range, which always becomes a leaf)
 * produces. The same winding-number moment computation described in
 * `node.h` runs alongside leaf/internal-node construction.
 *
 * ### References
 * - J. Goldsmith and J. Salmon, "Automatic Creation of Object
 *   Hierarchies for Ray Tracing," IEEE Computer Graphics and
 *   Applications 7(5), 1987. (the surface-area-heuristic cost model)
 * - I. Wald, "On fast Construction of SAH-based Bounding Volume
 *   Hierarchies," IEEE Symposium on Interactive Ray Tracing, 2007. (the
 *   binned approximation this class implements)
 */

namespace atlas {

/** @brief One bucket of `choose_sah_split`'s binned centroid histogram:
 *  the merged bound and primitive count of every primitive whose
 *  centroid falls in this bucket along the split axis. */
struct Bin {

    AABB bounds;

    int count = 0;
};

/**
 * @brief Binned Surface Area Heuristic BVH builder. See this file's
 *        top-of-file documentation for the SAH cost model and the
 *        binned approximation.
 */
class SAHBVH final : public BVH {
public:
    SAHBVH() = default;

    ~SAHBVH() override = default;

    ATLAS_HOST void
    build(const HostBuffer<TriangleContainer4>& triangles) override;

    ATLAS_NODISCARD ATLAS_HOST BvhGeometryOperator
    make_geometry_operator() const override;

    ATLAS_HOST void
    reset();

    /** @brief Below this many primitives, `build_recursive` always
     *  makes a leaf regardless of SAH cost (clamped to `>= 1`) — avoids
     *  splitting down to single primitives, where per-node traversal
     *  overhead outweighs any intersection-test savings. */
    void
    set_leaf_size(const int leaf_size) noexcept {
        _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
    }

    /** @brief Number of histogram buckets `choose_sah_split` bins
     *  centroids into per axis (clamped to `[4, 256]`); more bins
     *  approximate the exact (unbinned) SAH cost more closely, at
     *  proportionally more work per split. */
    void
    set_bin_count(int count) noexcept {
        if (count < 4) count = 4;
        if (count > 256) count = 256;

        _bin_count = count;
    }

    ATLAS_NODISCARD int
    leaf_size() const noexcept {
        return _leaf_size;
    }

    ATLAS_NODISCARD int
    bin_count() const noexcept {
        return _bin_count;
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

    ATLAS_NODISCARD const HostBuffer<Float3>&
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

    HostBuffer<Float3> h_centroids;

    int _root = -1;

    int _leaf_size = 32;

    int _bin_count = 100;

    DeviceBuffer<BVHNode> d_nodes;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<TriangleContainer4> d_triangles;

private:
    struct RangeBounds {
        AABB node;
        AABB centroid;
        int count {};
        bool degenerate {};
    };

    struct SplitChoice {
        int axis {};
        float cmin {};
        float den {};
        int split { -1 };
    };

    ATLAS_HOST void
    assign_solid_angle_moment(BVHNode& node,
                              int start,
                              int end,
                              const HostBuffer<TriangleContainer4>& triangles) const noexcept;

    ATLAS_HOST static void
    merge_solid_angle_moment(BVHNode& node,
                             const BVHNode& left,
                             const BVHNode& right) noexcept;

    ATLAS_HOST RangeBounds
    compute_range_bounds(int start, int end) const;

    ATLAS_HOST int
    make_leaf(int node_index,
              int start,
              int end,
              const RangeBounds& bounds,
              const HostBuffer<TriangleContainer4>& triangles);

    ATLAS_HOST SplitChoice
    choose_sah_split(int start, int end, const RangeBounds& bounds) const;

    ATLAS_HOST int
    partition_sah_split(int start, int end, const SplitChoice& split);

    ATLAS_HOST int
    make_internal(int node_index, int left_child, int right_child) noexcept;

    ATLAS_HOST int
    build_recursive(int start,
                    int end,
                    int& node_count,
                    const HostBuffer<TriangleContainer4>& triangles);
};

}
