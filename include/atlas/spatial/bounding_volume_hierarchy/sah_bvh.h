#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

namespace atlas {

/**
 * @brief One bucket of the binned surface-area heuristic.
 *
 * During a node split, every primitive centroid falls into one of `_bin_count`
 * equal-width bins along the chosen axis. A bin accumulates the union bound and
 * the count of the primitives it received; prefix/suffix sweeps over the bins
 * then evaluate every candidate split boundary in O(1).
 */
struct Bin {

    /// Union of the bounds of every primitive assigned to this bin.
    AABB bounds;

    /// Number of primitives assigned to this bin.
    int count = 0;
};

/**
 * @brief Top-down BVH built by the binned surface-area heuristic (SAH).
 *
 * Recursively splits the primitive range: at each node it bins centroids along
 * the axis of greatest centroid spread, evaluates the SAH cost
 * `C = 1 + (A_left*N_left + A_right*N_right) / A_parent` at every bin boundary,
 * and partitions at the cheapest boundary. Recursion stops when a range fits in
 * one leaf, when the centroids are spatially degenerate, or when no boundary
 * yields a real two-sided split. The resulting tree traverses more cheaply than
 * an @ref LBVH at the cost of a slower, serial-recursive host build.
 *
 * The node array is over-allocated to `2n - 1` up front and trimmed to the
 * nodes actually produced after the recursion completes.
 */
class SAHBVH final : public BVH {
public:
    /// Construct an empty hierarchy; call @ref build before querying.
    SAHBVH() = default;

    /// Destructor (defaulted); releases host and device buffers.
    ~SAHBVH() override = default;

    /**
     * @brief Build the SAH BVH from a triangle soup and upload it (host).
     *
     * Computes per-triangle bounds and centroids, recursively splits by binned
     * SAH via @ref build_recursive, trims the node array to the produced count,
     * and copies the result to the device. Any prior state is cleared first; an
     * empty input leaves the hierarchy empty (root = -1).
     *
     * @param triangles Triangles to index.
     */
    ATLAS_HOST void
    build(const HostBuffer<TriangleContainer4>& triangles) override;

    /**
     * @brief Gather the built device buffers into a capturable view.
     * @return A @ref BvhView of raw device pointers and the root index.
     */
    ATLAS_NODISCARD ATLAS_HOST BvhView
    view() const override;

    /**
     * @brief Clear all host and device buffers and mark the hierarchy empty.
     */
    ATLAS_HOST void
    reset();

    /**
     * @brief Set the maximum primitives per leaf, clamped to at least 1.
     *
     * A range this size or smaller becomes a leaf without further splitting.
     *
     * @param leaf_size Desired leaf capacity; values below 1 are raised to 1.
     */
    void
    set_leaf_size(const int leaf_size) noexcept {
        _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
    }

    /**
     * @brief Set the number of SAH bins per split, clamped to [4, 256].
     *
     * More bins evaluate more candidate split planes (finer, potentially better
     * splits) at higher per-node cost.
     *
     * @param count Desired bin count; clamped into [4, 256].
     */
    void
    set_bin_count(int count) noexcept {
        if (count < 4) count = 4;
        if (count > 256) count = 256;

        _bin_count = count;
    }

    /**
     * @brief Current maximum primitives per leaf.
     * @return The configured leaf size.
     */
    ATLAS_NODISCARD int
    leaf_size() const noexcept {
        return _leaf_size;
    }

    /**
     * @brief Current SAH bin count.
     * @return The configured number of bins per split.
     */
    ATLAS_NODISCARD int
    bin_count() const noexcept {
        return _bin_count;
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
     * @brief Host-side node array (trimmed to the produced node count).
     * @return Const reference to the host nodes.
     */
    ATLAS_NODISCARD const HostBuffer<BVHNode>&
    nodes() const noexcept {
        return h_nodes;
    }

    /**
     * @brief Host-side primitive index array, partitioned into leaf runs.
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
    /// Host node array, trimmed to the produced node count after a build.
    HostBuffer<BVHNode> h_nodes;

    /// Primitive ids partitioned so each leaf covers a contiguous run.
    HostBuffer<int> h_indices;

    /// Per-primitive AABBs, indexed by original triangle id.
    HostBuffer<AABB> h_prim_bounds;

    /// Per-primitive centroids, indexed by original triangle id.
    HostBuffer<Float3> h_centroids;

    /// Root node index, -1 until a successful build.
    int _root = -1;

    /// Maximum primitives per leaf (always >= 1); the recursion stops once a range fits.
    int _leaf_size = 32;

    /// Number of SAH bins per split (in [4, 256]).
    int _bin_count = 100;

    /// Device copy of @ref h_nodes.
    DeviceBuffer<BVHNode> d_nodes;

    /// Device copy of @ref h_indices.
    DeviceBuffer<int> d_indices;

    /// Device copy of the source triangles.
    DeviceBuffer<TriangleContainer4> d_triangles;

private:
    /**
     * @brief Accumulate solid-angle moments over a leaf's primitive range.
     *
     * Sums the area-weighted centroid, area-scaled normal, and area of every
     * triangle in the reordered range [start, end) into @p node, skipping
     * degenerate (zero-area) triangles.
     *
     * @param node      Leaf node to populate.
     * @param start     First offset into the reordered index array (inclusive).
     * @param end       One past the last offset (exclusive).
     * @param triangles Source triangle soup (indexed via @ref h_indices).
     */
    ATLAS_HOST void
    assign_solid_angle_moment(BVHNode& node,
                              int start,
                              int end,
                              const HostBuffer<TriangleContainer4>& triangles) const noexcept;

    /**
     * @brief Combine two children's solid-angle moments into their parent.
     * @param node  Parent node to fill.
     * @param left  Left child.
     * @param right Right child.
     */
    ATLAS_HOST static void
    merge_solid_angle_moment(BVHNode& node,
                             const BVHNode& left,
                             const BVHNode& right) noexcept;

    /**
     * @brief Recursively build the subtree covering primitives [start, end).
     *
     * Allocates the next node from @p node_count, computes its bounds, and
     * either emits a leaf (small, degenerate, or unsplittable range) or bins the
     * centroids, picks the cheapest SAH boundary, partitions @ref h_indices in
     * place, and recurses into both halves. Returns the index of the node it
     * created so the parent can link it.
     *
     * @param start      First offset into the reordered index array (inclusive).
     * @param end        One past the last offset (exclusive).
     * @param node_count Running high-water mark of allocated nodes; advanced here.
     * @param triangles  Source triangle soup.
     * @return Index of the node created for this range.
     */
    ATLAS_HOST int
    build_recursive(int start,
                    int end,
                    int& node_count,
                    const HostBuffer<TriangleContainer4>& triangles);
};

}