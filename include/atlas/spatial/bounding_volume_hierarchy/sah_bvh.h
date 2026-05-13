#pragma once

/**
 * @file sah_bvh.h
 * @brief Declares a binned surface-area-heuristic BVH builder and runtime export interface.
 *
 * @details
 * This header defines a concrete bounding volume hierarchy, abbreviated as BVH,
 * that uses a binned surface area heuristic, abbreviated as SAH, to recursively
 * partition triangle primitives.
 *
 * The main type is:
 *
 * - @ref atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy
 *
 * The builder stores construction data on the host and uploads traversal-ready
 * data to device buffers after construction. A lightweight
 * @ref BvhGeometryOperator can then be exported for runtime geometry queries.
 *
 * ## Surface area heuristic
 *
 * The surface area heuristic estimates the cost of splitting a parent node into
 * left and right child nodes. In this implementation, candidate splits are
 * evaluated using centroid bins. For a candidate split, the approximate cost is:
 *
 * @f[
 *     C
 *     =
 *     1
 *     +
 *     \frac{
 *         A_L N_L
 *         +
 *         A_R N_R
 *     }{
 *         A_P
 *     },
 * @f]
 *
 * where:
 *
 * - @f$A_P@f$ is the parent node surface area,
 * - @f$A_L@f$ is the left child surface area,
 * - @f$A_R@f$ is the right child surface area,
 * - @f$N_L@f$ is the number of primitives assigned to the left side,
 * - @f$N_R@f$ is the number of primitives assigned to the right side.
 *
 * The constant term represents the cost of visiting the current internal node.
 *
 * ## Build overview
 *
 * The implementation follows this high-level process:
 *
 * 1. compute one AABB per triangle primitive,
 * 2. compute one centroid per triangle primitive,
 * 3. initialize the primitive index order,
 * 4. recursively build BVH nodes over index intervals,
 * 5. choose split axes from centroid bounds,
 * 6. bin primitive centroids along the selected axis,
 * 7. evaluate SAH split costs between adjacent bins,
 * 8. partition primitive indices,
 * 9. terminate recursion when a leaf condition is met,
 * 10. upload the final node, index, and triangle buffers to device storage.
 *
 * ## Naming note
 *
 * The class name uses the spelling `Hierachy` in the current public API. The
 * spelling is preserved for source compatibility.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

namespace sah {

    /**
     * @brief Temporary centroid bin used during binned SAH split evaluation.
     *
     * @details
     * A bin accumulates the primitives whose centroids fall into the same
     * normalized interval along the selected split axis.
     *
     * If the centroid projection range on the selected axis is:
     *
     * @f[
     *     [c_{\min}, c_{\max}],
     * @f]
     *
     * then a primitive centroid coordinate @f$c_i@f$ is normalized as:
     *
     * @f[
     *     t_i
     *     =
     *     \frac{
     *         c_i - c_{\min}
     *     }{
     *         c_{\max} - c_{\min}
     *     }.
     * @f]
     *
     * The bin index is then computed approximately as:
     *
     * @f[
     *     b_i
     *     =
     *     \left\lfloor
     *         t_i N_{\mathrm{bins}}
     *     \right\rfloor,
     * @f]
     *
     * clamped to the valid range:
     *
     * @f[
     *     0 \le b_i < N_{\mathrm{bins}}.
     * @f]
     *
     * Each bin stores:
     *
     * - the merged primitive bounds of all primitives assigned to the bin,
     * - the number of primitives assigned to the bin.
     *
     * @tparam T Floating-point scalar type used for primitive bounds.
     */
    template <typename T>
    struct Bin {
        /**
         * @brief Bounds enclosing all primitives assigned to this bin.
         *
         * @details
         * If the bin contains primitives @f$P_0,\dots,P_{n-1}@f$, this value is
         * the component-wise union of their AABBs:
         *
         * @f[
         *     B_{\mathrm{bin}}
         *     =
         *     \bigcup_{k=0}^{n-1}
         *     B(P_k).
         * @f]
         */
        AABB<T> bounds;

        /**
         * @brief Number of primitives assigned to this bin.
         *
         * @details
         * A zero count means the bin is empty and its bounds should not contribute
         * to prefix or suffix bounds until a non-empty bin is merged.
         */
        int count = 0;
    };

} // namespace sah

/**
 * @brief Surface-area-heuristic bounding volume hierarchy for triangle primitives.
 *
 * @details
 * SurfaceAreaHeuristicBoundingVolumeHierachy builds a binary BVH over packed
 * triangle primitives. The BVH is constructed on the host and then mirrored into
 * device buffers for runtime traversal.
 *
 * The hierarchy stores primitives indirectly through an index buffer. During
 * recursive construction, @ref h_indices is partitioned instead of physically
 * rearranging the host primitive input. Leaf nodes store a contiguous interval
 * into this index buffer.
 *
 * ## Node ranges
 *
 * Each recursive build call operates on a half-open primitive index interval:
 *
 * @f[
 *     [\mathrm{start}, \mathrm{end}).
 * @f]
 *
 * The number of primitives in the current node is:
 *
 * @f[
 *     N
 *     =
 *     \mathrm{end}
 *     -
 *     \mathrm{start}.
 * @f]
 *
 * A leaf node stores:
 *
 * - @c start, the first index in @ref h_indices,
 * - @c count, the number of primitives in the leaf.
 *
 * An interior node stores:
 *
 * - @c left, the left child node index,
 * - @c right, the right child node index,
 * - merged bounds of both children.
 *
 * ## Split axis
 *
 * For each internal candidate node, centroid bounds are computed. The split axis
 * is selected as the major axis of the centroid extent:
 *
 * @f[
 *     \mathbf{e}
 *     =
 *     \mathbf{c}_{\max}
 *     -
 *     \mathbf{c}_{\min}.
 * @f]
 *
 * The selected axis is the component with the largest extent.
 *
 * ## Leaf conditions
 *
 * Recursion terminates and creates a leaf if:
 *
 * - the primitive count is less than or equal to @ref _leaf_size,
 * - centroid bounds are degenerate,
 * - the centroid projection range has zero width,
 * - no valid SAH split is found,
 * - partitioning fails to produce two non-empty child ranges.
 *
 * ## Host/device data model
 *
 * Host buffers are used for construction and inspection:
 *
 * - @ref h_nodes,
 * - @ref h_indices,
 * - @ref h_prim_bounds,
 * - @ref h_centroids.
 *
 * Device buffers are used for runtime traversal:
 *
 * - @ref d_nodes,
 * - @ref d_indices,
 * - @ref d_triangles.
 *
 * @tparam T Floating-point scalar type used by triangle geometry, AABBs, and
 *           centroid computations.
 *
 * @note The implementation uses binned SAH, not exact exhaustive SAH over every
 *       possible primitive split.
 * @note Larger bin counts can improve split quality but increase build cost.
 * @note The public class name keeps the existing `Hierachy` spelling for API
 *       compatibility.
 *
 * @see BoundingVolumeHierachy
 * @see BvhGeometryOperator
 * @see BVHNode
 * @see AABB
 */
template <typename T>
class SurfaceAreaHeuristicBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty hierarchy with default build parameters:
     *
     * - leaf size: @f$32@f$,
     * - number of SAH bins: @f$100@f$,
     * - root index: @f$-1@f$.
     *
     * No nodes or primitive data are allocated until build() is called.
     */
    SurfaceAreaHeuristicBoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Allows the hierarchy to be destroyed through the base
     * BoundingVolumeHierachy interface.
     */
    ~SurfaceAreaHeuristicBoundingVolumeHierachy() override = default;

    /**
     * @brief Builds the SAH BVH from host-side triangle primitives.
     *
     * @details
     * This function rebuilds the hierarchy from scratch.
     *
     * The build process:
     *
     * 1. clears the current hierarchy state,
     * 2. computes each primitive AABB,
     * 3. computes each primitive centroid,
     * 4. initializes primitive indices as @f$0,\dots,N-1@f$,
     * 5. recursively builds the BVH over @f$[0,N)@f$,
     * 6. trims unused host node storage,
     * 7. uploads nodes, indices, and triangles to device buffers.
     *
     * For each triangle primitive @f$P_i@f$, the implementation caches:
     *
     * @f[
     *     B_i = \mathrm{bounds}(P_i),
     *     \qquad
     *     \mathbf{c}_i = \mathrm{centroid}(P_i).
     * @f]
     *
     * If @p triangles is empty, the hierarchy is reset and remains empty.
     *
     * @param triangles Host-side packed triangle containers used as input
     *                  primitives.
     *
     * @note Existing hierarchy contents are discarded before building.
     * @note The input triangle order is preserved in @ref d_triangles, while
     *       @ref h_indices and @ref d_indices store the BVH traversal ordering.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /**
     * @brief Creates a lightweight geometry operator for device-side traversal.
     *
     * @details
     * The returned operator contains raw pointers to device buffers:
     *
     * - BVH nodes,
     * - primitive index order,
     * - packed triangle storage,
     * - root node index.
     *
     * This allows downstream geometry code to traverse the BVH without owning the
     * hierarchy container itself.
     *
     * @return BVH geometry operator bound to this hierarchy's device buffers.
     *
     * @warning The returned operator contains raw pointers into this object's
     *          device buffers. It remains valid only while the hierarchy object
     *          and its device buffers remain alive and unchanged.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhGeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Clears all hierarchy data and marks the tree as empty.
     *
     * @details
     * This function clears:
     *
     * - host node storage,
     * - host primitive indices,
     * - host primitive centroids,
     * - host primitive bounds,
     * - device node storage,
     * - device primitive indices,
     * - device triangle storage.
     *
     * It also resets the root index to @f$-1@f$.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    /**
     * @brief Sets the maximum number of primitives allowed in a leaf node.
     *
     * @details
     * The value is clamped to at least @f$1@f$:
     *
     * @f[
     *     \mathrm{leaf\_size}
     *     =
     *     \max(1, \mathrm{input}).
     * @f]
     *
     * A smaller leaf size generally creates a deeper tree with fewer primitives
     * per leaf. A larger leaf size generally creates a shallower tree with more
     * primitives per leaf.
     *
     * @param leaf_size Requested maximum primitive count per leaf.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    /**
     * @brief Sets the number of bins used for binned SAH split evaluation.
     *
     * @details
     * The requested value is clamped to:
     *
     * @f[
     *     4
     *     \le
     *     N_{\mathrm{bins}}
     *     \le
     *     256.
     * @f]
     *
     * More bins provide more candidate split positions but increase construction
     * cost.
     *
     * @param num_bins Requested number of SAH bins.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_num_of_bins(int num_bins) noexcept;

    /**
     * @brief Returns the configured maximum number of primitives per leaf.
     *
     * @return Leaf size used as a recursive stopping criterion.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /**
     * @brief Returns the configured number of SAH bins.
     *
     * @return Number of bins used for split-cost approximation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    num_of_bins() const noexcept;

    /**
     * @brief Returns the root node index.
     *
     * @details
     * The root index identifies the top node in @ref h_nodes and @ref d_nodes.
     * A value of @f$-1@f$ means the hierarchy is empty or has not been built.
     *
     * @return Root node index.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    /**
     * @brief Returns host-side BVH node storage.
     *
     * @details
     * The returned buffer contains both interior and leaf nodes in construction
     * order.
     *
     * @return Const reference to the host node buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    /**
     * @brief Returns host-side primitive index ordering.
     *
     * @details
     * Leaf nodes reference contiguous subranges of this buffer. Each value is an
     * index into the original triangle array.
     *
     * @return Const reference to the host primitive index buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    /**
     * @brief Returns host-side primitive bounds.
     *
     * @details
     * Stores one AABB per input primitive:
     *
     * @f[
     *     B_i = \mathrm{bounds}(P_i).
     * @f]
     *
     * @return Const reference to the primitive bounds buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    /**
     * @brief Returns host-side primitive centroids.
     *
     * @details
     * Stores one centroid per input primitive:
     *
     * @f[
     *     \mathbf{c}_i = \mathrm{centroid}(P_i).
     * @f]
     *
     * Centroids are used for binning and split-axis selection.
     *
     * @return Const reference to the primitive centroid buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    /**
     * @brief Returns device-side BVH node storage.
     *
     * @details
     * This buffer is used by exported geometry operators during device-side
     * traversal.
     *
     * @return Const reference to the device node buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    /**
     * @brief Returns device-side primitive index ordering.
     *
     * @details
     * This buffer mirrors the host primitive index order used by leaf nodes.
     *
     * @return Const reference to the device primitive index buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    /**
     * @brief Returns device-side packed triangle storage.
     *
     * @details
     * The exported geometry operator uses this buffer together with
     * @ref device_indices() to map BVH leaf references to triangle primitives.
     *
     * @return Const reference to the device triangle buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    /**
     * @brief Host-side BVH node storage.
     *
     * @details
     * Contains all generated BVH nodes. Interior nodes store child indices and
     * merged child bounds. Leaf nodes store a primitive interval in @ref h_indices.
     */
    HostBuffer<BVHNode<T>> h_nodes;

    /**
     * @brief Host-side primitive index permutation.
     *
     * @details
     * Recursive partitioning reorders this buffer. Leaf nodes refer to contiguous
     * intervals in this array instead of copying primitive data.
     */
    HostBuffer<int> h_indices;

    /**
     * @brief Host-side primitive bounds.
     *
     * @details
     * Stores one AABB per input triangle. These bounds are used to construct node
     * bounds and to evaluate SAH costs.
     */
    HostBuffer<AABB<T>> h_prim_bounds;

    /**
     * @brief Host-side primitive centroids.
     *
     * @details
     * Stores one centroid per input triangle. These centroids determine the split
     * axis, bin assignment, and final partitioning.
     */
    HostBuffer<Vector3<T>> h_centroids;

    /**
     * @brief Root node index.
     *
     * @details
     * Stores the index of the root node in the node buffers. The value is @f$-1@f$
     * when the hierarchy is empty.
     */
    int _root = -1;

    /**
     * @brief Maximum number of primitives per leaf node.
     *
     * @details
     * During recursive construction, a node becomes a leaf when:
     *
     * @f[
     *     N \le \mathrm{leaf\_size}.
     * @f]
     */
    int _leaf_size = 32;

    /**
     * @brief Number of bins used for binned SAH split approximation.
     *
     * @details
     * Candidate split positions are evaluated between adjacent bins. With
     * @f$B@f$ bins, there are at most:
     *
     * @f[
     *     B - 1
     * @f]
     *
     * candidate split positions.
     */
    int _num_of_bins = 100;

    /**
     * @brief Device-side BVH node storage.
     *
     * @details
     * Mirrors the completed host node buffer for device-side traversal.
     */
    DeviceBuffer<BVHNode<T>> d_nodes;

    /**
     * @brief Device-side primitive index permutation.
     *
     * @details
     * Mirrors @ref h_indices for runtime traversal on the device.
     */
    DeviceBuffer<int> d_indices;

    /**
     * @brief Device-side packed triangle primitive storage.
     *
     * @details
     * Stores the triangle primitives referenced by BVH leaf nodes.
     */
    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    /**
     * @brief Recursively builds a BVH subtree over a primitive-index interval.
     *
     * @details
     * This function constructs either a leaf node or an interior node for the
     * half-open interval:
     *
     * @f[
     *     [\mathrm{start}, \mathrm{end}).
     * @f]
     *
     * The primitive count in the interval is:
     *
     * @f[
     *     N = \mathrm{end} - \mathrm{start}.
     * @f]
     *
     * The function first computes:
     *
     * - node bounds from primitive AABBs,
     * - centroid bounds from primitive centroids.
     *
     * If the node should become a leaf, it stores:
     *
     * @f[
     *     \mathrm{start},
     *     \qquad
     *     \mathrm{count} = N.
     * @f]
     *
     * Otherwise, it:
     *
     * 1. selects the major axis of the centroid bounds,
     * 2. bins primitive centroids along that axis,
     * 3. computes prefix and suffix bounds/counts,
     * 4. evaluates SAH split costs,
     * 5. partitions @ref h_indices,
     * 6. recursively builds left and right children,
     * 7. stores an interior node with child indices.
     *
     * @param start Inclusive start index into @ref h_indices.
     * @param end Exclusive end index into @ref h_indices.
     * @param node_count Running node allocation counter. This value is incremented
     *                   whenever a new node is allocated.
     *
     * @return Index of the constructed subtree root node.
     */
    int
    build_recursive(int start, int end, int& node_count);
};

} // namespace atlas::spatial

namespace atlas {

/**
 * @brief Convenience alias for the SAH BVH implementation.
 *
 * @details
 * Exposes @ref atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy in the
 * top-level atlas namespace.
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
using SAHBVH = spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.hpp>