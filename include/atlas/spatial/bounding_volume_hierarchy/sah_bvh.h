#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

namespace sah {

    /**
     * @brief Temporary SAH histogram bin.
     *
     * @details
     * Each bin accumulates:
     * - the union of primitive bounds that fall into the bin, and
     * - the number of primitives in that subset.
     *
     * During evaluation, prefix and suffix scans estimate left/right subtree
     * costs without re-partitioning the primitives for every candidate split.
     */
    template <typename T>
    struct Bin {
        /// Union of primitive bounds assigned to this bin.
        AABB<T> bounds;

        /// Number of primitives accumulated in the bin.
        int count = 0;
    };

} // namespace sah

/**
 * @brief BVH builder using the surface area heuristic (SAH).
 *
 * @details
 * For a candidate split into left/right subsets, the builder estimates the
 * expected traversal cost with
 * \f[
 *   C_{\mathrm{split}} = C_t +
 *   \frac{A_L}{A_P} N_L +
 *   \frac{A_R}{A_P} N_R,
 * \f]
 * where \f$A_P\f$ is parent area, \f$A_L, A_R\f$ are child areas, and
 * \f$N_L, N_R\f$ are primitive counts. The area ratio approximates the
 * probability that a random ray entering the parent also visits a child.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class SurfaceAreaHeuristicBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /// Construct an empty SAH BVH with default tuning parameters.
    SurfaceAreaHeuristicBoundingVolumeHierachy() = default;

    /// Destroy the hierarchy and its owned buffers.
    ~SurfaceAreaHeuristicBoundingVolumeHierachy() override = default;

    /// Build the hierarchy from a triangle container.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /// Create a traversal operator referencing the built buffers.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhTraceOperator<T>
    make_trace_operator() const override;

    /// Reset the hierarchy to an empty state.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    /// Set the maximum number of primitives allowed in a leaf.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    /// Set the histogram resolution used for approximate SAH evaluation.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_num_of_bins(int num_bins) noexcept;

    /// Return the current leaf-size threshold.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /// Return the number of SAH bins.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    num_of_bins() const noexcept;

    /// Return the root node index, or `-1` when empty.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    /// Return host-side node storage.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    /// Return the primitive ordering stored in the BVH leaves.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    /// Return per-primitive bounds.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    /// Return primitive centroids.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    /// Return device-side node storage.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    /// Return device-side primitive indices.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    /// Return device-side triangles.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    HostBuffer<BVHNode<T>> h_nodes;
    HostBuffer<int> h_indices;
    HostBuffer<AABB<T>> h_prim_bounds;
    HostBuffer<Vector3<T>> h_centroids;

    int _root        = -1;
    int _leaf_size   = 32;
    int _num_of_bins = 100;

    DeviceBuffer<BVHNode<T>> d_nodes;
    DeviceBuffer<int> d_indices;
    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    /**
     * @brief Recursively partition the primitive range `[start, end)`.
     *
     * @param start Inclusive range start in `h_indices`.
     * @param end Exclusive range end in `h_indices`.
     * @param node_count Running allocator cursor for `h_nodes`.
     * @return Index of the constructed node.
     */
    int
    build_recursive(int start, int end, int& node_count);
};

} // namespace atlas::spatial

namespace atlas {

template <typename T>
using SAHBVH = spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.hpp>
