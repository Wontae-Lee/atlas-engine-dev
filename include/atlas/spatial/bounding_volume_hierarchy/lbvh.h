#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

/**
 * @brief Linear BVH builder based on Morton-code sorting.
 *
 * @details
 * The algorithm maps primitive centroids from a continuous domain to a discrete
 * integer lattice, computes a 3D Morton code for each centroid, sorts by that
 * code, and then reconstructs a binary radix tree from longest-common-prefix
 * relationships between adjacent keys.
 *
 * This yields build complexity dominated by sorting and approximates the spatial
 * locality relation
 * \f[
 *   \|c_i - c_j\| \text{ small } \Rightarrow \mathrm{Morton}(c_i)
 *   \text{ shares a long prefix with } \mathrm{Morton}(c_j).
 * \f]
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class LinearBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /// Construct an empty LBVH with default leaf and Morton settings.
    LinearBoundingVolumeHierachy() = default;

    /// Destroy the hierarchy and its owned buffers.
    ~LinearBoundingVolumeHierachy() override = default;

    /**
     * @brief Build the LBVH from triangles.
     *
     * @param triangles Primitive set to index.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /**
     * @brief Expose the built hierarchy as a traversal geometry operator.
     *
     * @return Non-owning geometry operator over the internal buffers.
     */
    ATLAS_HOST BvhGeometryOperator<T>
    make_geometry_operator() const override;

    /// Clear all host/device buffers and mark the hierarchy as empty.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    /// Set the target number of primitives per leaf, clamped to at least 1.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    /// Set the number of centroid quantization bits per axis, clamped to [1, 10].
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_morton_bits(int morton_bits) noexcept;

    /// Return the configured leaf size threshold.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /// Return the number of Morton bits used for each axis.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    morton_bits() const noexcept;

    /// Return the root node index, or `-1` when empty.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    /// Return host-side BVH node storage.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    /// Return the primitive permutation induced by Morton sorting.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    /// Return per-primitive axis-aligned bounds.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    /// Return per-primitive centroids used for Morton encoding.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    /// Return device-side BVH node storage for GPU traversal.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    /// Return device-side primitive index permutation.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    /// Return device-side primitive geometry storage.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    HostBuffer<BVHNode<T>> h_nodes;
    HostBuffer<int> h_indices;
    HostBuffer<AABB<T>> h_prim_bounds;
    HostBuffer<Vector3<T>> h_centroids;

    int _root        = -1;
    int _leaf_size   = 1;
    int _morton_bits = 10;

    DeviceBuffer<BVHNode<T>> d_nodes;
    DeviceBuffer<int> d_indices;
    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    /**
     * @brief Map the k-th leaf in Morton order to its node-array slot.
     *
     * @details
     * The implementation stores `n - 1` internal nodes first, followed by the
     * `n` leaves, so the leaf block begins at offset `n - 1`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    leaf_node_index(int k, int n) noexcept;

    /**
     * @brief Insert two zero bits between each input bit.
     *
     * @details
     * This is the classic bit-dilation step used to build 3D Morton codes:
     * an input pattern `abc...` becomes `a00b00c00...`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE unsigned
    expand_bits(unsigned v) noexcept;

    /**
     * @brief Compute a 30-bit Morton code for a point in the centroid bounds.
     *
     * @param p Primitive centroid.
     * @param cb Bounding box used for normalization.
     * @param bits Quantization bits per axis.
     * @return Interleaved Morton code.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE uint32_t
    morton3(const Vector3<T>& p, const AABB<T>& cb, int bits) const noexcept;

    /// Count leading zeros of a 32-bit integer.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz32(uint32_t x) noexcept;

    /// Count leading zeros of a 64-bit integer.
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz64(uint64_t x) noexcept;

    /**
     * @brief Compute the longest-common-prefix length between two sorted keys.
     *
     * @details
     * Larger values indicate that two Morton keys share more high-order bits and
     * therefore remain in the same radix-tree branch deeper into the hierarchy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    /**
     * @brief Choose the split point inside a sorted Morton interval.
     *
     * @details
     * The selected index maximizes the shared prefix with `first_code` while
     * staying strictly inside `(first, last)`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};

} // namespace atlas::spatial

namespace atlas {

template <typename T>
using LBVH = spatial::LinearBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.hpp>
