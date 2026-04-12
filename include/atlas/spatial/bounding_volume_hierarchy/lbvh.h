#pragma once

/**
 * @file lbvh.h
 * @brief Declares a linear BVH builder and storage container.
 */
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

/**
 * @brief Linear bounding volume hierarchy built from Morton ordering.
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
class LinearBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    LinearBoundingVolumeHierachy() = default;

    ~LinearBoundingVolumeHierachy() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    ATLAS_HOST BvhGeometryOperator<T>
    make_geometry_operator() const override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_morton_bits(int morton_bits) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    morton_bits() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    HostBuffer<BVHNode<T>> h_nodes; ///< Host-side node storage.
    HostBuffer<int> h_indices; ///< Host-side primitive index order.
    HostBuffer<AABB<T>> h_prim_bounds; ///< Host-side primitive bounds.
    HostBuffer<Vector3<T>> h_centroids; ///< Host-side primitive centroids.

    int _root        = -1; ///< Root node index.
    int _leaf_size   = 1; ///< Maximum primitives per leaf.
    int _morton_bits = 10; ///< Quantization bits used for Morton codes.

    DeviceBuffer<BVHNode<T>> d_nodes; ///< Device-side node storage.
    DeviceBuffer<int> d_indices; ///< Device-side primitive index order.
    DeviceBuffer<TriangleContainer4<T>> d_triangles; ///< Device-side primitive storage.

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    leaf_node_index(int k, int n) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE unsigned
    expand_bits(unsigned v) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE uint32_t
    morton3(const Vector3<T>& p, const AABB<T>& cb, int bits) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz32(uint32_t x) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz64(uint64_t x) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};

}

namespace atlas {

template <typename T>
using LBVH = spatial::LinearBoundingVolumeHierachy<T>;

}

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.hpp>
