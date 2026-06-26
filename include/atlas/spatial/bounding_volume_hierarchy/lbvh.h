#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/random/seed.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas {

template <typename T>
class LinearBoundingVolumeHierarchy final : public BoundingVolumeHierarchy<T> {
public:
    LinearBoundingVolumeHierarchy() = default;

    ~LinearBoundingVolumeHierarchy() override = default;

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
    HostBuffer<BVHNode<T>> h_nodes;

    HostBuffer<int> h_indices;

    HostBuffer<AABB<T>> h_prim_bounds;

    HostBuffer<Vector3<T>> h_centroids;

    int _root = -1;

    int _leaf_size = 1;

    int _morton_bits = 10;

    DeviceBuffer<BVHNode<T>> d_nodes;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    static void
    assign_solid_angle_moment(BVHNode<T>& node,
                              const TriangleContainer4<T>& triangle) noexcept;

    static void
    merge_solid_angle_moment(BVHNode<T>& node,
                             const BVHNode<T>& left,
                             const BVHNode<T>& right) noexcept;

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
using LBVH = LinearBoundingVolumeHierarchy<T>;

}

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.hpp>