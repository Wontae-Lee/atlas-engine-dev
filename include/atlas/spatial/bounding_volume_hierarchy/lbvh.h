#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <cstdint>

namespace atlas {

class LBVH final : public BVH {
public:
    LBVH() = default;

    ~LBVH() override = default;

    ATLAS_HOST void
    build(const HostBuffer<TriangleContainer4>& triangles) override;

    ATLAS_HOST BvhView
    view() const override;

    ATLAS_HOST void
    reset();

    void
    set_leaf_size(const int leaf_size) noexcept {
        _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
    }

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
    morton3(const Float3& p, const AABB& cb, int bits) const noexcept;

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