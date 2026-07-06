#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

namespace atlas {

struct Bin {

    AABB bounds;

    int count = 0;
};

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

    void
    set_leaf_size(const int leaf_size) noexcept {
        _leaf_size = (leaf_size < 1) ? 1 : leaf_size;
    }

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
    ATLAS_HOST void
    assign_solid_angle_moment(BVHNode& node,
                              int start,
                              int end,
                              const HostBuffer<TriangleContainer4>& triangles) const noexcept;

    ATLAS_HOST static void
    merge_solid_angle_moment(BVHNode& node,
                             const BVHNode& left,
                             const BVHNode& right) noexcept;

    ATLAS_HOST int
    build_recursive(int start,
                    int end,
                    int& node_count,
                    const HostBuffer<TriangleContainer4>& triangles);
};

}
