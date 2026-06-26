#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas {

template <typename T>
struct Bin {

    AABB<T> bounds;

    int count = 0;
};

template <typename T>
class SurfaceAreaHeuristicBoundingVolumeHierarchy final : public BoundingVolumeHierarchy<T> {
public:
    SurfaceAreaHeuristicBoundingVolumeHierarchy() = default;

    ~SurfaceAreaHeuristicBoundingVolumeHierarchy() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhGeometryOperator<T>
    make_geometry_operator() const override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_num_of_bins(int num_bins) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    num_of_bins() const noexcept;

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

    int _leaf_size = 32;

    int _num_of_bins = 100;

    DeviceBuffer<BVHNode<T>> d_nodes;

    DeviceBuffer<int> d_indices;

    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    struct RangeBounds {
        AABB<T> node;
        AABB<T> centroid;
        int count {};
        bool degenerate {};
    };

    struct SplitChoice {
        int axis {};
        T cmin {};
        T den {};
        int split { -1 };
    };

    void
    assign_solid_angle_moment(BVHNode<T>& node,
                              int start,
                              int end,
                              const HostBuffer<TriangleContainer4<T>>& triangles) const noexcept;

    static void
    merge_solid_angle_moment(BVHNode<T>& node,
                             const BVHNode<T>& left,
                             const BVHNode<T>& right) noexcept;

    RangeBounds
    compute_range_bounds(int start, int end) const;

    int
    make_leaf(int node_index,
              int start,
              int end,
              const RangeBounds& bounds,
              const HostBuffer<TriangleContainer4<T>>& triangles);

    SplitChoice
    choose_sah_split(int start, int end, const RangeBounds& bounds) const;

    int
    partition_sah_split(int start, int end, const SplitChoice& split);

    int
    make_internal(int node_index, int left_child, int right_child) noexcept;

    int
    build_recursive(int start,
                    int end,
                    int& node_count,
                    const HostBuffer<TriangleContainer4<T>>& triangles);
};

}

namespace atlas {

template <typename T>
using SAHBVH = SurfaceAreaHeuristicBoundingVolumeHierarchy<T>;

}

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.hpp>