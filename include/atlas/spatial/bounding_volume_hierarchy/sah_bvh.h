#ifndef INCLUDE_ATLAS_SPATIAL_BOUNDING_VOLUME_HIERARCHY_SAH_BVH_H
#define INCLUDE_ATLAS_SPATIAL_BOUNDING_VOLUME_HIERARCHY_SAH_BVH_H

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/triangle.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas {
namespace spatial {
    namespace sah {
        template <typename T>
        struct Bin {
            AABB<T> bounds;
            int count = 0;
        };
    }

    template <typename T>
    class SurfaceAreaHeuristicBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
    public:
        SurfaceAreaHeuristicBoundingVolumeHierachy()           = default;
        ~SurfaceAreaHeuristicBoundingVolumeHierachy() override = default;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        build(const HostBuffer<Triangle<T>>& triangles) override;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhTraceOperator<T>
        make_trace_operator() const override;

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
        ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Triangle<T>>&
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
        DeviceBuffer<Triangle<T>> d_triangles;

        int
        build_recursive(int start, int end, int& node_count);
    };
}

template <typename T>
using SAHBVH = spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<T>;

}

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.hpp>
#endif