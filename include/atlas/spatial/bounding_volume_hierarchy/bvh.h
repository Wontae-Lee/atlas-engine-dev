#pragma once
#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/triangle.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>

namespace atlas {
namespace spatial {
    template <typename T>
    struct BVHNode {
        AABB<T> bounds;
        int left = -1, right = -1;
        int start = -1, count = 0;
        bool is_leaf = false;
    };

    template <typename T>
    class BvhTraceOperator final {
    public:
        BvhTraceOperator()      = default;
        ~BvhTraceOperator()     = default;
        const BVHNode<T>* nodes = nullptr;
        const int* indices      = nullptr;
        const Triangle<T>* tris = nullptr;
        int root                = -1;
        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& r) const;
    };

    template <typename T>
    class BoundingVolumeHierachy {
    public:
        BoundingVolumeHierachy()          = default;
        virtual ~BoundingVolumeHierachy() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        build(const HostBuffer<Triangle<T>>& triangles)
            = 0;
        ATLAS_HOST virtual BvhTraceOperator<T>
        make_trace_operator() const = 0;
    };
}

template <typename T>
using BvhTraceOperator = spatial::BvhTraceOperator<T>;
template <typename T>
using BVH = spatial::BoundingVolumeHierachy<T>;
template <typename T>
using BVHHostPtr = atlas::host_shared_ptr<BVH<T>>;
template <typename T>
using BVHDevicePtr = atlas::device_shared_ptr<BVH<T>>;
}

#include <atlas/spatial/bounding_volume_hierarchy/bvh.hpp>