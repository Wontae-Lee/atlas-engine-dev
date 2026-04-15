#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>

namespace atlas::geometry {

template <typename T>
struct TriangleMeshGeometryOperator;

}

namespace atlas::spatial {

template <typename T>
using BvhGeometryOperator = atlas::geometry::TriangleMeshGeometryOperator<T>;

template <typename T>
class BoundingVolumeHierachy {
public:
    BoundingVolumeHierachy() = default;

    virtual ~BoundingVolumeHierachy() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    build(const HostBuffer<TriangleContainer4<T>>& triangles)
        = 0;

    ATLAS_HOST virtual BvhGeometryOperator<T>
    make_geometry_operator() const = 0;
};

}

namespace atlas {

template <typename T>
using BVH = spatial::BoundingVolumeHierachy<T>;

template <typename T>
using BVHHostPtr = atlas::host_shared_ptr<BVH<T>>;

template <typename T>
using BVHDevicePtr = atlas::device_shared_ptr<BVH<T>>;

}