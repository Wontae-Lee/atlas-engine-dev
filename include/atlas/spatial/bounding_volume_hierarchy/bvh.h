#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>

namespace atlas {

struct TriangleMeshGeometryOperator;

using BvhGeometryOperator = TriangleMeshGeometryOperator;

class BVH {
public:
    BVH() = default;

    virtual ~BVH() = default;

    ATLAS_HOST virtual void
    build(const HostBuffer<TriangleContainer4>& triangles)
        = 0;

    ATLAS_HOST virtual BvhGeometryOperator
    make_geometry_operator() const = 0;
};

using BVHHostPtr = atlas::host_shared_ptr<BVH>;

using BVHDevicePtr = atlas::device_shared_ptr<BVH>;

}
