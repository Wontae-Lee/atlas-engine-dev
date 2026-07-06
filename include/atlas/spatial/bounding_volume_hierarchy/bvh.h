#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>

/**
 * @file bvh.h
 * @brief Host-only interface for building a triangle-mesh BVH, common
 *        to `LBVH` (fast, parallel-friendly, lower query quality) and
 *        `SAHBVH` (slower to build, tighter/higher-quality tree). See
 *        `node.h` for what each node stores and why.
 */

namespace atlas {

struct TriangleMeshGeometryOperator;

using BvhGeometryOperator = TriangleMeshGeometryOperator;

/**
 * @brief Base interface for a triangle-mesh bounding volume hierarchy
 *        builder. See `lbvh.h`/`sah_bvh.h` for the two construction
 *        strategies.
 */
class BVH {
public:
    BVH() = default;

    virtual ~BVH() = default;

    /** @brief Builds the hierarchy over `triangles` from scratch,
     *  replacing any previously built tree. */
    ATLAS_HOST virtual void
    build(const HostBuffer<TriangleContainer4>& triangles)
        = 0;

    /** @brief Produces the device-callable `TriangleMeshGeometryOperator`
     *  view over this built hierarchy (device-side ray/point queries
     *  read the tree through this operator, not through `BVH` itself,
     *  which is host-only). */
    ATLAS_HOST virtual BvhGeometryOperator
    make_geometry_operator() const = 0;
};

using BVHHostPtr = atlas::host_shared_ptr<BVH>;

using BVHDevicePtr = atlas::device_shared_ptr<BVH>;

}
