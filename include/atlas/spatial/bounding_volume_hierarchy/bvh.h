#pragma once
#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

namespace atlas::geometry {

template <typename T>
struct TriangleMeshGeometryOperator;

} // namespace atlas::geometry

namespace atlas::spatial {

template <typename T>
using BvhGeometryOperator = atlas::geometry::TriangleMeshGeometryOperator<T>;

/**
 * @brief Abstract interface for triangle BVH builders.
 *
 * @details
 * Implementations differ only in how they partition primitives:
 * - LBVH uses Morton-code ordering, approximating spatial locality by bitwise
 *   sorting in a discretized centroid grid.
 * - SAH BVH minimizes an estimate of expected traversal work based on
 *   \f$C = C_t + \sum_i P_i C_i\f$, where \f$P_i\f$ is approximated by
 *   surface-area ratios.
 *
 * Regardless of the build strategy, the output must support the same geometry
 * operator contract: ray traversal returns the closest triangle intersection.
 *
 * @tparam T Floating-point scalar type used in geometry and bounds.
 */
template <typename T>
class BoundingVolumeHierachy {
public:
    BoundingVolumeHierachy()          = default;
    virtual ~BoundingVolumeHierachy() = default;

    /**
     * @brief Build the hierarchy from a triangle container.
     *
     * @param triangles Triangle data to index.
     *
     * @details
     * Builders are expected to compute primitive bounds, aggregate subtree
     * bounds, and populate any host/device storage required for traversal.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    build(const HostBuffer<TriangleContainer4<T>>& triangles)
        = 0;

    /**
     * @brief Create a lightweight traversal geometry operator.
     *
     * @return Geometry operator referencing the BVH storage.
     *
     * @details
     * The returned object is a compact view over the built hierarchy. It does
     * not own memory; callers must keep the source BVH alive.
     */
    ATLAS_HOST virtual BvhGeometryOperator<T>
    make_geometry_operator() const = 0;
};

} // namespace atlas::spatial

namespace atlas {

template <typename T>
using BVH = spatial::BoundingVolumeHierachy<T>;

template <typename T>
using BVHHostPtr = atlas::host_shared_ptr<BVH<T>>;

template <typename T>
using BVHDevicePtr = atlas::device_shared_ptr<BVH<T>>;

} // namespace atlas
