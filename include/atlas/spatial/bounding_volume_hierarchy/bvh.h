#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>

namespace atlas {

struct TriangleMeshView;

/**
 * @brief Device-capturable view of a built BVH.
 *
 * A BVH does not define its own view type; it reuses @ref TriangleMeshView,
 * whose `bvh_*` members (nodes, indices, triangles, root) are exactly the raw
 * device pointers a device-side traversal needs. Gathering them into one
 * trivially-copyable struct lets a device lambda capture the whole acceleration
 * structure by value.
 */
using BvhView = TriangleMeshView;

/**
 * @brief Abstract interface for a triangle bounding-volume hierarchy.
 *
 * Concrete builders (@ref LBVH, @ref SAHBVH) implement two host operations:
 * @ref build to construct the hierarchy from a triangle soup, and @ref view to
 * expose the uploaded device buffers for traversal. The class is polymorphic
 * (owned through the pointer aliases below) so the geometry layer can select a
 * build strategy at runtime.
 */
class BVH {
public:
    /// Default-construct an empty hierarchy; call @ref build before @ref view.
    BVH() = default;

    /// Virtual destructor so derived builders destruct correctly through a base pointer.
    virtual ~BVH() = default;

    /**
     * @brief Build the hierarchy from a triangle soup (host operation).
     *
     * Replaces any previously built state and uploads the result to the device.
     * An empty input leaves the hierarchy empty (root = -1).
     *
     * @param triangles Triangles to index, one @ref TriangleContainer4 each.
     */
    ATLAS_HOST virtual void
    build(const HostBuffer<TriangleContainer4>& triangles)
        = 0;

    /**
     * @brief Return a device-capturable view over the built hierarchy.
     * @return A @ref BvhView of raw device pointers; fields are null / root -1
     *         if nothing has been built.
     */
    ATLAS_HOST virtual BvhView
    view() const = 0;
};

/// Host-side shared ownership of a polymorphic BVH.
using BVHHostPtr = atlas::host_shared_ptr<BVH>;

/// Device-side shared ownership of a polymorphic BVH.
using BVHDevicePtr = atlas::device_shared_ptr<BVH>;

}