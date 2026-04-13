#pragma once

/**
 * @file bounding_volume_hierarchy.h
 * @brief Declares the abstract Bounding Volume Hierarchy (BVH) interface used for spatial acceleration.
 *
 * @details
 * This header defines:
 * - a geometry-operator alias for BVH-backed triangle meshes,
 * - the abstract base class @ref atlas::spatial::BoundingVolumeHierachy,
 * - convenience aliases for host/device shared ownership.
 *
 * ## Purpose
 * A Bounding Volume Hierarchy (BVH) is a spatial acceleration structure used to:
 * - speed up ray intersection queries,
 * - accelerate closest-point and distance computations,
 * - reduce complexity of geometry traversal for large triangle meshes.
 *
 * In Atlas, BVH implementations are primarily used by:
 * - triangle mesh geometry,
 * - collision detection,
 * - ray tracing and spatial queries.
 *
 * ## Design
 * The BVH interface is intentionally minimal and backend-agnostic:
 * - construction is performed on the host using triangle data,
 * - runtime queries are executed via a backend-portable geometry operator,
 * - concrete BVH implementations provide their own internal node layout.
 *
 * ## Geometry operator
 * The BVH exposes a @ref BvhGeometryOperator, which is:
 * - a value-type structure,
 * - safe to copy to device memory,
 * - used by runtime kernels for traversal and intersection.
 *
 * ## Ownership model
 * BVH objects are typically managed via:
 * - @ref atlas::BVHHostPtr for host-side ownership,
 * - @ref atlas::BVHDevicePtr for backend/device usage.
 *
 * ---
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>

namespace atlas::geometry {

/**
 * @brief Forward declaration of the triangle-mesh geometry operator.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct TriangleMeshGeometryOperator;

} // namespace atlas::geometry

namespace atlas::spatial {

/**
 * @brief Alias for the geometry operator produced by BVH implementations.
 *
 * @details
 * Currently resolves to @ref atlas::geometry::TriangleMeshGeometryOperator,
 * which encapsulates:
 * - triangle data,
 * - BVH node layout,
 * - traversal logic for spatial queries.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
using BvhGeometryOperator = atlas::geometry::TriangleMeshGeometryOperator<T>;

/**
 * @brief Abstract base class for Bounding Volume Hierarchy implementations.
 *
 * @details
 * @ref BoundingVolumeHierachy defines the interface required for spatial
 * acceleration structures operating on triangle meshes.
 *
 * Concrete implementations are responsible for:
 * - constructing a hierarchy of bounding volumes over input triangles,
 * - organizing triangle data into an efficient traversal structure,
 * - exposing a backend-portable geometry operator for runtime queries.
 *
 * ## Responsibilities
 * A derived BVH implementation must:
 * - implement @ref build to construct the hierarchy from triangle input,
 * - implement @ref make_geometry_operator to export a runtime query object.
 *
 * ## Typical workflow
 * 1. Construct a BVH instance.
 * 2. Call @ref build with triangle data.
 * 3. Retrieve a @ref BvhGeometryOperator for use in kernels or queries.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class BoundingVolumeHierachy {
public:
    /**
     * @brief Default constructor.
     */
    BoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     */
    virtual ~BoundingVolumeHierachy() = default;

    /**
     * @brief Build the BVH from triangle data.
     *
     * @details
     * Constructs the internal hierarchy using the provided triangle container.
     * The exact construction algorithm (e.g., SAH, median split) is defined by
     * the concrete implementation.
     *
     * @param triangles Host-side triangle container used as input geometry.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    build(const HostBuffer<TriangleContainer4<T>>& triangles)
        = 0;

    /**
     * @brief Create a backend-portable geometry operator.
     *
     * @details
     * Returns a value-type operator that encapsulates:
     * - triangle data,
     * - BVH nodes,
     * - traversal logic.
     *
     * This operator is intended for:
     * - device kernels,
     * - runtime spatial queries,
     * - collision and intersection routines.
     *
     * @return A geometry operator representing this BVH.
     */
    ATLAS_HOST virtual BvhGeometryOperator<T>
    make_geometry_operator() const = 0;
};

} // namespace atlas::spatial

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::spatial::BoundingVolumeHierachy.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
using BVH = spatial::BoundingVolumeHierachy<T>;

/**
 * @brief Host shared pointer alias for BVH.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
using BVHHostPtr = atlas::host_shared_ptr<BVH<T>>;

/**
 * @brief Device shared pointer alias for BVH.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
using BVHDevicePtr = atlas::device_shared_ptr<BVH<T>>;

} // namespace atlas