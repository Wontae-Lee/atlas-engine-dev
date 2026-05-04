#pragma once

/**
 * @file sah_bvh.h
 * @brief Declares a surface-area-heuristic (SAH) bounding volume hierarchy builder, storage container, and query-export interface.
 *
 * @details
 * This header defines:
 * - @ref atlas::spatial::sah::Bin, a helper structure used during SAH binning,
 * - @ref atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy, a concrete
 *   BVH implementation that partitions primitives using the surface area heuristic.
 *
 * ## Purpose
 * A SAH-based BVH attempts to reduce traversal cost by recursively partitioning
 * primitives so that the expected cost of visiting child nodes is minimized.
 * This usually produces higher-quality hierarchies than purely Morton-order-based
 * builders, especially for:
 * - ray tracing,
 * - closest-point queries,
 * - distance evaluation,
 * - broad-phase pruning over irregular meshes.
 *
 * ## Construction strategy
 * A typical SAH build proceeds as follows:
 * 1. compute primitive bounds and centroids,
 * 2. choose a split axis and candidate split positions,
 * 3. evaluate split cost using surface area and primitive counts,
 * 4. partition primitives into left/right subsets,
 * 5. recurse until a leaf criterion is met.
 *
 * This implementation exposes configuration parameters for:
 * - maximum primitives per leaf,
 * - number of bins used for split evaluation.
 *
 * ## Stored data
 * The hierarchy maintains:
 * - host-side node storage,
 * - host-side primitive index order,
 * - host-side primitive bounds,
 * - host-side primitive centroids,
 * - device-side node storage,
 * - device-side primitive index order,
 * - device-side primitive triangle storage.
 *
 * ## Query export
 * After construction, the hierarchy can export a backend-portable
 * @ref BvhGeometryOperator via @ref make_geometry_operator so downstream runtime
 * systems can perform accelerated traversal and intersection.
 *
 * ## Naming note
 * The class name uses the spelling `Hierachy` in the current API and is preserved
 * for compatibility with the surrounding codebase.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the geometry.
 */
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

namespace sah {

    /**
     * @brief Temporary bin used during surface-area-heuristic split evaluation.
     *
     * @details
     * Each bin accumulates:
     * - the bounding box of primitives assigned to the bin,
     * - the number of primitives assigned to the bin.
     *
     * Bins are commonly used to approximate the SAH cost efficiently without testing
     * every possible primitive split individually.
     *
     * ---
     *
     * @tparam T Floating-point scalar type used for bounds.
     */
    template <typename T>
    struct Bin {
        /**
         * @brief Bounding box enclosing all primitives assigned to this bin.
         */
        AABB<T> bounds;

        /**
         * @brief Number of primitives assigned to this bin.
         */
        int count = 0;
    };

} // namespace sah

/**
 * @brief Bounding volume hierarchy built using the surface area heuristic (SAH).
 *
 * @details
 * @ref SurfaceAreaHeuristicBoundingVolumeHierachy is a concrete implementation of
 * @ref BoundingVolumeHierachy that recursively partitions triangle primitives
 * using the surface area heuristic.
 *
 * ## High-level workflow
 * During @ref build, the implementation typically:
 * - computes primitive bounds,
 * - computes primitive centroids,
 * - recursively chooses low-cost SAH splits,
 * - creates internal and leaf nodes,
 * - stores the resulting hierarchy on the host,
 * - uploads traversal data to device buffers.
 *
 * ## Runtime use
 * Once built, the hierarchy can export a @ref BvhGeometryOperator for:
 * - accelerated ray tracing,
 * - nearest-surface queries,
 * - signed-distance and inside/outside support via mesh traversal.
 *
 * ## Configuration
 * Two important build parameters are exposed:
 * - @ref leaf_size : maximum primitives stored in a leaf,
 * - @ref num_of_bins : number of bins used to approximate SAH costs.
 *
 * Higher bin counts may improve split quality but generally increase build cost.
 *
 * ## Host/device data split
 * The class stores both:
 * - host-side construction and inspection buffers,
 * - device-side traversal buffers for runtime usage.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
class SurfaceAreaHeuristicBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty SAH BVH with default build parameters and no constructed nodes.
     */
    SurfaceAreaHeuristicBoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     */
    ~SurfaceAreaHeuristicBoundingVolumeHierachy() override = default;

    /**
     * @brief Build the SAH BVH from triangle primitives.
     *
     * @details
     * Consumes the supplied packed triangle containers, computes primitive bounds
     * and centroids, recursively partitions them using SAH, and updates device-side
     * buffers for runtime traversal.
     *
     * @param triangles Host-side triangle containers used as input primitives.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /**
     * @brief Create a backend-portable geometry operator for this hierarchy.
     *
     * @details
     * Returns a @ref BvhGeometryOperator containing the triangle and BVH data
     * required for accelerated runtime traversal.
     *
     * @return Geometry operator bound to this hierarchy.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhGeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Reset the hierarchy to an empty state.
     *
     * @details
     * Clears or reinitializes stored hierarchy state according to the
     * implementation policy in `sah_bvh.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    /**
     * @brief Set the maximum number of primitives per leaf node.
     *
     * @param leaf_size Maximum leaf occupancy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    /**
     * @brief Set the number of bins used for SAH split evaluation.
     *
     * @param num_bins Number of bins used to approximate split costs.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_num_of_bins(int num_bins) noexcept;

    /**
     * @brief Return the configured maximum primitives per leaf.
     *
     * @return Leaf size.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /**
     * @brief Return the configured number of SAH bins.
     *
     * @return Number of bins used for split evaluation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    num_of_bins() const noexcept;

    /**
     * @brief Return the root node index.
     *
     * @details
     * A negative value typically indicates that the hierarchy has not yet been built.
     *
     * @return Root node index.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    /**
     * @brief Return const access to host-side BVH nodes.
     *
     * @return Host buffer of BVH nodes.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    /**
     * @brief Return const access to host-side primitive index ordering.
     *
     * @return Host buffer of primitive indices.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    /**
     * @brief Return const access to host-side primitive bounds.
     *
     * @return Host buffer of primitive axis-aligned bounds.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    /**
     * @brief Return const access to host-side primitive centroids.
     *
     * @return Host buffer of primitive centroids.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    /**
     * @brief Return const access to device-side BVH nodes.
     *
     * @return Device buffer of BVH nodes.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    /**
     * @brief Return const access to device-side primitive index ordering.
     *
     * @return Device buffer of primitive indices.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    /**
     * @brief Return const access to device-side primitive triangle storage.
     *
     * @return Device buffer of packed triangle containers.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    /**
     * @brief Host-side BVH node storage.
     *
     * @details
     * Stores the constructed hierarchy topology and node bounds on the host.
     */
    HostBuffer<BVHNode<T>> h_nodes;

    /**
     * @brief Host-side primitive index order.
     *
     * @details
     * Stores the primitive permutation induced by recursive partitioning.
     */
    HostBuffer<int> h_indices;

    /**
     * @brief Host-side primitive bounds.
     *
     * @details
     * Stores one axis-aligned bounding box per primitive.
     */
    HostBuffer<AABB<T>> h_prim_bounds;

    /**
     * @brief Host-side primitive centroids.
     *
     * @details
     * Stores one centroid per primitive, used during SAH split evaluation.
     */
    HostBuffer<Vector3<T>> h_centroids;

    /**
     * @brief Root node index.
     *
     * @details
     * Negative when the hierarchy has not yet been built or is empty.
     */
    int _root = -1;

    /**
     * @brief Maximum number of primitives per leaf.
     *
     * @details
     * Controls the stopping criterion for recursive partitioning.
     */
    int _leaf_size = 32;

    /**
     * @brief Number of bins used for SAH split approximation.
     *
     * @details
     * Larger values can improve split quality at the cost of additional build work.
     */
    int _num_of_bins = 100;

    /**
     * @brief Device-side BVH node storage.
     *
     * @details
     * Used by runtime traversal and exported geometry operators.
     */
    DeviceBuffer<BVHNode<T>> d_nodes;

    /**
     * @brief Device-side primitive index order.
     *
     * @details
     * Maps traversal ordering to primitive indices.
     */
    DeviceBuffer<int> d_indices;

    /**
     * @brief Device-side packed primitive storage.
     *
     * @details
     * Stores the triangle primitives referenced by the hierarchy.
     */
    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    /**
     * @brief Recursively build a subtree over the primitive interval `[start, end)`.
     *
     * @details
     * Partitions the specified primitive range using the surface area heuristic,
     * creates internal or leaf nodes as appropriate, updates @p node_count, and
     * returns the node index of the constructed subtree root.
     *
     * @param start Inclusive start index of the primitive interval.
     * @param end Exclusive end index of the primitive interval.
     * @param node_count Running count of allocated nodes.
     * @return Node index of the constructed subtree root.
     */
    int
    build_recursive(int start, int end, int& node_count);
};

} // namespace atlas::spatial

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy.
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
using SAHBVH = spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.hpp>