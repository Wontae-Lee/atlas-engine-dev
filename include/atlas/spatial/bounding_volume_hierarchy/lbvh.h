#pragma once

/**
 * @file lbvh.h
 * @brief Declares a linear bounding volume hierarchy (LBVH) builder, storage container, and query-export interface.
 *
 * @details
 * This header defines @ref atlas::spatial::LinearBoundingVolumeHierachy, a
 * concrete Bounding Volume Hierarchy (BVH) implementation that organizes
 * triangle primitives using Morton-code ordering.
 *
 * ## Purpose
 * A linear BVH is a spatial acceleration structure designed to improve the
 * performance of geometric queries over triangle data, especially:
 * - ray intersection,
 * - closest-point search,
 * - distance queries,
 * - broad-phase pruning for complex meshes.
 *
 * Compared to naïve per-triangle traversal, an LBVH accelerates runtime queries
 * by hierarchically grouping primitives inside bounding volumes and traversing
 * only relevant subtrees.
 *
 * ## Construction strategy
 * This implementation follows the LBVH approach:
 * 1. compute primitive bounds and centroids,
 * 2. normalize centroids inside a global bounding box,
 * 3. quantize those positions,
 * 4. encode them using Morton codes,
 * 5. sort primitives by Morton order,
 * 6. build a binary hierarchy from longest-common-prefix relationships,
 * 7. upload the resulting node and primitive-order data to device buffers.
 *
 * ## Stored data
 * The class maintains:
 * - host-side node storage,
 * - host-side primitive index order,
 * - host-side primitive bounds,
 * - host-side primitive centroids,
 * - device-side node storage,
 * - device-side primitive index order,
 * - device-side primitive triangle storage.
 *
 * ## Query export
 * Once constructed, the LBVH can export a backend-portable
 * @ref BvhGeometryOperator through @ref make_geometry_operator so downstream
 * runtime systems can perform accelerated traversal.
 *
 * ## Configuration
 * Two important build parameters are exposed:
 * - @ref leaf_size : maximum number of primitives per leaf node,
 * - @ref morton_bits : number of quantization bits used in Morton encoding.
 *
 * ## Naming note
 * The class name uses the spelling `Hierachy` in the current API and is kept
 * unchanged for compatibility with the surrounding codebase.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the geometry.
 */
#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas::spatial {

/**
 * @brief Linear bounding volume hierarchy built from Morton ordering.
 *
 * @details
 * @ref LinearBoundingVolumeHierachy is a concrete implementation of
 * @ref BoundingVolumeHierachy that builds a BVH over triangle primitives using
 * a linear Morton-code-based construction strategy.
 *
 * ## High-level workflow
 * During @ref build, the implementation typically:
 * - computes primitive AABBs,
 * - computes primitive centroids,
 * - computes Morton codes from those centroids,
 * - sorts primitives by Morton code,
 * - constructs an LBVH topology,
 * - computes internal-node bounds,
 * - uploads relevant traversal data to device storage.
 *
 * ## Runtime use
 * After construction, the hierarchy may be queried indirectly through the
 * exported @ref BvhGeometryOperator, which carries:
 * - triangle data,
 * - node data,
 * - primitive ordering,
 * and supports accelerated spatial queries.
 *
 * ## Host/device data split
 * The class stores both:
 * - host-side buffers used during construction and inspection,
 * - device-side buffers used during runtime traversal.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
class LinearBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty LBVH with default build parameters and no constructed nodes.
     */
    LinearBoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     */
    ~LinearBoundingVolumeHierachy() override = default;

    /**
     * @brief Build the LBVH from triangle primitives.
     *
     * @details
     * Consumes the supplied packed triangle containers, constructs the host-side
     * hierarchy and associated primitive ordering, and updates device-side buffers
     * for runtime query use.
     *
     * @param triangles Host-side triangle containers used as input primitives.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /**
     * @brief Create a backend-portable geometry operator for this LBVH.
     *
     * @details
     * Returns a @ref BvhGeometryOperator containing the triangle and BVH data
     * required for accelerated query traversal.
     *
     * @return Geometry operator bound to this hierarchy.
     */
    ATLAS_HOST BvhGeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Reset the hierarchy to an empty state.
     *
     * @details
     * Clears or reinitializes stored hierarchy state according to the
     * implementation policy in `lbvh.hpp`.
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
     * @brief Set the number of quantization bits used for Morton encoding.
     *
     * @param morton_bits Number of bits used per coordinate for Morton codes.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_morton_bits(int morton_bits) noexcept;

    /**
     * @brief Return the configured maximum primitives per leaf.
     *
     * @return Leaf size.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /**
     * @brief Return the configured Morton quantization bit count.
     *
     * @return Number of Morton bits.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    morton_bits() const noexcept;

    /**
     * @brief Return the root node index.
     *
     * @details
     * A negative value typically indicates that no valid hierarchy root is present.
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
     * @details
     * The index buffer stores the primitive order induced by Morton sorting and
     * LBVH construction.
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
     * Stores the primitive permutation induced by Morton sorting and BVH construction.
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
     * Stores one centroid per primitive, typically used during Morton encoding.
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
     * Controls the stopping criterion for leaf construction.
     */
    int _leaf_size = 1;

    /**
     * @brief Quantization bits used for Morton-code generation.
     *
     * @details
     * Higher values provide finer centroid discretization before Morton encoding.
     */
    int _morton_bits = 10;

    /**
     * @brief Device-side BVH node storage.
     *
     * @details
     * Used by runtime traversal and geometry operators.
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
     * @brief Compute the node index corresponding to the `k`-th leaf in an LBVH layout.
     *
     * @param k Leaf ordinal.
     * @param n Total number of primitives or leaves, depending on layout convention.
     * @return Leaf node index in the node array.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    leaf_node_index(int k, int n) noexcept;

    /**
     * @brief Expand the bits of an integer for Morton interleaving.
     *
     * @details
     * Used as part of 3D Morton-code construction.
     *
     * @param v Input integer value.
     * @return Bit-expanded value suitable for interleaving.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE unsigned
    expand_bits(unsigned v) noexcept;

    /**
     * @brief Compute a 3D Morton code for a point inside a centroid bounding box.
     *
     * @details
     * The point is quantized relative to the centroid bounds and encoded using
     * the configured number of Morton bits.
     *
     * @param p Point to encode.
     * @param cb Bounding box of all primitive centroids.
     * @param bits Number of quantization bits.
     * @return Morton code for the point.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE uint32_t
    morton3(const Vector3<T>& p, const AABB<T>& cb, int bits) const noexcept;

    /**
     * @brief Count leading zeros in a 32-bit unsigned integer.
     *
     * @param x Input value.
     * @return Number of leading zero bits.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz32(uint32_t x) noexcept;

    /**
     * @brief Count leading zeros in a 64-bit unsigned integer.
     *
     * @param x Input value.
     * @return Number of leading zero bits.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz64(uint64_t x) noexcept;

    /**
     * @brief Compute the longest-common-prefix metric between two key positions.
     *
     * @details
     * This helper is typically used in LBVH topology construction to determine
     * range direction and split behavior.
     *
     * @param keys Morton or extended hierarchy keys.
     * @param n Number of keys.
     * @param i First key position.
     * @param j Second key position.
     * @return Longest-common-prefix measure.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    /**
     * @brief Find a split position inside a Morton-ordered primitive range.
     *
     * @details
     * Used during LBVH construction to divide a key range into two child ranges.
     *
     * @param codes Morton codes in sorted order.
     * @param first First index of the range.
     * @param last Last index of the range.
     * @return Split position separating the two subranges.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};

} // namespace atlas::spatial

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::spatial::LinearBoundingVolumeHierachy.
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
using LBVH = spatial::LinearBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.hpp>