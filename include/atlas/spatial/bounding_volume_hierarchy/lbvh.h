#pragma once

/**
 * @file lbvh.h
 * @brief Declares a Morton-code-based linear bounding volume hierarchy.
 *
 * @details
 * This header defines @ref atlas::LinearBoundingVolumeHierachy, a
 * concrete BVH implementation that constructs a binary hierarchy from
 * Morton-sorted triangle primitives.
 *
 * A linear bounding volume hierarchy, abbreviated LBVH, is built by mapping
 * primitive centroids into a discrete Morton-code domain, sorting primitives by
 * their Morton codes, and then deriving the binary hierarchy from
 * longest-common-prefix relationships between adjacent sorted keys.
 *
 * ## Morton ordering
 *
 * For each primitive centroid @f$\mathbf{c}_i@f$, the centroid is normalized
 * into the global centroid bounding box:
 *
 * @f[
 *     \mathbf{p}_i
 *     =
 *     \frac{
 *         \mathbf{c}_i - \mathbf{c}_{\min}
 *     }{
 *         \mathbf{c}_{\max} - \mathbf{c}_{\min}
 *     }.
 * @f]
 *
 * Each normalized coordinate is clamped to @f$[0,1]@f$, quantized to an integer
 * grid, and then encoded into a 3D Morton code by interleaving coordinate bits:
 *
 * @f[
 *     m_i
 *     =
 *     \mathrm{interleave}(x_i, y_i, z_i).
 * @f]
 *
 * Sorting by @f$m_i@f$ places spatially nearby primitives close to each other in
 * memory, which enables linear-time topology construction.
 *
 * ## LBVH topology construction
 *
 * After sorting, internal nodes are derived from longest-common-prefix, LCP,
 * comparisons between neighboring Morton keys. The common prefix length between
 * two keys @f$a@f$ and @f$b@f$ is:
 *
 * @f[
 *     \mathrm{lcp}(a,b)
 *     =
 *     \mathrm{clz}(a \oplus b),
 * @f]
 *
 * where @f$\mathrm{clz}@f$ counts leading zero bits and @f$\oplus@f$ is bitwise
 * XOR.
 *
 * ## Stored data
 *
 * The hierarchy stores construction-time data on the host and traversal-time
 * data on the device:
 *
 * - host-side BVH nodes,
 * - host-side primitive index ordering,
 * - host-side primitive bounds,
 * - host-side primitive centroids,
 * - device-side BVH nodes,
 * - device-side primitive index ordering,
 * - device-side packed triangle primitives.
 *
 * ## Naming note
 *
 * The public class name uses the spelling `Hierachy`. This spelling is
 * preserved for compatibility with the existing Atlas API.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/random/seed.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

namespace atlas {

/**
 * @brief Morton-code-based linear bounding volume hierarchy for triangle primitives.
 *
 * @details
 * LinearBoundingVolumeHierachy builds a binary BVH over packed triangle
 * primitives. It uses Morton codes computed from primitive centroids to obtain a
 * spatial ordering, then constructs a binary hierarchy from longest-common-prefix
 * relationships between sorted keys.
 *
 * ## Build process
 *
 * Calling build() performs the following operations:
 *
 * 1. clear the previous hierarchy state,
 * 2. compute primitive AABBs,
 * 3. compute primitive centroids,
 * 4. compute the global centroid bounding box,
 * 5. encode each centroid as a Morton code,
 * 6. stable-sort primitives by Morton code,
 * 7. initialize one leaf node per primitive,
 * 8. construct internal-node topology from LCP relationships,
 * 9. compute internal-node bounds bottom-up,
 * 10. upload nodes, indices, and triangles to device buffers.
 *
 * ## Node layout
 *
 * For @f$N@f$ primitives, the node array contains:
 *
 * @f[
 *     2N - 1
 * @f]
 *
 * nodes when @f$N > 0@f$.
 *
 * The first @f$N - 1@f$ nodes are internal nodes, and the final @f$N@f$ nodes are
 * leaf nodes. The index of the @f$k@f$-th leaf is:
 *
 * @f[
 *     i_{\mathrm{leaf}}(k)
 *     =
 *     (N - 1) + k.
 * @f]
 *
 * ## Primitive ordering
 *
 * The original triangle buffer is not physically reordered. Instead, the
 * primitive order induced by Morton sorting is stored in @ref h_indices and
 * @ref d_indices. A leaf stores a range into this index buffer.
 *
 * In the current implementation, each primitive is initialized as one leaf:
 *
 * @f[
 *     \mathrm{leaf.count} = 1.
 * @f]
 *
 * @tparam T Floating-point scalar type used by triangle geometry, AABBs, and
 *           centroid computations.
 *
 * @note The hierarchy is built on the host and then copied to device buffers.
 * @note Morton-code construction uses at most 10 bits per coordinate, producing
 *       up to a 30-bit 3D Morton code.
 * @note The public class name keeps the existing `Hierachy` spelling for API
 *       compatibility.
 *
 * @see BoundingVolumeHierachy
 * @see BvhGeometryOperator
 * @see BVHNode
 * @see AABB
 */
template <typename T>
class LinearBoundingVolumeHierachy final : public BoundingVolumeHierachy<T> {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates an empty LBVH with default build parameters:
     *
     * - leaf size: @f$1@f$,
     * - Morton quantization bits: @f$10@f$,
     * - root index: @f$-1@f$.
     *
     * No nodes or primitive data are allocated until build() is called.
     */
    LinearBoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Allows the hierarchy to be destroyed through the base
     * BoundingVolumeHierachy interface.
     */
    ~LinearBoundingVolumeHierachy() override = default;

    /**
     * @brief Builds the LBVH from host-side triangle primitives.
     *
     * @details
     * This function rebuilds the hierarchy from scratch.
     *
     * For each input triangle primitive @f$P_i@f$, the implementation computes:
     *
     * @f[
     *     B_i = \mathrm{bounds}(P_i),
     *     \qquad
     *     \mathbf{c}_i = \mathrm{centroid}(P_i).
     * @f]
     *
     * The centroids are normalized against the global centroid bounds and encoded
     * as Morton codes. The primitive index buffer is then reordered according to
     * sorted Morton code order.
     *
     * If @p triangles is empty, the hierarchy is reset and remains empty.
     *
     * @param triangles Host-side packed triangle containers used as input
     *                  primitives.
     *
     * @note Existing hierarchy contents are discarded before construction.
     * @note The triangle data are copied into @ref d_triangles after build.
     * @note @ref h_indices and @ref d_indices store the Morton-sorted primitive
     *       order.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build(const HostBuffer<TriangleContainer4<T>>& triangles) override;

    /**
     * @brief Creates a lightweight geometry operator for traversal.
     *
     * @details
     * The returned @ref BvhGeometryOperator stores raw pointers to the hierarchy's
     * device buffers:
     *
     * - BVH node buffer,
     * - primitive index buffer,
     * - triangle primitive buffer,
     * - root node index.
     *
     * This operator can be passed to runtime geometry code without transferring
     * ownership of the LBVH container.
     *
     * @return Geometry operator referencing this hierarchy's device data.
     *
     * @warning The returned operator contains raw pointers into this object's
     *          device buffers. It remains valid only while the hierarchy object
     *          and its device buffers remain alive and unchanged.
     */
    ATLAS_HOST BvhGeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Clears all hierarchy data and marks the tree as empty.
     *
     * @details
     * This function clears all host and device buffers and resets the root index
     * to @f$-1@f$.
     *
     * After reset(), root() returns @f$-1@f$ until a non-empty build succeeds.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

    /**
     * @brief Sets the configured leaf size.
     *
     * @details
     * The requested value is clamped to at least @f$1@f$:
     *
     * @f[
     *     \mathrm{leaf\_size}
     *     =
     *     \max(1, \mathrm{input}).
     * @f]
     *
     * @param leaf_size Requested leaf size.
     *
     * @note The current LBVH implementation initializes one primitive per leaf.
     *       This value is retained as a configurable parameter for API symmetry
     *       and future grouped-leaf construction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_leaf_size(int leaf_size) noexcept;

    /**
     * @brief Sets the number of quantization bits per coordinate for Morton codes.
     *
     * @details
     * The requested value is clamped to:
     *
     * @f[
     *     1
     *     \le
     *     b
     *     \le
     *     10.
     * @f]
     *
     * Each coordinate is quantized into:
     *
     * @f[
     *     2^b
     * @f]
     *
     * possible levels before bit interleaving. Since the implementation uses a
     * 30-bit 3D Morton code, at most 10 bits are used per coordinate.
     *
     * @param morton_bits Requested number of quantization bits per coordinate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_morton_bits(int morton_bits) noexcept;

    /**
     * @brief Returns the configured leaf size.
     *
     * @return Configured leaf size.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    leaf_size() const noexcept;

    /**
     * @brief Returns the configured Morton quantization bit count.
     *
     * @return Number of bits used per coordinate for Morton-code quantization.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    morton_bits() const noexcept;

    /**
     * @brief Returns the root node index.
     *
     * @details
     * The root index identifies the top node in @ref h_nodes and @ref d_nodes.
     * A value of @f$-1@f$ means the hierarchy is empty or has not been built.
     *
     * For a tree with one primitive, the root is the single leaf node. For a tree
     * with more than one primitive, the current implementation uses internal node
     * @f$0@f$ as the root.
     *
     * @return Root node index.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    root() const noexcept;

    /**
     * @brief Returns host-side BVH node storage.
     *
     * @details
     * The returned buffer contains both internal and leaf nodes.
     *
     * @return Const reference to the host node buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<BVHNode<T>>&
    nodes() const noexcept;

    /**
     * @brief Returns host-side primitive index ordering.
     *
     * @details
     * The index buffer stores the Morton-sorted primitive order. Leaf nodes refer
     * to entries in this buffer through their `start` and `count` fields.
     *
     * @return Const reference to the host primitive index buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<int>&
    indices() const noexcept;

    /**
     * @brief Returns host-side primitive bounds.
     *
     * @details
     * Stores one AABB per input triangle primitive:
     *
     * @f[
     *     B_i = \mathrm{bounds}(P_i).
     * @f]
     *
     * These bounds are used to initialize leaf nodes and compute internal-node
     * bounds.
     *
     * @return Const reference to the primitive bounds buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<AABB<T>>&
    bounds() const noexcept;

    /**
     * @brief Returns host-side primitive centroids.
     *
     * @details
     * Stores one centroid per input triangle primitive:
     *
     * @f[
     *     \mathbf{c}_i = \mathrm{centroid}(P_i).
     * @f]
     *
     * Centroids are used for Morton-code generation.
     *
     * @return Const reference to the primitive centroid buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<Vector3<T>>&
    centroids() const noexcept;

    /**
     * @brief Returns device-side BVH node storage.
     *
     * @details
     * This buffer mirrors @ref h_nodes after build and is used by exported
     * geometry operators during runtime traversal.
     *
     * @return Const reference to the device node buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<BVHNode<T>>&
    device_nodes() const noexcept;

    /**
     * @brief Returns device-side primitive index ordering.
     *
     * @details
     * This buffer mirrors @ref h_indices after build.
     *
     * @return Const reference to the device primitive index buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    device_indices() const noexcept;

    /**
     * @brief Returns device-side packed triangle primitive storage.
     *
     * @details
     * Stores the input triangle primitives copied to device-accessible storage.
     * The exported geometry operator uses this buffer together with
     * @ref device_indices().
     *
     * @return Const reference to the device triangle buffer.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<TriangleContainer4<T>>&
    device_triangles() const noexcept;

private:
    /**
     * @brief Host-side BVH node storage.
     *
     * @details
     * For @f$N@f$ primitives, this buffer stores up to @f$2N-1@f$ nodes. Internal
     * nodes and leaf nodes are stored in one contiguous array.
     */
    HostBuffer<BVHNode<T>> h_nodes;

    /**
     * @brief Host-side Morton-sorted primitive index order.
     *
     * @details
     * Values in this buffer index into the original triangle array. Leaf nodes
     * reference contiguous intervals in this buffer.
     */
    HostBuffer<int> h_indices;

    /**
     * @brief Host-side primitive bounds.
     *
     * @details
     * Stores one AABB per input triangle. These bounds are used for leaf bounds
     * and bottom-up internal bounds.
     */
    HostBuffer<AABB<T>> h_prim_bounds;

    /**
     * @brief Host-side primitive centroids.
     *
     * @details
     * Stores one centroid per input triangle. These centroids are normalized and
     * converted to Morton codes during construction.
     */
    HostBuffer<Vector3<T>> h_centroids;

    /**
     * @brief Root node index.
     *
     * @details
     * Stores the root node index in @ref h_nodes and @ref d_nodes. The value is
     * @f$-1@f$ when the hierarchy is empty.
     */
    int _root = -1;

    /**
     * @brief Configured leaf size.
     *
     * @details
     * This value is clamped to at least @f$1@f$ by set_leaf_size().
     *
     * @note The current LBVH build path creates one primitive per leaf.
     */
    int _leaf_size = 1;

    /**
     * @brief Morton quantization bits per coordinate.
     *
     * @details
     * This value controls how finely normalized centroid coordinates are
     * quantized before bit interleaving.
     *
     * The valid range enforced by set_morton_bits() is:
     *
     * @f[
     *     1 \le \_morton\_bits \le 10.
     * @f]
     */
    int _morton_bits = 10;

    /**
     * @brief Device-side BVH node storage.
     *
     * @details
     * Mirrors the completed host node buffer for runtime traversal.
     */
    DeviceBuffer<BVHNode<T>> d_nodes;

    /**
     * @brief Device-side primitive index ordering.
     *
     * @details
     * Mirrors @ref h_indices for device-side traversal.
     */
    DeviceBuffer<int> d_indices;

    /**
     * @brief Device-side packed triangle primitive storage.
     *
     * @details
     * Stores the packed triangle primitives referenced by the BVH.
     */
    DeviceBuffer<TriangleContainer4<T>> d_triangles;

private:
    /**
     * @brief Assigns fast-winding aggregate data for a single primitive leaf.
     */
    static void
    assign_solid_angle_moment(BVHNode<T>& node,
                              const TriangleContainer4<T>& triangle) noexcept;

    /**
     * @brief Merges fast-winding aggregate data from two child nodes.
     */
    static void
    merge_solid_angle_moment(BVHNode<T>& node,
                             const BVHNode<T>& left,
                             const BVHNode<T>& right) noexcept;

    /**
     * @brief Computes the node index of the @p k-th leaf node.
     *
     * @details
     * In the LBVH array layout, internal nodes occupy indices:
     *
     * @f[
     *     [0, N - 1)
     * @f]
     *
     * and leaves occupy:
     *
     * @f[
     *     [N - 1, 2N - 1).
     * @f]
     *
     * Therefore, the @p k-th leaf is stored at:
     *
     * @f[
     *     i_{\mathrm{leaf}} = (N - 1) + k.
     * @f]
     *
     * @param k Leaf ordinal in Morton-sorted order.
     * @param n Total number of leaves.
     * @return Node index of the selected leaf.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    leaf_node_index(int k, int n) noexcept;

    /**
     * @brief Expands coordinate bits for 3D Morton interleaving.
     *
     * @details
     * This helper takes the lower bits of @p v and inserts two zero bits between
     * neighboring input bits. The result can be combined with similarly expanded
     * y and z coordinates to form a 3D Morton code.
     *
     * Conceptually, for input bits:
     *
     * @f[
     *     v =
     *     b_9 b_8 \dots b_1 b_0,
     * @f]
     *
     * the expanded form places them at every third bit:
     *
     * @f[
     *     b_9 00\ b_8 00\ \dots\ b_1 00\ b_0.
     * @f]
     *
     * @param v Input integer coordinate.
     * @return Bit-expanded value suitable for Morton interleaving.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE unsigned
    expand_bits(unsigned v) noexcept;

    /**
     * @brief Computes a 3D Morton code for a point inside centroid bounds.
     *
     * @details
     * The input point @p p is normalized relative to centroid bounds @p cb:
     *
     * @f[
     *     n_x =
     *     \frac{p_x - l_x}{u_x - l_x},
     *     \qquad
     *     n_y =
     *     \frac{p_y - l_y}{u_y - l_y},
     *     \qquad
     *     n_z =
     *     \frac{p_z - l_z}{u_z - l_z}.
     * @f]
     *
     * Degenerate axes are mapped to @f$0@f$. The normalized coordinates are
     * clamped to @f$[0,1]@f$ and quantized as:
     *
     * @f[
     *     q_x =
     *     \left\lfloor
     *         n_x(2^b - 1) + 0.5
     *     \right\rfloor,
     * @f]
     *
     * and similarly for @f$q_y@f$ and @f$q_z@f$, where @f$b@f$ is @p bits.
     *
     * The final code is:
     *
     * @f[
     *     m =
     *     (\mathrm{expand}(q_x) \ll 2)
     *     \ |\
     *     (\mathrm{expand}(q_y) \ll 1)
     *     \ |\
     *     \mathrm{expand}(q_z).
     * @f]
     *
     * @param p Point to encode, usually a primitive centroid.
     * @param cb Bounding box of all primitive centroids.
     * @param bits Number of quantization bits per coordinate.
     * @return 3D Morton code.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE uint32_t
    morton3(const Vector3<T>& p, const AABB<T>& cb, int bits) const noexcept;

    /**
     * @brief Counts leading zero bits in a 32-bit unsigned integer.
     *
     * @param x Input value.
     * @return Number of leading zero bits. Returns @f$32@f$ when @p x is zero.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz32(uint32_t x) noexcept;

    /**
     * @brief Counts leading zero bits in a 64-bit unsigned integer.
     *
     * @param x Input value.
     * @return Number of leading zero bits. Returns @f$64@f$ when @p x is zero.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    clz64(uint64_t x) noexcept;

    /**
     * @brief Computes the longest-common-prefix metric between two sorted keys.
     *
     * @details
     * The metric is computed as:
     *
     * @f[
     *     \delta(i,j)
     *     =
     *     \mathrm{clz}
     *     \left(
     *         k_i \oplus k_j
     *     \right),
     * @f]
     *
     * where @f$k_i@f$ and @f$k_j@f$ are 64-bit extended Morton keys. Out-of-range
     * neighbor positions return @f$-1@f$.
     *
     * This value is used during LBVH topology construction to determine the range
     * direction and range extent for each internal node.
     *
     * @param keys Sorted 64-bit extended Morton keys.
     * @param n Number of keys.
     * @param i First key position.
     * @param j Second key position.
     * @return Longest-common-prefix length, or @f$-1@f$ for an invalid neighbor.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    delta_lcp(const HostBuffer<uint64_t>& keys, int n, int i, int j) noexcept;

    /**
     * @brief Finds a split position inside a Morton-code range.
     *
     * @details
     * Given a sorted Morton-code range @f$[\mathrm{first}, \mathrm{last}]@f$,
     * this function finds the last index whose prefix with the first code remains
     * longer than the common prefix of the full range.
     *
     * Let:
     *
     * @f[
     *     p_{\mathrm{range}}
     *     =
     *     \mathrm{clz}
     *     \left(
     *         c_{\mathrm{first}}
     *         \oplus
     *         c_{\mathrm{last}}
     *     \right).
     * @f]
     *
     * The split is found by binary search over the range using prefix lengths
     * relative to @f$c_{\mathrm{first}}@f$.
     *
     * If the first and last Morton codes are identical, the function returns the
     * midpoint:
     *
     * @f[
     *     \left\lfloor
     *         \frac{\mathrm{first} + \mathrm{last}}{2}
     *     \right\rfloor.
     * @f]
     *
     * @param codes Sorted 32-bit Morton codes.
     * @param first First index of the inclusive range.
     * @param last Last index of the inclusive range.
     * @return Split position separating the two child ranges.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE int
    find_split(const HostBuffer<uint32_t>& codes, int first, int last) noexcept;
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for the LBVH implementation.
 *
 * @details
 * Exposes @ref atlas::LinearBoundingVolumeHierachy in the top-level
 * atlas namespace.
 *
 * @tparam T Floating-point scalar used by the geometry.
 */
template <typename T>
using LBVH = LinearBoundingVolumeHierachy<T>;

} // namespace atlas

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.hpp>
