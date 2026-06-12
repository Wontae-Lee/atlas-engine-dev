#pragma once

/**
 * @file bvh.h
 * @brief Declares the abstract Bounding Volume Hierarchy interface used for triangle-mesh spatial acceleration.
 *
 * @details
 * This header defines the common BVH interface used by Atlas triangle-mesh
 * acceleration structures.
 *
 * A Bounding Volume Hierarchy, abbreviated BVH, organizes geometric primitives
 * into a tree of bounding volumes. Each internal node bounds the primitives
 * contained by its child subtrees, while each leaf node references one or more
 * primitives.
 *
 * For a set of triangle primitives
 *
 * @f[
 *     P = \{P_0, P_1, \dots, P_{N-1}\},
 * @f]
 *
 * a BVH stores bounding boxes that conservatively enclose subsets of those
 * primitives. During traversal, expensive primitive tests can be skipped when a
 * query does not intersect a node bound.
 *
 * ## Purpose
 *
 * BVHs are used to accelerate:
 *
 * - ray-triangle intersection,
 * - closest-point queries,
 * - distance queries,
 * - collision detection,
 * - broad-phase pruning for triangle meshes.
 *
 * Instead of testing every primitive, traversal first tests bounding volumes.
 * If a query misses a node bound, the whole subtree can be rejected.
 *
 * ## Interface design
 *
 * The interface is intentionally minimal:
 *
 * - build() constructs the hierarchy from host-side triangle data,
 * - make_geometry_operator() exports a lightweight runtime operator.
 *
 * Concrete implementations decide their own construction algorithm and storage
 * layout. For example, one implementation may use Morton-code ordering, while
 * another may use a surface-area heuristic.
 *
 * ## Geometry operator export
 *
 * The exported @ref atlas::BvhGeometryOperator is a value-type runtime
 * view over the constructed hierarchy. It usually contains raw pointers or
 * lightweight references to device-side node, index, and triangle buffers.
 *
 * This separation allows the owning BVH object to manage construction and memory,
 * while kernels or query routines operate through a compact traversal operator.
 *
 * ## Naming note
 *
 * The public class name uses the spelling `Hierachy`. This spelling is preserved
 * for compatibility with the existing Atlas API.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/memory/memory.h>

namespace atlas {

/**
 * @brief Forward declaration of the triangle-mesh geometry operator.
 *
 * @details
 * Concrete BVH implementations export this operator type through
 * @ref atlas::BvhGeometryOperator. The operator is expected to provide
 * the runtime traversal and query functionality for BVH-backed triangle meshes.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 */
template <typename T>
struct TriangleMeshGeometryOperator;

} // namespace atlas

namespace atlas {

/**
 * @brief Geometry-operator type exported by BVH implementations.
 *
 * @details
 * This alias currently resolves to
 * @ref atlas::TriangleMeshGeometryOperator.
 *
 * A BVH implementation returns this type from make_geometry_operator() so that
 * downstream systems can perform accelerated spatial queries without depending
 * on the concrete BVH container type.
 *
 * Conceptually, the exported operator represents:
 *
 * - triangle primitive storage,
 * - BVH node storage,
 * - primitive index ordering,
 * - root-node information,
 * - traversal logic for geometric queries.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 *
 * @note The operator is a runtime view of BVH data. Its validity depends on the
 *       lifetime and stability of the buffers owned by the concrete BVH object.
 */
template <typename T>
using BvhGeometryOperator = atlas::TriangleMeshGeometryOperator<T>;

/**
 * @brief Abstract base class for triangle-mesh Bounding Volume Hierarchies.
 *
 * @details
 * BoundingVolumeHierachy defines the common interface implemented by concrete
 * BVH builders in Atlas.
 *
 * A BVH implementation takes triangle primitives as input and constructs an
 * acceleration structure that can later be exported as a
 * @ref BvhGeometryOperator.
 *
 * ## Responsibilities of derived classes
 *
 * A derived class must provide:
 *
 * - a build() implementation that constructs the hierarchy,
 * - a make_geometry_operator() implementation that exports the runtime operator.
 *
 * The base class does not prescribe:
 *
 * - node layout,
 * - split heuristic,
 * - leaf size policy,
 * - host/device buffer organization,
 * - traversal implementation.
 *
 * These details are owned by the concrete BVH implementation.
 *
 * ## Typical usage
 *
 * @code
 * atlas::LBVH<float> bvh;
 * bvh.build(triangles);
 *
 * auto op = bvh.make_geometry_operator();
 * @endcode
 *
 * The returned operator can then be passed to geometry query code.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 *
 * @note Construction is host-side in the current interface.
 * @note Runtime traversal is performed through the exported geometry operator.
 * @note The class uses a virtual destructor so derived BVH objects can be safely
 *       destroyed through a base pointer.
 */
template <typename T>
class BoundingVolumeHierachy {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs the abstract base portion of a BVH object.
     */
    BoundingVolumeHierachy() = default;

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Enables safe destruction through a pointer or reference to
     * BoundingVolumeHierachy.
     */
    virtual ~BoundingVolumeHierachy() = default;

    /**
     * @brief Builds the BVH from host-side triangle data.
     *
     * @details
     * This function constructs or reconstructs the concrete hierarchy using the
     * supplied packed triangle primitives.
     *
     * The input triangles represent the primitive set:
     *
     * @f[
     *     P = \{P_0, P_1, \dots, P_{N-1}\}.
     * @f]
     *
     * A concrete implementation typically computes primitive bounds:
     *
     * @f[
     *     B_i = \mathrm{bounds}(P_i),
     * @f]
     *
     * then groups those bounds into a tree suitable for accelerated traversal.
     *
     * The exact construction algorithm is implementation-defined by the derived
     * class. Examples include:
     *
     * - linear BVH construction from Morton codes,
     * - surface-area-heuristic construction,
     * - median or spatial split construction.
     *
     * @param triangles Host-side packed triangle container used as input
     *                  geometry.
     *
     * @note Implementations usually discard any previously built hierarchy before
     *       constructing a new one.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    build(const HostBuffer<TriangleContainer4<T>>& triangles)
        = 0;

    /**
     * @brief Exports a runtime geometry operator for accelerated traversal.
     *
     * @details
     * This function creates a value-type operator that represents the constructed
     * BVH in a form suitable for query code.
     *
     * The exported operator generally provides access to:
     *
     * - triangle primitive data,
     * - BVH node data,
     * - primitive index data,
     * - the root node,
     * - traversal and intersection routines.
     *
     * @return Geometry operator representing this BVH.
     *
     * @warning The returned operator may contain raw pointers or lightweight
     *          references into buffers owned by the concrete BVH object. It must
     *          not outlive the BVH object or be used after the BVH is rebuilt or
     *          reset.
     */
    ATLAS_HOST virtual BvhGeometryOperator<T>
    make_geometry_operator() const = 0;
};

} // namespace atlas

namespace atlas {

/**
 * @brief Convenience alias for atlas::BoundingVolumeHierachy.
 *
 * @details
 * Exposes the abstract BVH interface in the top-level atlas namespace.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 */
template <typename T>
using BVH = BoundingVolumeHierachy<T>;

/**
 * @brief Host shared pointer alias for BVH objects.
 *
 * @details
 * Represents shared host ownership of an object implementing the abstract BVH
 * interface.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 */
template <typename T>
using BVHHostPtr = atlas::host_shared_ptr<BVH<T>>;

/**
 * @brief Device shared pointer alias for BVH objects.
 *
 * @details
 * Represents device-managed shared ownership of an object implementing the
 * abstract BVH interface.
 *
 * @tparam T Floating-point scalar type used by triangle geometry and queries.
 */
template <typename T>
using BVHDevicePtr = atlas::device_shared_ptr<BVH<T>>;

} // namespace atlas