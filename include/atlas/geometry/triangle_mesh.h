#pragma once

/**
 * @file triangle_mesh.h
 * @brief Triangle mesh geometry primitive (host-side storage) with optional BVH acceleration and builder.
 *
 * @details
 * This header defines @ref atlas::geometry::TriangleMesh, a geometry primitive representing a
 * collection of triangles stored on the host, with optional acceleration via a
 * Bounding Volume Hierarchy (BVH).
 *
 * `TriangleMesh` implements the @ref atlas::geometry::Geometry interface and provides:
 * - host-side storage of triangles (as @ref HostBuffer of @ref TriangleContainer4),
 * - query and trace operator construction for interop with generic systems,
 * - OBJ loading (host-only),
 * - closest point/normal and signed distance evaluation,
 * - centroid and AABB computation,
 * - validity checks,
 * - lazy BVH construction to accelerate repeated spatial queries.
 *
 * ## Host vs device design intent
 * The `TriangleMesh` object owns triangle data on the host and can build a host BVH.
 * Query/trace operators can then be produced to enable:
 * - host-side geometry queries with optional acceleration,
 * - preparation for upload to device-side structures (depending on how operators/pointers are handled).
 *
 * @warning
 * A @ref TriangleMeshQueryOperator (see `query_operator.h`) that stores raw pointers to contiguous
 * vertex/index arrays is a different representation than the `TriangleContainer4<T>` storage here.
 * Ensure that your `make_query_operator()` implementation matches the expected memory layout for the
 * operator it returns.
 *
 * ## BVH behavior
 * The BVH is built lazily:
 * - `bvh_built == false` initially
 * - `ensure_bvh()` creates/builds a BVH the first time it is needed
 *
 * The exact criteria for "needed" is implementation-defined (for example: used by tracing, or used by
 * closest-point queries over large meshes).
 *
 * ## Triangle storage
 * The mesh stores triangles as @ref TriangleContainer4<T> elements. This typically encodes:
 * - three vertices (a, b, c) plus an extra cached value (often normal or padding), depending on your container design.
 *
 * Confirm the exact layout in `atlas/container/container.h` and the implementation in `triangle_mesh.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @note
 * - This type is primarily host-oriented (OBJ loading, BVH building).
 * - Thread-safety depends on how @ref HostBuffer and BVH internals are used; this class does not
 *   imply internal locking.
 */

#include <atlas/container/container.h>
#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <string>
#include <type_traits>

namespace atlas ::geometry {

/**
 * @brief Triangle mesh geometry composed of many triangles with optional BVH acceleration.
 *
 * @details
 * `TriangleMesh` represents a collection of triangles and supports standard geometry queries
 * required by @ref Geometry:
 * - closest point / closest normal
 * - signed distance
 * - centroid
 * - bounds
 *
 * ### Performance considerations
 * A naive closest-point or distance query over a mesh is O(N) in triangle count.
 * To improve performance, this class can build and use a BVH:
 * - tracing and closest-feature queries can reduce work by pruning triangles whose bounding
 *   boxes cannot contain a better answer.
 *
 * ### Lifetime and mutability
 * If you mutate the triangle buffer after a BVH is built, the BVH becomes stale.
 * Implementations typically:
 * - mark `bvh_built = false` when triangles change, or
 * - rebuild automatically on next query.
 *
 * Confirm exact behavior in `triangle_mesh.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class TriangleMesh final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "TriangleMesh requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref TriangleMesh.
     *
     * @details
     * See @ref TriangleMesh<T>::Builder for supported initialization routes:
     * - provide triangles directly
     * - load from OBJ
     * - build by value or as host_shared_ptr
     */
    class Builder;

public:
    /**
     * @brief Host-side triangle storage.
     *
     * @details
     * Each element encodes a triangle (and potentially cached information) using the container
     * type @ref TriangleContainer4.
     *
     * @note
     * The exact meaning of "4" depends on your container definition (commonly 3 vertices + normal,
     * or 3 vertices + padding/alignment slot).
     */
    HostBuffer<TriangleContainer4<T>> triangles;

    /**
     * @brief Default constructor (empty mesh).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh() noexcept
        = default;

    /**
     * @brief Construct from a triangle buffer (copy).
     *
     * @param triangles_ Triangle buffer to copy.
     *
     * @note
     * Typically marks BVH as not built; BVH will be built lazily on demand.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept;

    /**
     * @brief Construct from a triangle buffer (move).
     *
     * @param triangles_ Triangle buffer to move from.
     *
     * @note
     * Typically marks BVH as not built; BVH will be built lazily on demand.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /// @brief Defaulted copy operations.
    TriangleMesh(const TriangleMesh&)     = default;
    TriangleMesh(TriangleMesh&&) noexcept = default;
    TriangleMesh&
    operator=(const TriangleMesh&)
        = default;
    TriangleMesh&
    operator=(TriangleMesh&&) noexcept
        = default;
    ~TriangleMesh() override = default;

    /**
     * @brief Replace triangle storage (copy).
     *
     * @details
     * Updates @ref triangles with a copied buffer. Implementations should invalidate the BVH
     * (set `bvh_built = false`) because geometry changed.
     *
     * @param triangles_ New triangle buffer to copy.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_);

    /**
     * @brief Create a trace operator bound to this mesh.
     *
     * @details
     * Returns a @ref TraceOperator for ray/primitive intersection queries.
     * For performance, this may require a BVH; implementations often call @ref ensure_bvh().
     *
     * @return Trace operator referencing this mesh (implementation-defined contents).
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE TraceOperator<T>
    make_trace_operator() const override;

    /**
     * @brief Create a query operator bound to this mesh.
     *
     * @details
     * Returns a @ref QueryOperator for closest point/normal and distance queries.
     * Implementations may return:
     * - a mesh query operator with pointers to contiguous geometry data, or
     * - a fallback that performs O(N) scanning using `triangles` depending on your operator design.
     *
     * @return Query operator referencing this mesh (implementation-defined).
     *
     * @note Host-only: operator construction typically binds pointers to host memory.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE QueryOperator<T>
    make_query_operator() const override;

    /**
     * @brief Load mesh triangles from a Wavefront OBJ file.
     *
     * @details
     * Reads an OBJ file from disk and populates @ref triangles. The loader may:
     * - triangulate faces,
     * - compute normals if not present,
     * - discard non-triangular primitives or convert them (implementation-defined).
     *
     * @param filename Path to OBJ file.
     * @param verbose  If `true`, prints diagnostic information (counts, warnings).
     * @return `true` on success; `false` on failure (parse/read error).
     *
     * @note
     * Host-only: performs file I/O and typically uses host memory allocations.
     * On success, the BVH should be invalidated and rebuilt lazily.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    load_from_obj(const std::string& filename, bool verbose = false);

    /**
     * @brief Compute the closest point on the mesh to a query point.
     *
     * @details
     * Typically searches over triangles:
     * - naive O(N) scan, or
     * - BVH-accelerated search if available/built.
     *
     * @param p Query point.
     * @return Closest point on the mesh surface.
     *
     * @note
     * If this function relies on a BVH, ensure the BVH exists (implementation may call @ref ensure_bvh()).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest normal on the mesh relative to a query point.
     *
     * @details
     * Returns the normal of the triangle (or interpolated normal if supported) corresponding
     * to the closest point/feature.
     *
     * @param p Query point.
     * @return Normal vector (typically unit length).
     *
     * @note
     * If triangle normals are not available, the implementation may compute them on the fly.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Signed distance from a point to the mesh.
     *
     * @details
     * A triangle mesh is not necessarily closed, so "signed distance" may be ambiguous.
     * Common approaches include:
     * - unsigned distance to closest triangle (always non-negative),
     * - signed distance using nearest triangle normal (local sign),
     * - robust SDF sign using winding/inside tests for closed meshes (more expensive).
     *
     * @param p Query point.
     * @return Signed distance value (implementation-defined semantics).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the mesh according to the nearest-triangle sign rule.
     *
     * @param p Query point.
     * @param tolerance Allowed positive slack relative to the local surface.
     * @return `true` if the point is classified as inside.
     *
     * @note
     * For open or non-watertight meshes, this is a local nearest-surface classification rather
     * than a globally robust winding-based inside test.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the mesh surface within a tolerance band.
     *
     * @param p Query point.
     * @param tolerance Allowed absolute deviation from the nearest triangle surface.
     * @return `true` if the point is classified as on the surface.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return a representative centroid for the mesh.
     *
     * @details
     * Possible implementations:
     * - average of triangle centroids (optionally area-weighted),
     * - average of all vertices,
     * - centroid of the bounding box.
     *
     * @return Representative mesh centroid (implementation-defined).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Axis-aligned bounding box of the mesh.
     *
     * @details
     * Typically computed as component-wise min/max over all triangle vertices.
     *
     * @return Mesh AABB bounds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Returns whether this mesh is valid for queries.
     *
     * @details
     * Typical checks include:
     * - triangle buffer non-empty (optional policy),
     * - finite vertex coordinates,
     * - non-degenerate triangles,
     * - BVH validity if BVH is built.
     *
     * @return `true` if usable; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this mesh.
     *
     * @details
     * Identifies the active type in a polymorphic context
     * (e.g., when using `Geometry<T>` pointers).
     *
     * @return `GeometryType::TriangleMesh` for this class.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow builder to configure internals without exposing mutators.
    friend class Builder;

private:
    /**
     * @brief Host-side BVH used to accelerate queries and tracing.
     *
     * @details
     * Owned through a host shared pointer.
     */
    BVHHostPtr<T> _bvh = nullptr;

    /**
     * @brief Cached contiguous vertex array for TriangleMeshQueryOperator construction.
     *
     * @details
     * Each triangle contributes its three vertices in sequence.
     */
    mutable DeviceBuffer<Vector3<T>> _query_vertices;

    /**
     * @brief Cached contiguous triangle index array for TriangleMeshQueryOperator construction.
     */
    mutable DeviceBuffer<int> _query_indices;

    /**
     * @brief Whether the BVH has been constructed for the current triangle data.
     */
    bool bvh_built = false;

    /**
     * @brief Whether the query-operator cache matches @ref triangles.
     */
    mutable bool query_cache_built = false;

    /**
     * @brief Ensure the BVH exists and is built for current triangles.
     *
     * @details
     * Host-side helper that constructs and/or rebuilds the BVH if needed.
     *
     * @note
     * Marked `noexcept` here; implementations should avoid throwing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_bvh() noexcept;

    /**
     * @brief Build (or rebuild) the BVH from current triangles.
     *
     * @details
     * Typically:
     * - allocates a BVH if `_bvh == nullptr`,
     * - inserts primitives and builds hierarchy,
     * - sets `bvh_built = true`.
     *
     * @note
     * Host-only: may allocate host memory and perform significant work.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_bvh();

    /**
     * @brief Ensure the query-operator cache matches the current triangle soup.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_query_cache() const;

    /**
     * @brief Rebuild the contiguous vertex/index cache used by TriangleMeshQueryOperator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_query_cache() const;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

/**
 * @brief Fluent builder for @ref TriangleMesh.
 *
 * @details
 * Provides construction routes:
 * - set triangles by copy/move,
 * - load triangles from an OBJ file,
 * - build by value or as a host_shared_ptr.
 *
 * The builder is host-only because it performs allocations and may do file I/O.
 *
 * ## Typical usage
 * @code
 * atlas::TriangleMeshF mesh = atlas::TriangleMeshF::builder()
 *     .load_from_obj("bunny.obj", true)
 *     .build();
 * @endcode
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class TriangleMesh<T>::Builder final {
public:
    /// @brief Default constructor.
    Builder() = default;

    /**
     * @brief Provide triangles by copy.
     *
     * @param ts Triangle buffer to copy.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(const HostBuffer<TriangleContainer4<T>>& ts);

    /**
     * @brief Provide triangles by move.
     *
     * @param ts Triangle buffer to move.
     * @return `*this` for chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(HostBuffer<TriangleContainer4<T>>&& ts);

    /**
     * @brief Load triangles from an OBJ file into the builder.
     *
     * @param filename Path to OBJ file.
     * @param verbose  Print diagnostics if true.
     * @return `*this` for chaining.
     *
     * @note
     * This typically overwrites any previously staged triangles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    load_from_obj(const std::string& filename, bool verbose = false);

    /**
     * @brief Build a configured @ref TriangleMesh (by value).
     *
     * @return Constructed mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh<T>
    build() const;

    /**
     * @brief Build a configured @ref TriangleMesh in a host_shared_ptr.
     *
     * @return Shared pointer owning the constructed mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<TriangleMesh<T>>
    make_host_shared() const;

private:
    /// @brief Validate staged parameters.
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Staged triangle buffer.
    HostBuffer<TriangleContainer4<T>> _triangles;
};

} // namespace geometry

namespace atlas {
/* ====================================================================== */
/* Public aliases                                                          */
/* ====================================================================== */

template <typename T>
using TriangleMesh  = geometry::TriangleMesh<T>;
using TriangleMeshF = geometry::TriangleMesh<float>;
using TriangleMeshD = geometry::TriangleMesh<double>;

template <typename T>
using TriangleMeshHostPtr = atlas::host_shared_ptr<geometry::TriangleMesh<T>>;

template <typename T>
using TriangleMeshDevicePtr = atlas::device_shared_ptr<geometry::TriangleMesh<T>>;

} // namespace atlas

#include <atlas/geometry/triangle_mesh.hpp>
