#pragma once

/**
 * @file triangle_mesh.h
 * @brief Declares a triangle-mesh geometry primitive and its lightweight runtime query/trace operator.
 *
 * @details
 * This header defines @ref atlas::geometry::TriangleMesh, a surface geometry
 * represented by a collection of triangles, together with
 * @ref TriangleMeshGeometryOperator, a lightweight non-owning operator used for
 * backend-friendly geometric queries.
 *
 * A triangle mesh is useful for representing complex surfaces that cannot be
 * described analytically by a single primitive. In Atlas, the mesh supports:
 * - closest-point evaluation,
 * - closest-normal evaluation,
 * - signed-distance evaluation,
 * - inside/outside classification,
 * - surface classification,
 * - centroid and bounding-box queries,
 * - ray tracing,
 * - optional acceleration through a bounding volume hierarchy (BVH).
 *
 * ## Internal representation
 * The owning @ref TriangleMesh stores triangles in a host-side container of
 * @ref TriangleContainer4 objects. For query and tracing operations, it can also
 * maintain:
 * - cached query vertices and triangle indices,
 * - a BVH acceleration structure,
 * - a bound @ref TriangleMeshGeometryOperator that references those caches.
 *
 * ## Inside/outside queries
 * The operator exposes helper methods such as @ref solid_angle and
 * @ref winding_number, which are commonly used to classify a point with respect
 * to a closed surface mesh.
 *
 * ## Host/device split
 * - The owning @ref TriangleMesh object participates in the polymorphic
 *   @ref atlas::Geometry interface.
 * - The @ref TriangleMeshGeometryOperator provides a value-like non-owning view
 *   suitable for backend execution and runtime dispatch.
 *
 * ## Construction
 * A triangle mesh may be:
 * - default-constructed,
 * - constructed from a triangle buffer,
 * - populated from an OBJ file,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and distances.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

#include <string>
#include <type_traits>

namespace atlas::geometry {

/**
 * @brief Lightweight non-owning runtime query operator for a triangle mesh.
 *
 * @details
 * @ref TriangleMeshGeometryOperator stores raw pointers to mesh query buffers and
 * optional BVH acceleration data. It exposes a uniform set of geometric queries
 * without requiring host-side ownership or virtual dispatch.
 *
 * The operator references:
 * - a vertex array,
 * - an index array describing triangles,
 * - the total triangle count,
 * - optional BVH nodes, indices, and packed triangles for accelerated tracing.
 *
 * Since these pointers are non-owning, the referenced data must remain valid for
 * the duration of any use of the operator.
 *
 * ## Acceleration
 * When BVH data is available, the operator can use it for more efficient ray
 * tracing and nearest-surface queries. When it is absent, implementations may
 * fall back to direct triangle traversal.
 *
 * ## Inside/outside reasoning
 * The operator provides @ref solid_angle and @ref winding_number helpers, which
 * can be used to classify points relative to a closed triangle mesh.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct TriangleMeshGeometryOperator {
    /**
     * @brief Pointer to the query vertex array.
     *
     * @details
     * Non-owning pointer to vertex positions used by mesh queries.
     */
    const atlas::math::Vector<T, 3>* vertices = nullptr;

    /**
     * @brief Pointer to the triangle index array.
     *
     * @details
     * Non-owning pointer to the flattened index buffer describing mesh triangles.
     */
    const int* indices = nullptr;

    /**
     * @brief Number of triangles described by the query buffers.
     */
    int triangle_count = 0;

    /**
     * @brief Pointer to BVH nodes used for accelerated traversal.
     */
    const atlas::spatial::BVHNode<T>* bvh_nodes = nullptr;

    /**
     * @brief Pointer to BVH primitive index indirection.
     */
    const int* bvh_indices = nullptr;

    /**
     * @brief Pointer to packed triangle containers referenced by the BVH.
     */
    const TriangleContainer4<T>* bvh_tris = nullptr;

    /**
     * @brief Index of the BVH root node.
     *
     * @details
     * A negative value typically indicates that no valid BVH root is available.
     */
    int bvh_root = -1;

    /**
     * @brief Compute the oriented solid angle subtended by a triangle at a query point.
     *
     * @details
     * This helper is commonly used as part of winding-number-based inside/outside
     * classification for closed meshes.
     *
     * @param p Query point.
     * @param a First triangle vertex.
     * @param b Second triangle vertex.
     * @param c Third triangle vertex.
     * @return Oriented solid angle contribution.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    solid_angle(const atlas::math::Vector<T, 3>& p,
                const atlas::math::Vector<T, 3>& a,
                const atlas::math::Vector<T, 3>& b,
                const atlas::math::Vector<T, 3>& c) const noexcept;

    /**
     * @brief Compute the winding number of the mesh at a query point.
     *
     * @details
     * For closed consistently oriented meshes, the winding number can be used to
     * classify whether a point lies inside or outside the enclosed volume.
     *
     * @param p Query point.
     * @return Winding-number value at @p p.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    winding_number(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest point on the triangle mesh to a query point.
     *
     * @details
     * The result is the nearest point on any triangle of the mesh. Depending on
     * the implementation and available acceleration data, this may be evaluated
     * using direct iteration or BVH-guided traversal.
     *
     * @param p Query point.
     * @return Closest point on the mesh surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the closest surface normal associated with a query point.
     *
     * @details
     * The returned normal is typically taken from the triangle that owns the
     * closest point or nearest ray hit, subject to implementation policy.
     *
     * @param p Query point.
     * @return Closest surface normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Compute the signed distance from a query point to the mesh.
     *
     * @details
     * The unsigned part is determined by the closest-surface distance, while the
     * sign is typically derived from inside/outside classification such as the
     * winding number.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    /**
     * @brief Test whether a point lies inside the mesh within a tolerance.
     *
     * @details
     * For closed meshes, this is commonly implemented using winding-number logic
     * or a related inside/outside classification method.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if the point is classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Test whether a point lies on the mesh surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Absolute tolerance used for surface classification.
     * @return `true` if the point is classified as on the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    /**
     * @brief Return a centroid-like representative point of the triangle mesh.
     *
     * @details
     * The exact definition is implementation-defined. It may represent the
     * centroid of all vertices, all triangles, or another stable representative.
     *
     * @return Mesh centroid or representative point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    /**
     * @brief Return the axis-aligned bounding box of the triangle mesh.
     *
     * @return Axis-aligned bounding box enclosing the mesh.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    /**
     * @brief Return whether the operator references a valid triangle mesh query state.
     *
     * @details
     * Typical validity checks include:
     * - non-null query buffers,
     * - non-negative triangle count,
     * - internally consistent index and acceleration data.
     *
     * @return `true` if the operator is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    /**
     * @brief Trace a ray against the triangle mesh.
     *
     * @details
     * Intersects the ray with the mesh and typically returns the nearest valid
     * forward hit. Implementations may use the BVH when available.
     *
     * @param ray Query ray.
     * @return Surface hit record describing the ray-mesh intersection result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    /**
     * @brief Function-call alias for @ref trace.
     *
     * @param ray Query ray.
     * @return Surface hit record.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

/**
 * @brief Triangle-mesh geometry primitive implementing @ref atlas::Geometry.
 *
 * @details
 * @ref TriangleMesh represents a geometric surface defined by a collection of
 * triangles. It is suitable for complex models imported from external assets or
 * assembled procedurally.
 *
 * The mesh stores:
 * - the source triangle containers,
 * - an optional BVH acceleration structure,
 * - mutable query caches for flattened vertices and indices,
 * - a cached @ref TriangleMeshGeometryOperator bound to those query caches.
 *
 * ## Query and tracing support
 * The mesh provides:
 * - closest-point queries,
 * - closest-normal queries,
 * - signed-distance evaluation,
 * - inside/surface classification,
 * - centroid and bounding-box queries,
 * - geometry-type reporting,
 * - OBJ loading for mesh import.
 *
 * ## Caching and acceleration
 * To support efficient runtime queries, the mesh may:
 * - lazily build a BVH,
 * - lazily rebuild query caches,
 * - update its cached geometry operator as internal state changes.
 *
 * ## Ownership model
 * The owning object stores triangle data on the host, while the exported
 * @ref TriangleMeshGeometryOperator provides a lightweight runtime view.
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
     * The builder stages triangle data or loads mesh data from an OBJ file,
     * validates it, and constructs either:
     * - a mesh by value, or
     * - a host-owned shared pointer to a mesh.
     */
    class Builder;

public:
    /**
     * @brief Source triangle storage for the mesh.
     *
     * @details
     * Stores the mesh triangles in host-side packed containers.
     */
    HostBuffer<TriangleContainer4<T>> triangles;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty mesh with no triangles and no acceleration data.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh() noexcept = default;

    /**
     * @brief Construct a mesh from a const host triangle buffer.
     *
     * @param triangles_ Source triangle containers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept;

    /**
     * @brief Construct a mesh from an rvalue host triangle buffer.
     *
     * @param triangles_ Source triangle containers to move into the mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies mesh data and associated runtime state as defined in the implementation.
     *
     * @param other Source mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh(const TriangleMesh& other);

    /**
     * @brief Move constructor.
     *
     * @details
     * Moves mesh data and associated runtime state from the source mesh.
     *
     * @param other Source mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh(TriangleMesh&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @param other Source mesh.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh&
    operator=(const TriangleMesh& other);

    /**
     * @brief Move assignment operator.
     *
     * @param other Source mesh.
     * @return `*this`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh&
    operator=(TriangleMesh&& other) noexcept;

    /**
     * @brief Virtual destructor.
     */
    ~TriangleMesh() override = default;

    /**
     * @brief Replace the mesh triangle storage.
     *
     * @details
     * Updates the mesh triangles and typically invalidates or rebuilds dependent
     * caches and acceleration data as needed by the implementation.
     *
     * @param triangles_ New triangle storage.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_);

    /**
     * @brief Create a geometry operator bound to this mesh.
     *
     * @details
     * Returns a lightweight non-owning operator referencing the mesh query caches
     * and acceleration data.
     *
     * @return Bound geometry operator.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    /**
     * @brief Load mesh data from an OBJ file.
     *
     * @details
     * Parses the supplied OBJ file and populates the mesh triangles accordingly.
     *
     * @param filename Path to the OBJ file.
     * @param verbose Whether to emit verbose loading diagnostics.
     * @return `true` if loading succeeded; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    load_from_obj(const std::string& filename, bool verbose = false);

    /**
     * @brief Compute the closest point on the mesh to a query point.
     *
     * @param p Query point.
     * @return Closest point on the mesh surface.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the closest surface normal associated with a query point.
     *
     * @param p Query point.
     * @return Closest surface normal.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Compute the signed distance from a query point to the mesh.
     *
     * @param p Query point.
     * @return Signed distance value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    /**
     * @brief Test whether a point lies inside the mesh within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Classification tolerance.
     * @return `true` if classified as inside; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Test whether a point lies on the mesh surface within a tolerance.
     *
     * @param p Query point.
     * @param tolerance Surface tolerance.
     * @return `true` if classified as on the surface; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    /**
     * @brief Return a centroid-like representative point of the mesh.
     *
     * @return Mesh centroid or representative point.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    /**
     * @brief Return the axis-aligned bounding box of the mesh.
     *
     * @return Axis-aligned bounding box enclosing the mesh.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    /**
     * @brief Return whether this triangle mesh is valid.
     *
     * @details
     * Typical validity checks include:
     * - non-empty triangle storage,
     * - valid triangle data,
     * - consistent caches and acceleration state when built.
     *
     * @return `true` if the mesh is well-formed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    /**
     * @brief Return the geometry type tag for this primitive.
     *
     * @return `GeometryType` tag corresponding to a triangle mesh.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    /// @brief Allow the builder to configure mesh internals directly.
    friend class Builder;

private:
    /**
     * @brief Optional BVH acceleration structure.
     *
     * @details
     * Built lazily or on demand to accelerate query and tracing operations.
     */
    BVHHostPtr<T> _bvh = nullptr;

    /**
     * @brief Cached flattened query vertices.
     *
     * @details
     * Mutable cache used by the runtime operator.
     */
    mutable HostBuffer<Vector3<T>> _query_vertices;

    /**
     * @brief Cached flattened query indices.
     *
     * @details
     * Mutable cache used by the runtime operator.
     */
    mutable HostBuffer<int> _query_indices;

    /**
     * @brief Whether the BVH has been built.
     */
    bool bvh_built = false;

    /**
     * @brief Whether the query cache has been built.
     */
    mutable bool query_cache_built = false;

    /**
     * @brief Cached runtime operator bound to this mesh.
     *
     * @details
     * Stores raw pointers to cached query data and acceleration structures.
     */
    mutable TriangleMeshGeometryOperator<T> _operator {};

    /**
     * @brief Ensure that the BVH exists.
     *
     * @details
     * Builds or refreshes the BVH on demand if it is not already available.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_bvh() noexcept;

    /**
     * @brief Build the BVH acceleration structure.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_bvh();

    /**
     * @brief Ensure that the query cache exists.
     *
     * @details
     * Rebuilds the cached query buffers on demand if they are invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_query_cache() const;

    /**
     * @brief Rebuild the cached query buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_query_cache() const;

    /**
     * @brief Refresh the cached runtime operator bindings.
     *
     * @details
     * Updates the raw-pointer fields of @ref _operator so that they reference
     * the current query caches and acceleration structures.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_operator() const;
};

/**
 * @brief Fluent builder for @ref TriangleMesh.
 *
 * @details
 * The builder provides a controlled construction path for triangle meshes by
 * staging triangle data or loading mesh content from an OBJ file.
 *
 * ## Typical usage
 * @code
 * auto mesh = atlas::TriangleMeshF::builder()
 *     .with_triangles(tris)
 *     .build();
 * @endcode
 *
 * or
 *
 * @code
 * auto mesh = atlas::TriangleMeshF::builder()
 *     .load_from_obj("model.obj", true)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - triangle storage is non-empty,
 * - triangle data is internally valid,
 * - loaded mesh data is usable for runtime queries.
 *
 * The exact validation rules are implementation-defined in `triangle_mesh.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class TriangleMesh<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with empty triangle storage.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref TriangleMesh by value after validation.
     *
     * @return Constructed mesh value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh<T>
    build() const;

    /**
     * @brief Build a configured @ref TriangleMesh in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<TriangleMesh<T>>` owning the constructed mesh.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<TriangleMesh<T>>
    make_host_shared() const;

    /**
     * @brief Set the mesh triangles from a const host buffer.
     *
     * @param ts Source triangle containers.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(const HostBuffer<TriangleContainer4<T>>& ts);

    /**
     * @brief Set the mesh triangles from an rvalue host buffer.
     *
     * @param ts Source triangle containers to move into the builder.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(HostBuffer<TriangleContainer4<T>>&& ts);

    /**
     * @brief Load builder triangle data from an OBJ file.
     *
     * @param filename Path to the OBJ file.
     * @param verbose Whether to emit verbose loading diagnostics.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    load_from_obj(const std::string& filename, bool verbose = false);

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged triangle data.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending triangle storage.
     *
     * @details
     * Triangles staged for the final mesh construction.
     */
    HostBuffer<TriangleContainer4<T>> _triangles;
};

} // namespace atlas::geometry

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::geometry::TriangleMesh.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using TriangleMesh = geometry::TriangleMesh<T>;

/**
 * @brief Common specialization of @ref atlas::geometry::TriangleMesh for `float`.
 */
using TriangleMeshF = geometry::TriangleMesh<float>;

/**
 * @brief Common specialization of @ref atlas::geometry::TriangleMesh for `double`.
 */
using TriangleMeshD = geometry::TriangleMesh<double>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::geometry::TriangleMesh.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using TriangleMeshHostPtr = atlas::host_shared_ptr<geometry::TriangleMesh<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::geometry::TriangleMesh.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using TriangleMeshDevicePtr = atlas::device_shared_ptr<geometry::TriangleMesh<T>>;

} // namespace atlas

#include <atlas/geometry/triangle_mesh.hpp>