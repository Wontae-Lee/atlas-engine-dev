#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file triangle_mesh_layer.h
 * @brief Declares a visualization layer that renders triangle-mesh geometry as line primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::TriangleMeshLayer, a concrete
 * @ref atlas::vizkit::GeometryLayer implementation used to visualize a triangle
 * mesh associated with an @ref atlas::Unit.
 *
 * ## Purpose
 * The triangle-mesh layer is intended for:
 * - debugging mesh geometry imported into Atlas,
 * - inspecting transformed mesh placement in world space,
 * - visualizing collision or query meshes,
 * - lightweight wireframe rendering in Vizkit.
 *
 * ## Rendering model
 * The layer renders the mesh as a wireframe using line primitives rather than
 * as a filled shaded surface.
 *
 * A typical implementation traverses all mesh triangles and emits line segments
 * for their edges:
 * - edge `(a, b)`,
 * - edge `(b, c)`,
 * - edge `(c, a)`.
 *
 * Depending on the implementation in `triangle_mesh_layer.hpp`, duplicate edges
 * may be:
 * - emitted redundantly for simplicity, or
 * - filtered or deduplicated for a cleaner wireframe.
 *
 * ## Coordinate convention
 * Geometry is generated in the unit's **local coordinate frame**. The base
 * @ref GeometryLayer is responsible for:
 * - transforming local vertices into world space using the unit sync,
 * - maintaining render buffers,
 * - integrating the layer into the Vizkit rendering lifecycle.
 *
 * ## Typical use cases
 * This layer is useful when visually verifying:
 * - imported mesh topology,
 * - unit transforms applied to triangle meshes,
 * - BVH and spatial-query test scenes,
 * - collider/source/sink geometry based on meshes.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - construct the layer by value,
 * - construct the layer under shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::TriangleMeshLayer<float>::builder()
 *     .with_unit(unit)
 *     .make_shared();
 * @endcode
 *
 * ---
 */

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a triangle mesh as a wireframe.
 *
 * @details
 * @ref TriangleMeshLayer extracts triangle-mesh geometry from a bound
 * @ref atlas::Unit and emits local-space line vertices representing the mesh
 * edges.
 *
 * The layer is intended for mesh visualization in debugging and inspection
 * workflows where a lightweight line-based representation is preferable to
 * filled rendering.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - generating local-space wireframe geometry for the mesh,
 * - handing those vertices to the base @ref GeometryLayer.
 *
 * The base @ref GeometryLayer handles:
 * - synchronization into world space,
 * - OpenGL buffer management,
 * - shader setup and rendering integration.
 *
 * ## Mesh interpretation
 * The associated unit is expected to reference triangle-mesh-compatible
 * geometry. A typical implementation of @ref build_geometry:
 * - accesses the mesh triangles,
 * - expands them into per-edge line segments,
 * - appends those segments into the output position buffer.
 *
 * ## Rendering characteristics
 * For a mesh with `N` triangles, a simple wireframe representation typically
 * emits:
 * - `3N` edges,
 * - `6N` vertices when rendered with `GL_LINES`,
 * before any optional edge deduplication.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class TriangleMeshLayer final : public GeometryLayer<T> {

public:
    /**
     * @brief Fluent builder for constructing @ref TriangleMeshLayer.
     */
    class Builder;

    /**
     * @brief Construct a triangle-mesh layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose triangle-mesh geometry will be visualized.
     */
    TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Create a fluent builder for @ref TriangleMeshLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder();

protected:
    /**
     * @brief Build local-space wireframe geometry for the triangle mesh.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when the local
     * geometry representation needs to be generated or refreshed.
     *
     * A typical implementation:
     * - iterates over all triangles in the mesh,
     * - extracts triangle vertices,
     * - appends line segment endpoints for each triangle edge,
     * - writes the resulting vertices into @p pos in local coordinates.
     *
     * @param pos Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

/**
 * @brief Fluent builder for @ref TriangleMeshLayer.
 *
 * @details
 * The builder stages configuration for a triangle-mesh visualization layer and
 * constructs either:
 * - a value instance of @ref TriangleMeshLayer, or
 * - a `std::shared_ptr<TriangleMeshLayer>`.
 *
 * ## Required configuration
 * The builder requires a valid @ref atlas::UnitHostPtr corresponding to the
 * triangle mesh that should be visualized.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the bound unit contains mesh-compatible geometry,
 * - the mesh is valid or non-empty according to the implementation policy.
 *
 * The exact validation behavior is implementation-defined in
 * `triangle_mesh_layer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class TriangleMeshLayer<T>::Builder {

public:
    /**
     * @brief Assign the unit to be visualized.
     *
     * @param u Host-side shared pointer to the target unit.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& u);

    /**
     * @brief Build a @ref TriangleMeshLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed triangle-mesh layer.
     */
    TriangleMeshLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref TriangleMeshLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<TriangleMeshLayer>` owning the constructed layer.
     */
    std::shared_ptr<TriangleMeshLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that the staged unit pointer and any mesh-specific assumptions
     * required by the layer are satisfied.
     */
    void
    validate() const;

    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/triangle_mesh_layer.hpp>

#endif