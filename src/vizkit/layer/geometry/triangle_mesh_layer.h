#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a triangle mesh from indexed query data.
 *
 * @tparam T Scalar type used by queried geometry and generated vertices.
 *
 * The layer reads a triangle mesh from the bound unit's query operator and
 * expands the indexed mesh into a flat triangle list. Each triangle contributes
 * three positions in local space, which are then synchronized and rendered by
 * the base @ref GeometryLayer.
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::TriangleMesh` with:
 * - a vertex array
 * - an index array
 * - a valid triangle count
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_TRIANGLES`
 * - Output topology: non-indexed triangle list derived from the mesh indices
 *
 * @note Indices are expanded eagerly into duplicated vertices. This keeps the
 * render path simple at the layer level, at the cost of higher vertex count.
 */
template <typename T>
class TriangleMeshLayer final : public GeometryLayer<T> {

public:
    /** @brief Fluent builder for constructing @ref TriangleMeshLayer instances. */
    class Builder;

    /**
     * @brief Constructs a triangle mesh layer bound to a unit.
     *
     * @param unit Unit whose query operator provides triangle mesh data.
     */
    TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Creates a builder for @ref TriangleMeshLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder();

protected:
    /**
     * @brief Expands indexed mesh triangles into a flat local-space vertex list.
     *
     * @param pos Destination buffer receiving three vertices per triangle.
     *
     * If the bound unit is null, the query type is not `TriangleMesh`, or the
     * mesh is missing vertices or indices, the output is cleared and left empty.
     */
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

template <typename T>
class TriangleMeshLayer<T>::Builder {

public:
    /**
     * @brief Sets the unit supplying triangle mesh query data.
     *
     * @param u Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& u);

    /**
     * @brief Builds a @ref TriangleMeshLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    TriangleMeshLayer
    build() const;

    /**
     * @brief Builds a heap-allocated @ref TriangleMeshLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    std::shared_ptr<TriangleMeshLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that the builder has a unit to visualize.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    void
    validate() const;

    /** @brief Unit whose triangle mesh query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
};

}

#include <vizkit/layer/geometry/triangle_mesh_layer.hpp>

#endif
