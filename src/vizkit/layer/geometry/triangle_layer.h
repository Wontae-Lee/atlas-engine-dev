#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders one queried triangle.
 *
 * @tparam T Scalar type used by queried geometry and generated vertices.
 *
 * The layer extracts the three triangle vertices from the bound unit's query
 * operator and forwards them as a single local-space triangle. The base
 * @ref GeometryLayer handles subsequent synchronization into world space and
 * rendering.
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::Triangle` with vertices `a`, `b`, and `c`.
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_TRIANGLES`
 * - Output topology: exactly one triangle when query data is complete
 */
template <typename T>
class TriangleLayer final : public GeometryLayer<T> {
public:
    /** @brief Fluent builder for constructing @ref TriangleLayer instances. */
    class Builder;

    /**
     * @brief Constructs a triangle layer bound to a unit.
     *
     * @param unit Unit whose query operator provides triangle geometry.
     */
    TriangleLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Creates a builder for @ref TriangleLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder();

protected:
    /**
     * @brief Generates the local-space vertex list for one triangle.
     *
     * @param pos Destination buffer receiving three vertices.
     *
     * If the bound unit is null, the query type is not `Triangle`, or any of
     * the three vertices is missing, the output is cleared and left empty.
     */
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

template <typename T>
class TriangleLayer<T>::Builder {
public:
    /**
     * @brief Sets the unit supplying triangle query data.
     *
     * @param u Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& u);

    /**
     * @brief Builds a @ref TriangleLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    TriangleLayer
    build() const;

    /**
     * @brief Builds a heap-allocated @ref TriangleLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    std::shared_ptr<TriangleLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that the builder has a unit to visualize.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    void
    validate() const;

    /** @brief Unit whose triangle query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
};

}

#include <vizkit/layer/geometry/triangle_layer.hpp>

#endif
