#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a queried box as a wireframe edge set.
 *
 * @tparam T Scalar type used by the underlying atlas geometry and vertex data.
 *
 * This layer reads box geometry from the bound unit's query operator and emits
 * local-space line segments for the 12 edges of the axis-aligned box. The base
 * @ref GeometryLayer is then responsible for synchronizing those local vertices
 * into world space and uploading them to the GPU.
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::Box` with valid lower and upper corners.
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_LINES`
 * - Vertex layout: 24 positions, two vertices per edge
 * - Topology: wireframe only, no filled faces
 *
 * @note The generated box is taken directly from the query operator's local
 * bounds. No extra validation is performed here beyond checking that both
 * corners exist and that the query type is `Box`.
 */
template <typename T>
class BoxLayer final : public GeometryLayer<T> {
public:
    /** @brief Fluent builder for constructing @ref BoxLayer instances. */
    class Builder;

public:
    /**
     * @brief Constructs a box layer bound to a unit.
     *
     * @param unit Unit whose query operator provides box geometry.
     */
    ATLAS_HOST explicit BoxLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Creates a builder for @ref BoxLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder() noexcept;

    ~BoxLayer() override = default;

protected:
    /**
     * @brief Generates the local-space line vertex list for the box.
     *
     * @param positions Destination buffer filled with box edge endpoints.
     *
     * If the bound unit is null, the query type is not `Box`, or the query does
     * not provide both corners, the output is cleared and left empty.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Appends the 12 edges of a box as line-list vertices.
     *
     * @param min_corner Lower corner of the local axis-aligned box.
     * @param max_corner Upper corner of the local axis-aligned box.
     * @param positions Output vertex buffer written as line segment endpoints.
     *
     * The function expands the eight corner points of the box and emits 24
     * vertices describing:
     * - 4 edges on the bottom face
     * - 4 edges on the top face
     * - 4 vertical connecting edges
     */
    void
    append_local_box_lines(
        const Vector3<T>& min_corner,
        const Vector3<T>& max_corner,
        std::vector<Vector3<T>>& positions) const;
};

template <typename T>
class BoxLayer<T>::Builder {
public:
    /** @brief Creates an empty builder. */
    Builder() = default;

    /**
     * @brief Sets the unit supplying box query data.
     *
     * @param unit Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    /**
     * @brief Builds a @ref BoxLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If required builder state is missing.
     */
    BoxLayer
    build() const;

    /**
     * @brief Builds a heap-allocated @ref BoxLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If required builder state is missing.
     */
    std::shared_ptr<BoxLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that the builder is ready to construct a layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     */
    void
    validate() const;

private:
    /** @brief Unit whose box query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
};
}

#include <vizkit/layer/geometry/box_layer.hpp>

#endif
