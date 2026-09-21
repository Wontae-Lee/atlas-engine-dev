#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file triangle_layer.h
 * @brief Declares a visualization layer that renders triangle geometry as line primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::TriangleLayer, a concrete
 * @ref atlas::vizkit::GeometryLayer implementation used to visualize a single
 * triangle associated with an @ref atlas::Unit.
 *
 * ## Purpose
 * The triangle layer is useful for:
 * - debugging analytic triangle geometry,
 * - inspecting local-to-world synchronization of triangular primitives,
 * - visualizing collision or mesh facets in isolation,
 * - lightweight wireframe rendering in Vizkit.
 *
 * ## Rendering model
 * The layer renders the triangle as a wireframe using line primitives rather
 * than as a filled surface.
 *
 * A typical representation consists of the three triangle edges:
 * - edge from vertex A to vertex B,
 * - edge from vertex B to vertex C,
 * - edge from vertex C to vertex A.
 *
 * ## Coordinate convention
 * Geometry is generated in the unit's **local coordinate frame**. The base
 * @ref GeometryLayer is responsible for:
 * - transforming those local vertices into world space using the unit sync,
 * - managing CPU/GPU geometry buffers,
 * - integrating with the Vizkit rendering lifecycle.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - construct the layer by value,
 * - construct the layer under shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::TriangleLayer<float>::builder()
 *     .with_unit(unit)
 *     .make_shared();
 * @endcode
 *
 * ---
 */

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a triangle as a wireframe.
 *
 * @details
 * @ref TriangleLayer extracts triangle geometry from a bound @ref atlas::Unit
 * and emits local-space line vertices representing the triangle edges.
 *
 * The layer is intended for visualization of a single analytic triangle rather
 * than a full mesh. It is especially useful for debugging:
 * - triangle placement,
 * - triangle orientation,
 * - closest-point and ray-query test cases,
 * - synchronization behavior of transformed primitives.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - generating local-space line geometry for a triangle,
 * - passing those vertices to the base @ref GeometryLayer.
 *
 * The base @ref GeometryLayer handles:
 * - synchronization to world space,
 * - OpenGL buffer management,
 * - shader interaction,
 * - draw/update lifecycle integration.
 *
 * ## Wireframe representation
 * The local geometry typically consists of:
 * - 3 line segments,
 * - 6 vertices total when using `GL_LINES`.
 *
 * ## Assumptions
 * The associated unit is expected to contain triangle-compatible geometry.
 * The exact extraction of triangle vertices is implementation-defined in
 * `triangle_layer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class TriangleLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref TriangleLayer.
     */
    class Builder;

    /**
     * @brief Construct a triangle layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose triangle geometry will be visualized.
     */
    TriangleLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Create a fluent builder for @ref TriangleLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder();

protected:
    /**
     * @brief Build local-space wireframe geometry for the triangle.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when the local
     * geometry representation must be generated or refreshed.
     *
     * A typical implementation:
     * - extracts triangle vertices from the bound unit geometry,
     * - appends line endpoints for the three triangle edges,
     * - writes those vertices into @p pos in local coordinates.
     *
     * @param pos Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

/**
 * @brief Fluent builder for @ref TriangleLayer.
 *
 * @details
 * The builder stages configuration for a triangle visualization layer and
 * constructs either:
 * - a value instance of @ref TriangleLayer, or
 * - a `std::shared_ptr<TriangleLayer>`.
 *
 * ## Required configuration
 * The builder requires a valid @ref atlas::UnitHostPtr corresponding to the
 * triangle that should be visualized.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the bound unit contains geometry suitable for triangle visualization.
 *
 * The exact validation behavior is implementation-defined in
 * `triangle_layer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class TriangleLayer<T>::Builder {
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
     * @brief Build a @ref TriangleLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed triangle layer.
     */
    TriangleLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref TriangleLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<TriangleLayer>` owning the constructed layer.
     */
    std::shared_ptr<TriangleLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that the staged unit pointer and any geometry-specific assumptions
     * required by the triangle layer are satisfied.
     */
    void
    validate() const;

    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/triangle_layer.hpp>

#endif