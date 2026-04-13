#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file box_layer.h
 * @brief Declares a visualization layer that renders box geometry as line primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::BoxLayer, a concrete implementation of
 * @ref atlas::vizkit::GeometryLayer that visualizes an axis-aligned box associated
 * with a @ref atlas::Unit.
 *
 * ## Purpose
 * The box layer is used for:
 * - debugging geometry placement,
 * - visualizing bounding volumes,
 * - inspecting simulation units in world space,
 * - lightweight wireframe rendering in Vizkit.
 *
 * Instead of rendering filled geometry, the layer emits **line segments**
 * representing the edges of the box.
 *
 * ## Geometry model
 * The layer builds geometry in **local space**:
 * - it extracts the box bounds (min/max corners),
 * - generates the 12 edges of the box,
 * - appends them as line segments into a position buffer.
 *
 * The base @ref GeometryLayer is responsible for:
 * - transforming these positions into world space using the unit’s
 *   @ref atlas::system::Sync,
 * - uploading and rendering them with the configured OpenGL primitive mode.
 *
 * ## Rendering characteristics
 * - Primitive type: typically `GL_LINES`,
 * - Geometry: 12 edges → 24 vertices,
 * - Stateless: geometry is rebuilt when required by the base layer.
 *
 * ## Builder pattern
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - construct the layer,
 * - optionally wrap it in shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::BoxLayer<float>::builder()
 *     .with_unit(unit)
 *     .make_shared();
 * @endcode
 *
 * ---
 */

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a box as a wireframe.
 *
 * @details
 * @ref BoxLayer extracts box geometry from a @ref atlas::Unit and emits
 * line segments representing the edges of the box in local space.
 *
 * The base @ref GeometryLayer handles:
 * - synchronization to world space,
 * - buffer management,
 * - rendering.
 *
 * ## Responsibilities
 * This class is responsible only for:
 * - generating local-space vertex positions,
 * - defining how a box is decomposed into line segments.
 *
 * ## Assumptions
 * - The associated unit contains a box-like geometry or compatible bounds.
 * - The unit’s transform is applied externally by the base layer.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry positions.
 */
template <typename T>
class BoxLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref BoxLayer.
     */
    class Builder;

public:
    /**
     * @brief Construct a box layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose geometry will be visualized.
     */
    ATLAS_HOST explicit BoxLayer(const atlas::UnitHostPtr<T>& unit);

    /**
     * @brief Create a fluent builder for @ref BoxLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~BoxLayer() override = default;

protected:
    /**
     * @brief Build local-space geometry for the box.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when geometry needs
     * to be generated or refreshed.
     *
     * It populates the provided position buffer with line segment endpoints
     * representing the edges of the box in local coordinates.
     *
     * @param positions Output vector receiving local-space vertex positions.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Append line segments representing a box defined by its corners.
     *
     * @details
     * Generates the 12 edges of the axis-aligned box defined by:
     * - @p min_corner (lower bound),
     * - @p max_corner (upper bound).
     *
     * Each edge is represented as a pair of vertices appended to @p positions.
     *
     * @param min_corner Minimum corner of the box.
     * @param max_corner Maximum corner of the box.
     * @param positions Output vector receiving line segment vertices.
     */
    void
    append_local_box_lines(
        const Vector3<T>& min_corner,
        const Vector3<T>& max_corner,
        std::vector<Vector3<T>>& positions) const;
};

/**
 * @brief Fluent builder for @ref BoxLayer.
 *
 * @details
 * The builder stages configuration for a box visualization layer and constructs
 * either:
 * - a value instance of @ref BoxLayer, or
 * - a `std::shared_ptr<BoxLayer>`.
 *
 * ## Required configuration
 * - A valid @ref atlas::UnitHostPtr must be provided.
 *
 * ## Validation
 * The builder enforces that:
 * - the unit pointer is non-null before construction.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry positions.
 */
template <typename T>
class BoxLayer<T>::Builder {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Assign the unit to be visualized.
     *
     * @param unit Host-side shared pointer to the target unit.
     * @return Reference to this builder for chaining.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    /**
     * @brief Build a @ref BoxLayer instance.
     *
     * @return Constructed box layer.
     */
    BoxLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref BoxLayer.
     *
     * @return `std::shared_ptr<BoxLayer>` owning the constructed layer.
     */
    std::shared_ptr<BoxLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that required configuration (such as the unit pointer) is present.
     */
    void
    validate() const;

private:
    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/box_layer.hpp>

#endif