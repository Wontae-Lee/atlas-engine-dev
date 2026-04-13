#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file circle_layer.h
 * @brief Declares a visualization layer that renders circle geometry as line segments.
 *
 * @details
 * This header defines @ref atlas::vizkit::CircleLayer, a concrete implementation
 * of @ref atlas::vizkit::GeometryLayer that visualizes a circle associated with
 * a @ref atlas::Unit.
 *
 * ## Purpose
 * The circle layer is intended for:
 * - debugging circle geometry placement,
 * - inspecting planar circular primitives in a scene,
 * - lightweight wireframe visualization in Vizkit,
 * - validating local-to-world synchronization of circle-based units.
 *
 * ## Rendering model
 * The layer does not render a filled disk. Instead, it approximates the circle
 * boundary using a polyline composed of line segments.
 *
 * The number of segments is configurable:
 * - larger values produce a smoother apparent circle,
 * - smaller values reduce vertex count and rendering cost.
 *
 * ## Geometry generation
 * Geometry is built in the unit's **local coordinate frame**:
 * - the circle boundary is sampled uniformly in angle,
 * - consecutive samples are connected with line segments,
 * - the resulting local-space vertex positions are stored in a geometry buffer.
 *
 * The base @ref GeometryLayer is responsible for:
 * - transforming these positions to world space using the unit pose,
 * - managing render buffers,
 * - issuing draw calls using the configured primitive mode.
 *
 * ## Segment count
 * The `_segments` parameter controls tessellation quality:
 * - `segments = 3` gives the minimum closed polygon approximation,
 * - higher values improve visual smoothness,
 * - the default value is `64`.
 *
 * ## Builder pattern
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - configure the tessellation segment count,
 * - construct the layer by value or shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::CircleLayer<float>::builder()
 *     .with_unit(unit)
 *     .with_segments(128)
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
 * @brief Geometry layer that renders a circle as a polyline made of line segments.
 *
 * @details
 * @ref CircleLayer extracts circle geometry from a @ref atlas::Unit and emits
 * local-space line geometry approximating the circular boundary.
 *
 * The layer is intended for analytic circle visualization, where the underlying
 * geometry is planar and the visible representation should emphasize the
 * perimeter rather than a filled area.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - storing the tessellation segment count,
 * - generating local-space circle boundary vertices,
 * - providing those vertices to the base @ref GeometryLayer.
 *
 * The base layer handles:
 * - synchronization into world space,
 * - buffer upload,
 * - rendering lifecycle integration.
 *
 * ## Approximation quality
 * A circle cannot be rendered exactly with straight line primitives, so the
 * boundary is approximated by a regular polygon with @ref _segments edges.
 *
 * In general:
 * - more segments improve visual fidelity,
 * - fewer segments improve performance.
 *
 * ## Assumptions
 * The associated unit is expected to hold circle-compatible geometry. The exact
 * interpretation of center, normal, and radius is delegated to the underlying
 * geometry/unit pipeline.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class CircleLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref CircleLayer.
     */
    class Builder;

public:
    /**
     * @brief Construct a circle layer bound to a unit.
     *
     * @details
     * The layer stores the target unit and the number of segments used to
     * approximate the circle boundary.
     *
     * @param unit Host-side shared pointer to the unit whose circle geometry will be visualized.
     * @param segments Number of segments used for the circle approximation.
     */
    ATLAS_HOST explicit CircleLayer(
        const atlas::UnitHostPtr<T>& unit,
        int segments = 64);

    /**
     * @brief Create a fluent builder for @ref CircleLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~CircleLayer() override = default;

protected:
    /**
     * @brief Build local-space geometry for the circle.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when the local
     * geometry representation needs to be generated or refreshed.
     *
     * It populates @p positions with line segment endpoints representing a
     * polygonal approximation of the circle boundary in local coordinates.
     *
     * A typical implementation:
     * - samples boundary points at uniform angular intervals,
     * - connects each point to the next,
     * - closes the loop by connecting the final sample back to the first.
     *
     * @param positions Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Number of segments used to approximate the circle boundary.
     *
     * @details
     * Higher values produce a smoother visual approximation at the cost of more
     * generated vertices.
     */
    int _segments;
};

/**
 * @brief Fluent builder for @ref CircleLayer.
 *
 * @details
 * The builder stages configuration for a circle visualization layer and
 * constructs either:
 * - a value instance of @ref CircleLayer, or
 * - a `std::shared_ptr<CircleLayer>`.
 *
 * ## Configurable state
 * The builder stages:
 * - the target @ref atlas::UnitHostPtr,
 * - the segment count used for tessellation.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the segment count is valid for a closed polyline approximation.
 *
 * A typical validation policy requires `segments >= 3`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class CircleLayer<T>::Builder {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - no target unit,
     * - a default segment count of `64`.
     */
    Builder() = default;

    /**
     * @brief Assign the unit to be visualized.
     *
     * @param unit Host-side shared pointer to the target unit.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    /**
     * @brief Set the number of segments used for circle tessellation.
     *
     * @param segments Polygon segment count used to approximate the circle boundary.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_segments(int segments) noexcept;

    /**
     * @brief Build a @ref CircleLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed circle layer.
     */
    CircleLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref CircleLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<CircleLayer>` owning the constructed layer.
     */
    std::shared_ptr<CircleLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that required configuration is present and that the staged segment
     * count is suitable for rendering a closed circular approximation.
     */
    void
    validate() const;

private:
    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;

    /**
     * @brief Staged segment count for circle tessellation.
     */
    int _segments = 64;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/circle_layer.hpp>

#endif