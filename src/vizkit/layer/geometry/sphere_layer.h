#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file sphere_layer.h
 * @brief Declares a visualization layer that renders sphere geometry as line-based wireframe primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::SphereLayer, a concrete
 * @ref atlas::vizkit::GeometryLayer implementation used to visualize a sphere
 * associated with an @ref atlas::Unit.
 *
 * ## Purpose
 * The sphere layer is intended for:
 * - debugging analytic sphere geometry,
 * - inspecting unit placement and transform synchronization,
 * - visualizing collision or bounding shapes,
 * - lightweight wireframe rendering in Vizkit.
 *
 * ## Rendering model
 * The layer does not render a filled sphere surface. Instead, it generates a
 * wireframe approximation using line primitives derived from a tessellated
 * latitude/longitude parameterization.
 *
 * A typical visualization includes:
 * - horizontal rings for constant polar elevation,
 * - vertical arcs for constant azimuth,
 * - a configurable angular subdivision in both directions.
 *
 * ## Tessellation parameters
 * Sphere tessellation is controlled by:
 * - @ref _slices : subdivision count around the azimuth direction,
 * - @ref _stacks : subdivision count along the polar/elevation direction.
 *
 * In general:
 * - more slices improve smoothness around the equator,
 * - more stacks improve smoothness from pole to pole,
 * - higher counts increase vertex count and rendering cost.
 *
 * ## Coordinate convention
 * Geometry is built in the unit's **local space**. The base
 * @ref GeometryLayer is responsible for:
 * - transforming local vertices into world space using the unit sync,
 * - managing CPU/GPU geometry buffers,
 * - issuing the final draw calls.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - configure slice and stack counts,
 * - construct the layer by value or shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::SphereLayer<float>::builder()
 *     .with_unit(unit)
 *     .with_slices(48)
 *     .with_stacks(24)
 *     .make_shared();
 * @endcode
 *
 * ---
 */

#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a sphere as a wireframe approximation.
 *
 * @details
 * @ref SphereLayer extracts sphere geometry from a bound @ref atlas::Unit and
 * emits local-space line vertices representing a tessellated wireframe sphere.
 *
 * The generated geometry is typically based on spherical parameter lines:
 * - constant-latitude rings,
 * - constant-longitude arcs.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - storing the tessellation parameters,
 * - generating local-space sphere wireframe geometry,
 * - handing that geometry to the base @ref GeometryLayer.
 *
 * The base layer handles:
 * - synchronization into world space,
 * - GPU upload and rendering lifecycle,
 * - camera-space rendering integration.
 *
 * ## Approximation quality
 * Since the sphere is rendered using straight line segments, the visible quality
 * depends on both @ref _slices and @ref _stacks.
 *
 * In general:
 * - increasing @ref _slices improves circumferential smoothness,
 * - increasing @ref _stacks improves vertical smoothness.
 *
 * ## Assumptions
 * The associated unit is expected to hold sphere-compatible geometry. The exact
 * interpretation of center and radius is delegated to the underlying geometry
 * and unit pipeline.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class SphereLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref SphereLayer.
     */
    class Builder;

public:
    /**
     * @brief Construct a sphere layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose sphere geometry will be visualized.
     * @param slices Number of azimuthal subdivisions used for sphere tessellation.
     * @param stacks Number of polar subdivisions used for sphere tessellation.
     */
    SphereLayer(const atlas::UnitHostPtr<T>& unit, int slices, int stacks);

    /**
     * @brief Create a fluent builder for @ref SphereLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder() noexcept;

protected:
    /**
     * @brief Build local-space line geometry for the sphere.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when the local
     * geometry representation needs to be generated or refreshed.
     *
     * A typical implementation:
     * - samples points on a sphere using spherical coordinates,
     * - connects neighboring samples along slice and stack directions,
     * - appends the resulting line segment endpoints to @p positions.
     *
     * @param positions Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Number of azimuthal subdivisions around the sphere.
     *
     * @details
     * Controls tessellation smoothness around the equatorial direction.
     */
    int _slices;

    /**
     * @brief Number of polar subdivisions from one pole to the other.
     *
     * @details
     * Controls tessellation smoothness along the vertical direction.
     */
    int _stacks;
};

/**
 * @brief Fluent builder for @ref SphereLayer.
 *
 * @details
 * The builder stages configuration for a sphere visualization layer and
 * constructs either:
 * - a value instance of @ref SphereLayer, or
 * - a `std::shared_ptr<SphereLayer>`.
 *
 * ## Configurable state
 * The builder stages:
 * - the target @ref atlas::UnitHostPtr,
 * - the slice count used for azimuthal tessellation,
 * - the stack count used for polar tessellation.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the slice count is large enough for a closed circular approximation,
 * - the stack count is large enough for a meaningful spherical subdivision.
 *
 * A typical validation policy requires:
 * - `slices >= 3`
 * - `stacks >= 2`
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class SphereLayer<T>::Builder {
public:
    /**
     * @brief Assign the unit to be visualized.
     *
     * @param unit Host-side shared pointer to the target unit.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;

    /**
     * @brief Set the number of azimuthal slices used for sphere tessellation.
     *
     * @param slices Azimuthal subdivision count.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_slices(int slices) noexcept;

    /**
     * @brief Set the number of polar stacks used for sphere tessellation.
     *
     * @param stacks Polar subdivision count.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_stacks(int stacks) noexcept;

    /**
     * @brief Build a @ref SphereLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed sphere layer.
     */
    SphereLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref SphereLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<SphereLayer>` owning the constructed layer.
     */
    std::shared_ptr<SphereLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that required configuration is present and that the staged slice
     * and stack counts are suitable for rendering a sphere wireframe.
     */
    void
    validate() const;

private:
    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;

    /**
     * @brief Staged azimuthal subdivision count.
     */
    int _slices = 32;

    /**
     * @brief Staged polar subdivision count.
     */
    int _stacks = 16;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/sphere_layer.hpp>

#endif