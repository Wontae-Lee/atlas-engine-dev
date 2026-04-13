#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file cylinder_layer.h
 * @brief Declares a visualization layer that renders cylinder geometry as line primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::CylinderLayer, a concrete
 * @ref atlas::vizkit::GeometryLayer implementation used to visualize a cylinder
 * associated with an @ref atlas::Unit.
 *
 * ## Purpose
 * The cylinder layer is intended for:
 * - debugging analytic cylinder geometry,
 * - inspecting unit placement and orientation,
 * - visualizing collision or domain shapes,
 * - lightweight wireframe rendering in Vizkit.
 *
 * ## Rendering model
 * The layer generates a wireframe-style representation of a cylinder using line
 * primitives rather than filled surfaces.
 *
 * A typical visualization includes:
 * - the top circular rim,
 * - the bottom circular rim,
 * - vertical side lines connecting corresponding rim samples.
 *
 * ## Tessellation
 * The circular parts of the cylinder are approximated using a configurable
 * number of angular subdivisions called **slices**:
 * - higher slice counts improve smoothness,
 * - lower slice counts reduce vertex count and render cost.
 *
 * ## Coordinate convention
 * Geometry is generated in the unit's **local space**. The base
 * @ref GeometryLayer is responsible for:
 * - transforming those vertices into world space through the unit's sync,
 * - maintaining visualization buffers,
 * - issuing the final draw calls.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - configure the number of slices,
 * - construct the layer by value or shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::CylinderLayer<float>::builder()
 *     .with_unit(unit)
 *     .with_slices(48)
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
 * @brief Geometry layer that renders a cylinder as a wireframe approximation.
 *
 * @details
 * @ref CylinderLayer extracts cylinder geometry from a bound @ref atlas::Unit
 * and emits local-space line vertices representing a wireframe view of the
 * cylinder.
 *
 * The generated geometry is typically composed of:
 * - a polygonal approximation of the top circle,
 * - a polygonal approximation of the bottom circle,
 * - line segments connecting the two circles along the cylinder side.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - storing the tessellation slice count,
 * - generating local-space line geometry for the cylinder,
 * - handing that geometry to the base @ref GeometryLayer.
 *
 * The base layer is responsible for:
 * - synchronization into world space,
 * - GPU/upload lifecycle,
 * - rendering integration.
 *
 * ## Approximation quality
 * Since circular rims are rendered with straight segments, the visual quality
 * depends on @ref _slices.
 *
 * In general:
 * - more slices produce a smoother rim,
 * - fewer slices reduce memory and draw cost.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class CylinderLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref CylinderLayer.
     */
    class Builder;

public:
    /**
     * @brief Construct a cylinder layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose cylinder geometry will be visualized.
     * @param slices Number of angular subdivisions used to approximate the cylinder rims.
     */
    ATLAS_HOST explicit CylinderLayer(
        const atlas::UnitHostPtr<T>& unit,
        int slices = 32);

    /**
     * @brief Create a fluent builder for @ref CylinderLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~CylinderLayer() override = default;

protected:
    /**
     * @brief Build local-space line geometry for the cylinder.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when geometry must
     * be generated or refreshed.
     *
     * A typical implementation samples the top and bottom circular rims using
     * @ref _slices angular steps and appends:
     * - rim edges on the top cap,
     * - rim edges on the bottom cap,
     * - vertical side edges between corresponding samples.
     *
     * @param positions Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Number of angular subdivisions used for the cylinder approximation.
     *
     * @details
     * Controls the number of line segments used to represent the top and bottom
     * circular rims.
     */
    int _slices;
};

/**
 * @brief Fluent builder for @ref CylinderLayer.
 *
 * @details
 * The builder stages configuration for a cylinder visualization layer and
 * constructs either:
 * - a value instance of @ref CylinderLayer, or
 * - a `std::shared_ptr<CylinderLayer>`.
 *
 * ## Configurable state
 * The builder stages:
 * - the target @ref atlas::UnitHostPtr,
 * - the number of slices used for angular tessellation.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the slice count is valid for a circular approximation.
 *
 * A typical validation policy requires:
 * - `slices >= 3`
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class CylinderLayer<T>::Builder {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - no target unit,
     * - a default slice count of `32`.
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
     * @brief Set the number of slices used for cylinder tessellation.
     *
     * @param slices Angular subdivision count used for the cylinder wireframe.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_slices(int slices) noexcept;

    /**
     * @brief Build a @ref CylinderLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed cylinder layer.
     */
    CylinderLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref CylinderLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<CylinderLayer>` owning the constructed layer.
     */
    std::shared_ptr<CylinderLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that required configuration is present and that the staged slice
     * count is suitable for rendering a cylinder wireframe.
     */
    void
    validate() const;

private:
    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;

    /**
     * @brief Staged angular subdivision count.
     */
    int _slices = 32;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/cylinder_layer.hpp>

#endif