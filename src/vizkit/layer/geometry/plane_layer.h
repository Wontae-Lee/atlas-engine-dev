#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file plane_layer.h
 * @brief Declares a visualization layer that renders plane geometry as finite line primitives.
 *
 * @details
 * This header defines @ref atlas::vizkit::PlaneLayer, a concrete
 * @ref atlas::vizkit::GeometryLayer implementation used to visualize a plane
 * associated with an @ref atlas::Unit.
 *
 * ## Purpose
 * Since an analytic plane is infinite, it cannot be rendered directly in a
 * finite scene. This layer instead visualizes the plane using a finite local
 * patch whose size is controlled by an extent parameter.
 *
 * The plane layer is useful for:
 * - debugging analytic plane geometry,
 * - inspecting collision or boundary surfaces,
 * - visualizing support planes and half-space boundaries,
 * - lightweight wireframe rendering in Vizkit.
 *
 * ## Rendering model
 * The layer generates a finite local-space representation of the plane, usually
 * as a square or rectangular wireframe patch centered around the plane origin.
 *
 * A typical representation includes:
 * - boundary edges of the finite patch,
 * - optionally cross lines or guide lines depending on the implementation in
 *   `plane_layer.hpp`.
 *
 * ## Extent
 * The `_extent` parameter controls the half-size or size scale of the rendered
 * patch, depending on the implementation convention.
 *
 * Larger extents:
 * - make the plane easier to see in large scenes,
 * - provide more spatial context,
 * but also:
 * - increase the likelihood of visual overlap with unrelated geometry.
 *
 * ## Coordinate convention
 * Geometry is built in the unit's **local space**. The base
 * @ref GeometryLayer is responsible for:
 * - transforming local vertices into world space using the unit sync,
 * - managing GPU-side buffers,
 * - issuing the final draw calls.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target unit,
 * - configure the rendered plane extent,
 * - construct the layer by value or shared ownership.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::PlaneLayer<float>::builder()
 *     .with_unit(unit)
 *     .with_extent(10.0f)
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
 * @brief Geometry layer that renders a finite wireframe representation of a plane.
 *
 * @details
 * @ref PlaneLayer extracts plane geometry from a bound @ref atlas::Unit and
 * emits local-space line vertices representing a finite patch of that plane.
 *
 * Because a mathematical plane is infinite, the layer must approximate it with
 * a bounded visual representation. The patch size is controlled by @ref _extent.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - storing the rendered plane extent,
 * - generating local-space line geometry for the plane patch,
 * - passing that geometry to the base @ref GeometryLayer.
 *
 * The base layer handles:
 * - synchronization into world space,
 * - OpenGL buffer management,
 * - per-frame rendering integration.
 *
 * ## Visual interpretation
 * The rendered result is intended as a visualization aid only. It does not alter
 * the analytic behavior of the underlying plane geometry used elsewhere in the
 * runtime.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class PlaneLayer final : public GeometryLayer<T> {
public:
    /**
     * @brief Fluent builder for constructing @ref PlaneLayer.
     */
    class Builder;

public:
    /**
     * @brief Construct a plane layer bound to a unit.
     *
     * @param unit Host-side shared pointer to the unit whose plane geometry will be visualized.
     * @param extent Finite size parameter used for the rendered plane patch.
     */
    ATLAS_HOST explicit PlaneLayer(
        const atlas::UnitHostPtr<T>& unit,
        T extent = T(5));

    /**
     * @brief Create a fluent builder for @ref PlaneLayer.
     *
     * @return A default-initialized builder.
     */
    static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~PlaneLayer() override = default;

protected:
    /**
     * @brief Build local-space line geometry for the finite plane patch.
     *
     * @details
     * This function is called by the base @ref GeometryLayer when the local
     * geometry representation needs to be generated or refreshed.
     *
     * A typical implementation constructs a square or rectangular wireframe patch
     * lying in the plane's local frame, with scale determined by @ref _extent.
     *
     * @param positions Output vector receiving local-space line vertices.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /**
     * @brief Finite size parameter for the rendered plane patch.
     *
     * @details
     * Controls how large the visualized plane region appears in local space.
     */
    T _extent;
};

/**
 * @brief Fluent builder for @ref PlaneLayer.
 *
 * @details
 * The builder stages configuration for a plane visualization layer and
 * constructs either:
 * - a value instance of @ref PlaneLayer, or
 * - a `std::shared_ptr<PlaneLayer>`.
 *
 * ## Configurable state
 * The builder stages:
 * - the target @ref atlas::UnitHostPtr,
 * - the finite rendering extent of the plane patch.
 *
 * ## Validation
 * The builder is expected to ensure:
 * - the unit pointer is non-null,
 * - the extent is positive and suitable for rendering.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class PlaneLayer<T>::Builder {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - no target unit,
     * - a default extent of `5`.
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
     * @brief Set the finite extent used for rendering the plane patch.
     *
     * @param extent Extent value used for plane visualization.
     * @return Reference to this builder for fluent chaining.
     */
    Builder&
    with_extent(T extent) noexcept;

    /**
     * @brief Build a @ref PlaneLayer instance.
     *
     * @details
     * Validates the staged configuration and constructs the final layer by value.
     *
     * @return Constructed plane layer.
     */
    PlaneLayer
    build() const;

    /**
     * @brief Build a shared pointer to a @ref PlaneLayer.
     *
     * @details
     * Validates the staged configuration and constructs the final layer under
     * shared ownership.
     *
     * @return `std::shared_ptr<PlaneLayer>` owning the constructed layer.
     */
    std::shared_ptr<PlaneLayer>
    make_shared() const;

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @details
     * Ensures that required configuration is present and that the staged extent
     * value is suitable for building a visible finite plane patch.
     */
    void
    validate() const;

private:
    /**
     * @brief Target unit to visualize.
     */
    atlas::UnitHostPtr<T> _unit = nullptr;

    /**
     * @brief Staged finite extent for the rendered plane patch.
     */
    T _extent = T(5);
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/plane_layer.hpp>

#endif