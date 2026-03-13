#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that samples a sphere into a triangle-style vertex grid.
 *
 * @tparam T Scalar type used by queried geometry and generated vertices.
 *
 * The layer reads a sphere center and radius from the bound unit and generates
 * points on the sphere surface using angular subdivision parameters
 * (`_slices`, `_stacks`). The vertices are emitted in local space and later
 * passed through the standard @ref GeometryLayer synchronization and rendering
 * pipeline.
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::Sphere` with valid center and radius data.
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_TRIANGLES` via the base class
 * - Surface sampling parameters:
 *   - `_slices`: azimuthal subdivisions around the sphere
 *   - `_stacks`: polar subdivisions from top to bottom
 *
 * @note The current implementation emits one sampled vertex per `(stack, slice)`
 * grid cell and does not itself assemble indexed triangles in the header-level
 * contract. This class therefore documents the produced surface sampling rather
 * than promising a specific closed mesh topology.
 */
template <typename T>
class SphereLayer final : public GeometryLayer<T> {
public:
    /** @brief Fluent builder for constructing @ref SphereLayer instances. */
    class Builder;

public:
    /**
     * @brief Constructs a sphere layer.
     *
     * @param unit Unit whose query operator provides sphere geometry.
     * @param slices Number of azimuthal subdivisions around the sphere.
     * @param stacks Number of polar subdivisions from pole to pole.
     */
    SphereLayer(const atlas::UnitHostPtr<T>& unit, int slices, int stacks);

    /**
     * @brief Creates a builder for @ref SphereLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder() noexcept;

protected:
    /**
     * @brief Generates the local-space sphere surface samples.
     *
     * @param positions Destination buffer receiving sampled sphere vertices.
     *
     * If the unit is null, the query type is not `Sphere`, or the required
     * sphere parameters are absent, the output is cleared and left empty.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /** @brief Number of azimuthal subdivisions around the sphere. */
    int _slices;
    /** @brief Number of polar subdivisions from north to south. */
    int _stacks;
};

template <typename T>
class SphereLayer<T>::Builder {
public:
    /**
     * @brief Sets the unit supplying sphere query data.
     *
     * @param unit Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;
    /**
     * @brief Sets the azimuthal subdivision count.
     *
     * @param slices Number of longitudinal slices.
     * @return Reference to this builder.
     */
    Builder&
    with_slices(int slices) noexcept;
    /**
     * @brief Sets the polar subdivision count.
     *
     * @param stacks Number of latitudinal stacks.
     * @return Reference to this builder.
     */
    Builder&
    with_stacks(int stacks) noexcept;

    /**
     * @brief Builds a @ref SphereLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If required builder state is missing.
     */
    SphereLayer
    build() const;
    /**
     * @brief Builds a heap-allocated @ref SphereLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If required builder state is missing.
     */
    std::shared_ptr<SphereLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that the builder can construct a sphere layer.
     *
     * @throws std::runtime_error If no unit has been provided.
     *
     * @note The current builder implementation only checks that the unit is
     * non-null; geometric consistency is deferred to runtime generation.
     */
    void
    validate() const;

private:
    /** @brief Unit whose sphere query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
    /** @brief Requested azimuthal subdivision count. */
    int _slices                 = 32;
    /** @brief Requested polar subdivision count. */
    int _stacks                 = 16;
};

}

#include <vizkit/layer/geometry/sphere_layer.hpp>

#endif
