#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that renders a cylinder as a triangle mesh.
 *
 * @tparam T Scalar type used by the queried geometry and generated vertices.
 *
 * The layer reads cylinder parameters from the bound unit and tessellates the
 * shape into triangles for the side wall, top cap, and bottom cap. Generated
 * vertices remain in local space; world-space synchronization and GPU upload
 * are handled by @ref GeometryLayer.
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::Cylinder` with valid center, radius, and
 * height values.
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_TRIANGLES`
 * - Tessellation control: `_slices` angular subdivisions around the cylinder
 * - Each slice contributes 4 triangles:
 *   - 2 for the side wall
 *   - 1 for the top cap
 *   - 1 for the bottom cap
 *
 * @note The constructor clamps the slice count to at least 3 so the generated
 * mesh is always topologically meaningful.
 */
template <typename T>
class CylinderLayer final : public GeometryLayer<T> {
public:
    /** @brief Fluent builder for constructing @ref CylinderLayer instances. */
    class Builder;

public:
    /**
     * @brief Constructs a cylinder layer.
     *
     * @param unit Unit whose query operator provides cylinder geometry.
     * @param slices Number of angular subdivisions used during tessellation.
     */
    ATLAS_HOST explicit CylinderLayer(
        const atlas::UnitHostPtr<T>& unit,
        int slices = 32);

    /**
     * @brief Creates a builder for @ref CylinderLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder() noexcept;

    ~CylinderLayer() override = default;

protected:
    /**
     * @brief Generates the local-space triangle list for the cylinder.
     *
     * @param positions Destination buffer receiving triangle vertices.
     *
     * If the bound unit is null, the query type is not `Cylinder`, required
     * parameters are missing, or radius/height are non-positive, the output is
     * cleared and left empty.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /** @brief Number of angular subdivisions around the cylinder axis. */
    int _slices;
};

template <typename T>
class CylinderLayer<T>::Builder {
public:
    /** @brief Creates an empty builder with a default 32-slice tessellation. */
    Builder() = default;

    /**
     * @brief Sets the unit supplying cylinder query data.
     *
     * @param unit Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;
    /**
     * @brief Sets the angular tessellation density.
     *
     * @param slices Desired number of slices around the cylinder.
     * @return Reference to this builder.
     */
    Builder&
    with_slices(int slices) noexcept;

    /**
     * @brief Builds a @ref CylinderLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If the builder state is invalid.
     */
    CylinderLayer
    build() const;
    /**
     * @brief Builds a heap-allocated @ref CylinderLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If the builder state is invalid.
     */
    std::shared_ptr<CylinderLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that construction inputs are complete and coherent.
     *
     * Validation checks:
     * - unit is not null
     * - query type is `Cylinder`
     * - center, radius, and height are present
     * - radius and height are strictly positive
     *
     * @throws std::runtime_error If any requirement is violated.
     */
    void
    validate() const;

private:
    /** @brief Unit whose cylinder query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
    /** @brief Requested angular subdivision count. */
    int _slices                 = 32;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/cylinder_layer.hpp>

#endif
