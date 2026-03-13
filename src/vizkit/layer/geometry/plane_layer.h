#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/core/macros.h>
#include <atlas/unit/unit.h>
#include <vizkit/layer/geometry/geometry_layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Geometry layer that visualizes an infinite plane as a finite quad.
 *
 * @tparam T Scalar type used by queried geometry and generated vertices.
 *
 * A mathematical plane is unbounded, so this layer renders a finite square
 * patch centered on one point of the queried plane. The patch is built in local
 * space from:
 * - the plane normal
 * - the signed offset
 * - a configurable half-extent used to span two in-plane basis vectors
 *
 * The layer expects the associated unit to expose
 * `atlas::geometry::GeometryType::Plane` with a non-zero normal and a valid
 * offset.
 *
 * Rendering characteristics:
 * - OpenGL primitive mode: `GL_TRIANGLES`
 * - Output topology: two triangles forming one quad
 * - Patch size: side length `2 * extent`
 *
 * @note The generated patch is only a visualization proxy for the plane, not a
 * geometric clipping of an actually infinite primitive.
 */
template <typename T>
class PlaneLayer final : public GeometryLayer<T> {
public:
    /** @brief Fluent builder for constructing @ref PlaneLayer instances. */
    class Builder;

public:
    /**
     * @brief Constructs a plane layer.
     *
     * @param unit Unit whose query operator provides plane geometry.
     * @param extent Half-size of the rendered square patch in each in-plane axis.
     */
    ATLAS_HOST explicit PlaneLayer(
        const atlas::UnitHostPtr<T>& unit,
        T extent = T(5));

    /**
     * @brief Creates a builder for @ref PlaneLayer.
     *
     * @return Default-initialized builder.
     */
    static Builder
    builder() noexcept;

    ~PlaneLayer() override = default;

protected:
    /**
     * @brief Generates the local-space triangle list for the plane patch.
     *
     * @param positions Destination buffer receiving six vertices.
     *
     * The implementation normalizes the plane normal, chooses a reference axis
     * that is not nearly parallel to it, builds an orthonormal in-plane basis,
     * and emits a square patch as two triangles.
     */
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    /** @brief Half-size of the rendered square patch. */
    T _extent;
};

template <typename T>
class PlaneLayer<T>::Builder {
public:
    /** @brief Creates an empty builder with a default extent of 5 units. */
    Builder() = default;

    /**
     * @brief Sets the unit supplying plane query data.
     *
     * @param unit Unit to visualize.
     * @return Reference to this builder.
     */
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& unit) noexcept;
    /**
     * @brief Sets the half-extent of the rendered plane patch.
     *
     * @param extent Positive half-size of the square patch.
     * @return Reference to this builder.
     */
    Builder&
    with_extent(T extent) noexcept;

    /**
     * @brief Builds a @ref PlaneLayer by value.
     *
     * @return Constructed layer.
     *
     * @throws std::runtime_error If the builder state is invalid.
     */
    PlaneLayer
    build() const;
    /**
     * @brief Builds a heap-allocated @ref PlaneLayer.
     *
     * @return Shared pointer owning the constructed layer.
     *
     * @throws std::runtime_error If the builder state is invalid.
     */
    std::shared_ptr<PlaneLayer>
    make_shared() const;

private:
    /**
     * @brief Validates that the builder has enough information to build a layer.
     *
     * Validation checks:
     * - unit is not null
     * - query type is `Plane`
     * - plane normal and offset are present
     * - plane normal is non-zero
     * - extent is strictly positive
     *
     * @throws std::runtime_error If any requirement is violated.
     */
    void
    validate() const;

private:
    /** @brief Unit whose plane query data will be visualized. */
    atlas::UnitHostPtr<T> _unit = nullptr;
    /** @brief Requested half-size of the rendered patch. */
    T _extent                   = T(5);
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/plane_layer.hpp>

#endif
