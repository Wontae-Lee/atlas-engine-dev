#pragma once

/**
 * @file geometry_operator.h
 * @brief Unified geometry query and trace dispatcher.
 */

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

namespace atlas::geometry {

template <typename T>
struct GeometryOperator {
    GeometryType type = GeometryType::Sphere;

    union {
        BoxGeometryOperator<T> box;
        CircleGeometryOperator<T> circle;
        CylinderGeometryOperator<T> cylinder;
        PlaneGeometryOperator<T> plane;
        SphereGeometryOperator<T> sphere;
        TriangleGeometryOperator<T> triangle;
        TriangleMeshGeometryOperator<T> triangle_mesh;
    };

    ATLAS_HOST
    GeometryOperator() noexcept;
    ATLAS_HOST
    GeometryOperator(const GeometryOperator& other) noexcept;

    ATLAS_HOST GeometryOperator&
    operator=(const GeometryOperator& other) noexcept;

    ATLAS_HOST explicit GeometryOperator(const BoxGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const CircleGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const CylinderGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const PlaneGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const SphereGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const TriangleGeometryOperator<T>& op);
    ATLAS_HOST explicit GeometryOperator(const TriangleMeshGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

} // namespace atlas::geometry

#include <atlas/geometry/geometry_operator.hpp>
