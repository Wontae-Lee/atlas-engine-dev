#pragma once

/**
 * @file geometry_operator.h
 * @brief Declares the backend-portable tagged-union geometry operator.
 *
 * GeometryOperator erases concrete geometry type into a value object that can
 * be copied into kernels and other device-friendly runtime structures.
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

/**
 * @brief Tagged-union wrapper over all supported geometry operators.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct GeometryOperator {
    GeometryType type = GeometryType::Sphere; ///< Active geometry variant.

    union {
        BoxGeometryOperator<T> box; ///< Box operator.
        CircleGeometryOperator<T> circle; ///< Circle operator.
        CylinderGeometryOperator<T> cylinder; ///< Cylinder operator.
        PlaneGeometryOperator<T> plane; ///< Plane operator.
        SphereGeometryOperator<T> sphere; ///< Sphere operator.
        TriangleGeometryOperator<T> triangle; ///< Triangle operator.
        TriangleMeshGeometryOperator<T> triangle_mesh; ///< Triangle-mesh operator.
    };

    ATLAS_ALL_DEVICE
    GeometryOperator() noexcept;
    ATLAS_ALL_DEVICE
    GeometryOperator(const GeometryOperator& other) noexcept;

    ATLAS_ALL_DEVICE GeometryOperator&
    operator=(const GeometryOperator& other) noexcept;

    ATLAS_ALL_DEVICE explicit GeometryOperator(const BoxGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const CircleGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const CylinderGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const PlaneGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const SphereGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleGeometryOperator<T>& op);
    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleMeshGeometryOperator<T>& op);

    /**
     * @brief Dispatches the closest-point query to the active geometry variant.
     */
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

}

#include <atlas/geometry/geometry_operator.hpp>
