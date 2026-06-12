#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

#include <stdexcept>

namespace atlas {

template <typename T>
struct GeometryOperator {

    GeometryType type = GeometryType::Sphere;

    union {

        BoxGeometryOperator<T> box;

        CircleGeometryOperator<T> circle;

        CylinderGeometryOperator<T> cylinder;

        PlaneGeometryOperator<T> plane;

        SphereGeometryOperator<T> sphere;

        SquareGeometryOperator<T> square;

        TriangleGeometryOperator<T> triangle;

        TriangleMeshGeometryOperator<T> triangle_mesh;
    };

    ATLAS_ALL_DEVICE
    GeometryOperator() noexcept;

    ATLAS_ALL_DEVICE
    GeometryOperator(const GeometryOperator& other) noexcept;

    ATLAS_ALL_DEVICE GeometryOperator&
    operator=(const GeometryOperator& other) noexcept;

    ATLAS_ALL_DEVICE
    ~GeometryOperator() noexcept;

    ATLAS_ALL_DEVICE explicit GeometryOperator(const BoxGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const CircleGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const CylinderGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const PlaneGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const SphereGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const SquareGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE explicit GeometryOperator(const TriangleMeshGeometryOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::Ray<T>& ray) const noexcept;
};

template <typename T>
ATLAS_HOST ATLAS_NODISCARD GeometryOperator<T>
make_device_geometry_view(const Geometry<T>& geometry) {
    const auto* factory = dynamic_cast<const DeviceGeometryViewFactory<T>*>(&geometry);

    if (!factory) {
        throw std::runtime_error("Geometry does not provide a device geometry view.");
    }

    return factory->make_device_geometry_view();
}

}

#include <atlas/geometry/geometry_operator.hpp>