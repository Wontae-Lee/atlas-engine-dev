#pragma once

#include <limits>

namespace atlas::geometry {

/* GeometryOperator<T> (tagged union dispatcher)                           */
/* ====================================================================== */

template <typename T>

GeometryOperator<T>::GeometryOperator() noexcept
    : type(GeometryType::Sphere)
    , sphere() {
    // Default query operator is a sphere operator.
    //
    // Rationale:
    // - Safe default shape.
    // - Ensures GeometryOperator<T> is always usable after default construction.
    //
    // Note:
    // - sphere() here is default-constructed SphereGeometryOperator<T>,
    //   which is expected to be a lightweight POD-like object.
}

template <typename T>

GeometryOperator<T>::GeometryOperator(const GeometryOperator& other) noexcept
    : type(other.type) {
    // Copy constructor for tagged union.
    //
    // Rule:
    // - Copy ONLY the active member selected by `type`.
    // - For an invalid `type`, fall back to sphere for safety.
    switch (type) {
    case GeometryType::Box:
        box = other.box;
        return;
    case GeometryType::Cylinder:
        cylinder = other.cylinder;
        return;
    case GeometryType::Plane:
        plane = other.plane;
        return;
    case GeometryType::Sphere:
        sphere = other.sphere;
        return;
    case GeometryType::Triangle:
        triangle = other.triangle;
        return;
    case GeometryType::TriangleMesh:
        triangle_mesh = other.triangle_mesh;
        return;
    default:
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return;
    }
}

template <typename T>
GeometryOperator<T>&
GeometryOperator<T>::operator=(const GeometryOperator& other) noexcept {
    // Copy assignment for tagged union.
    if (this == &other) return *this;

    type = other.type;

    switch (type) {
    case GeometryType::Box:
        box = other.box;
        return *this;
    case GeometryType::Cylinder:
        cylinder = other.cylinder;
        return *this;
    case GeometryType::Plane:
        plane = other.plane;
        return *this;
    case GeometryType::Sphere:
        sphere = other.sphere;
        return *this;
    case GeometryType::Triangle:
        triangle = other.triangle;
        return *this;
    case GeometryType::TriangleMesh:
        triangle_mesh = other.triangle_mesh;
        return *this;
    default:
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return *this;
    }
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const BoxGeometryOperator<T>& op)
    : type(GeometryType::Box)
    , box(op) {
    // Construct GeometryOperator as "Box" variant.
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const CylinderGeometryOperator<T>& op)
    : type(GeometryType::Cylinder)
    , cylinder(op) {
    // Construct GeometryOperator as "Cylinder" variant.
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const PlaneGeometryOperator<T>& op)
    : type(GeometryType::Plane)
    , plane(op) {
    // Construct GeometryOperator as "Plane" variant.
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const SphereGeometryOperator<T>& op)
    : type(GeometryType::Sphere)
    , sphere(op) {
    // Construct GeometryOperator as "Sphere" variant.
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const TriangleGeometryOperator<T>& op)
    : type(GeometryType::Triangle)
    , triangle(op) {
    // Construct GeometryOperator as "Triangle" variant.
}

template <typename T>
ATLAS_HOST
GeometryOperator<T>::GeometryOperator(const TriangleMeshGeometryOperator<T>& op)
    : type(GeometryType::TriangleMesh)
    , triangle_mesh(op) {
    // Construct GeometryOperator as "TriangleMesh" variant.
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch to the active operator.
    switch (type) {
    case GeometryType::Box:
        return box.closest_point(p);
    case GeometryType::Cylinder:
        return cylinder.closest_point(p);
    case GeometryType::Plane:
        return plane.closest_point(p);
    case GeometryType::Sphere:
        return sphere.closest_point(p);
    case GeometryType::Triangle:
        return triangle.closest_point(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.closest_point(p);
    default:
        // Defensive fallback: return input.
        return p;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch to the active operator.
    switch (type) {
    case GeometryType::Box:
        return box.closest_normal(p);
    case GeometryType::Cylinder:
        return cylinder.closest_normal(p);
    case GeometryType::Plane:
        return plane.closest_normal(p);
    case GeometryType::Sphere:
        return sphere.closest_normal(p);
    case GeometryType::Triangle:
        return triangle.closest_normal(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.closest_normal(p);
    default:
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
T
GeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch signed-distance query.
    switch (type) {
    case GeometryType::Box:
        return box.signed_distance(p);
    case GeometryType::Cylinder:
        return cylinder.signed_distance(p);
    case GeometryType::Plane:
        return plane.signed_distance(p);
    case GeometryType::Sphere:
        return sphere.signed_distance(p);
    case GeometryType::Triangle:
        return triangle.signed_distance(p);
    case GeometryType::TriangleMesh:
        return triangle_mesh.signed_distance(p);
    default:
        return std::numeric_limits<T>::infinity();
    }
}

template <typename T>
bool
GeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    switch (type) {
    case GeometryType::Box:
        return box.is_inside(p, tolerance);
    case GeometryType::Cylinder:
        return cylinder.is_inside(p, tolerance);
    case GeometryType::Plane:
        return plane.is_inside(p, tolerance);
    case GeometryType::Sphere:
        return sphere.is_inside(p, tolerance);
    case GeometryType::Triangle:
        return triangle.is_inside(p, tolerance);
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_inside(p, tolerance);
    default:
        return false;
    }
}

template <typename T>
bool
GeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    switch (type) {
    case GeometryType::Box:
        return box.is_on_surface(p, tolerance);
    case GeometryType::Cylinder:
        return cylinder.is_on_surface(p, tolerance);
    case GeometryType::Plane:
        return plane.is_on_surface(p, tolerance);
    case GeometryType::Sphere:
        return sphere.is_on_surface(p, tolerance);
    case GeometryType::Triangle:
        return triangle.is_on_surface(p, tolerance);
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_on_surface(p, tolerance);
    default:
        return false;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::centroid() const noexcept {
    // Dispatch centroid query.
    switch (type) {
    case GeometryType::Box:
        return box.centroid();
    case GeometryType::Cylinder:
        return cylinder.centroid();
    case GeometryType::Plane:
        return plane.centroid();
    case GeometryType::Sphere:
        return sphere.centroid();
    case GeometryType::Triangle:
        return triangle.centroid();
    case GeometryType::TriangleMesh:
        return triangle_mesh.centroid();
    default:
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
GeometryOperator<T>::bound() const noexcept {
    // Dispatch bounding-box query.
    switch (type) {
    case GeometryType::Box:
        return box.bound();
    case GeometryType::Cylinder:
        return cylinder.bound();
    case GeometryType::Plane:
        return plane.bound();
    case GeometryType::Sphere:
        return sphere.bound();
    case GeometryType::Triangle:
        return triangle.bound();
    case GeometryType::TriangleMesh:
        return triangle_mesh.bound();
    default:
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }
}

template <typename T>
bool
GeometryOperator<T>::is_valid() const noexcept {
    // Dispatch validity query.
    switch (type) {
    case GeometryType::Box:
        return box.is_valid();
    case GeometryType::Cylinder:
        return cylinder.is_valid();
    case GeometryType::Plane:
        return plane.is_valid();
    case GeometryType::Sphere:
        return sphere.is_valid();
    case GeometryType::Triangle:
        return triangle.is_valid();
    case GeometryType::TriangleMesh:
        return triangle_mesh.is_valid();
    default:
        return false;
    }
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    switch (type) {
    case GeometryType::Box:
        return box.trace(ray);
    case GeometryType::Cylinder:
        return cylinder.trace(ray);
    case GeometryType::Plane:
        return plane.trace(ray);
    case GeometryType::Sphere:
        return sphere.trace(ray);
    case GeometryType::Triangle:
        return triangle.trace(ray);
    case GeometryType::TriangleMesh:
        return triangle_mesh.trace(ray);
    default:
        return HitSurface<T> {};
    }
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

} // namespace atlas::geometry
