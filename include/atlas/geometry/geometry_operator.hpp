#pragma once

#include <limits>

namespace atlas::geometry {

template <typename T>
GeometryOperator<T>::GeometryOperator() noexcept
    : type(GeometryType::Sphere)
    , sphere() {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const GeometryOperator& other) noexcept
    : type(other.type) {

    switch (type) {
    case GeometryType::Box:

        box = other.box;
        return;

    case GeometryType::Circle:

        circle = other.circle;
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

    if (this == &other) return *this;

    type = other.type;

    switch (type) {
    case GeometryType::Box:

        box = other.box;
        return *this;

    case GeometryType::Circle:

        circle = other.circle;
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
GeometryOperator<T>::GeometryOperator(const BoxGeometryOperator<T>& op)
    : type(GeometryType::Box)
    , box(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CircleGeometryOperator<T>& op)
    : type(GeometryType::Circle)
    , circle(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CylinderGeometryOperator<T>& op)
    : type(GeometryType::Cylinder)
    , cylinder(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const PlaneGeometryOperator<T>& op)
    : type(GeometryType::Plane)
    , plane(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const SphereGeometryOperator<T>& op)
    : type(GeometryType::Sphere)
    , sphere(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleGeometryOperator<T>& op)
    : type(GeometryType::Triangle)
    , triangle(op) {
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleMeshGeometryOperator<T>& op)
    : type(GeometryType::TriangleMesh)
    , triangle_mesh(op) {
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {

    switch (type) {
    case GeometryType::Box:

        return box.closest_point(p);

    case GeometryType::Circle:

        return circle.closest_point(p);

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

        return p;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {

    switch (type) {
    case GeometryType::Box:

        return box.closest_normal(p);

    case GeometryType::Circle:

        return circle.closest_normal(p);

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

    switch (type) {
    case GeometryType::Box:

        return box.signed_distance(p);

    case GeometryType::Circle:

        return circle.signed_distance(p);

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

    case GeometryType::Circle:

        return circle.is_inside(p, tolerance);

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

    case GeometryType::Circle:

        return circle.is_on_surface(p, tolerance);

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

    switch (type) {
    case GeometryType::Box:

        return box.centroid();

    case GeometryType::Circle:

        return circle.centroid();

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

    switch (type) {
    case GeometryType::Box:

        return box.bound();

    case GeometryType::Circle:

        return circle.bound();

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

    switch (type) {
    case GeometryType::Box:

        return box.is_valid();

    case GeometryType::Circle:

        return circle.is_valid();

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

    case GeometryType::Circle:

        return circle.trace(ray);

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

}