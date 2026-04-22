#pragma once

#include <limits>

namespace atlas::geometry {

template <typename T>
GeometryOperator<T>::GeometryOperator() noexcept
    : type(GeometryType::Sphere)
    , sphere() {
    // Default-construct the generic geometry operator in a valid state.
    //
    // Design choice:
    // - use `Sphere` as the default active discriminator
    // - default-construct the matching `sphere` operator member
    //
    // This ensures the tagged-union-like object always starts with a
    // well-defined active geometry variant.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const GeometryOperator& other) noexcept
    : type(other.type) {
    // Copy the active geometry type first so dispatch below knows which
    // concrete operator member should be copied into this object.
    //
    // Only the operator associated with the active `type` is considered
    // semantically valid at a time.

    switch (type) {
    case GeometryType::Box:
        // Copy the box operator when the source stores a box.
        box = other.box;
        return;

    case GeometryType::Circle:
        // Copy the circle operator when the source stores a circle.
        circle = other.circle;
        return;

    case GeometryType::Cylinder:
        // Copy the cylinder operator when the source stores a cylinder.
        cylinder = other.cylinder;
        return;

    case GeometryType::Plane:
        // Copy the plane operator when the source stores a plane.
        plane = other.plane;
        return;

    case GeometryType::Sphere:
        // Copy the sphere operator when the source stores a sphere.
        sphere = other.sphere;
        return;

    case GeometryType::Square:
        // Copy the square operator when the source stores a square.
        square = other.square;
        return;

    case GeometryType::Triangle:
        // Copy the triangle operator when the source stores a triangle.
        triangle = other.triangle;
        return;

    case GeometryType::TriangleMesh:
        // Copy the triangle-mesh operator when the source stores a mesh.
        triangle_mesh = other.triangle_mesh;
        return;

    default:
        // Defensive fallback:
        // - normalize the discriminator to `Sphere`
        // - copy the sphere operator
        //
        // This keeps the object in a valid, usable state even if the
        // source type value is unexpected.
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return;
    }
}

template <typename T>
GeometryOperator<T>&
GeometryOperator<T>::operator=(const GeometryOperator& other) noexcept {
    // Guard against self-assignment to avoid redundant work.
    if (this == &other) return *this;

    // Copy the active geometry discriminator first.
    type = other.type;

    // Copy only the concrete operator associated with the active type.
    switch (type) {
    case GeometryType::Box:
        // Assign the box operator payload.
        box = other.box;
        return *this;

    case GeometryType::Circle:
        // Assign the circle operator payload.
        circle = other.circle;
        return *this;

    case GeometryType::Cylinder:
        // Assign the cylinder operator payload.
        cylinder = other.cylinder;
        return *this;

    case GeometryType::Plane:
        // Assign the plane operator payload.
        plane = other.plane;
        return *this;

    case GeometryType::Sphere:
        // Assign the sphere operator payload.
        sphere = other.sphere;
        return *this;

    case GeometryType::Square:
        // Assign the square operator payload.
        square = other.square;
        return *this;

    case GeometryType::Triangle:
        // Assign the triangle operator payload.
        triangle = other.triangle;
        return *this;

    case GeometryType::TriangleMesh:
        // Assign the triangle-mesh operator payload.
        triangle_mesh = other.triangle_mesh;
        return *this;

    default:
        // Defensive fallback:
        // - restore a known valid discriminator
        // - assign the sphere payload
        type   = GeometryType::Sphere;
        sphere = other.sphere;
        return *this;
    }
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const BoxGeometryOperator<T>& op)
    : type(GeometryType::Box)
    , box(op) {
    // Construct a generic geometry operator from a concrete box operator.
    //
    // The discriminator is set to `Box` so all later virtual-by-switch
    // dispatch routes to `box`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CircleGeometryOperator<T>& op)
    : type(GeometryType::Circle)
    , circle(op) {
    // Construct a generic geometry operator from a concrete circle operator.
    //
    // The discriminator is set to `Circle` so all later dispatch routes
    // to `circle`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CylinderGeometryOperator<T>& op)
    : type(GeometryType::Cylinder)
    , cylinder(op) {
    // Construct a generic geometry operator from a concrete cylinder operator.
    //
    // The discriminator is set to `Cylinder` so all later dispatch routes
    // to `cylinder`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const PlaneGeometryOperator<T>& op)
    : type(GeometryType::Plane)
    , plane(op) {
    // Construct a generic geometry operator from a concrete plane operator.
    //
    // The discriminator is set to `Plane` so all later dispatch routes
    // to `plane`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const SphereGeometryOperator<T>& op)
    : type(GeometryType::Sphere)
    , sphere(op) {
    // Construct a generic geometry operator from a concrete sphere operator.
    //
    // The discriminator is set to `Sphere` so all later dispatch routes
    // to `sphere`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const SquareGeometryOperator<T>& op)
    : type(GeometryType::Square)
    , square(op) {
    // Construct a generic geometry operator from a concrete square operator.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleGeometryOperator<T>& op)
    : type(GeometryType::Triangle)
    , triangle(op) {
    // Construct a generic geometry operator from a concrete triangle operator.
    //
    // The discriminator is set to `Triangle` so all later dispatch routes
    // to `triangle`.
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleMeshGeometryOperator<T>& op)
    : type(GeometryType::TriangleMesh)
    , triangle_mesh(op) {
    // Construct a generic geometry operator from a concrete triangle-mesh operator.
    //
    // The discriminator is set to `TriangleMesh` so all later dispatch routes
    // to `triangle_mesh`.
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch the closest-point query to the concrete operator selected
    // by the active geometry type.
    //
    // This provides a unified interface over several geometry primitives.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box closest-point implementation.
        return box.closest_point(p);

    case GeometryType::Circle:
        // Forward to the circle closest-point implementation.
        return circle.closest_point(p);

    case GeometryType::Cylinder:
        // Forward to the cylinder closest-point implementation.
        return cylinder.closest_point(p);

    case GeometryType::Plane:
        // Forward to the plane closest-point implementation.
        return plane.closest_point(p);

    case GeometryType::Sphere:
        // Forward to the sphere closest-point implementation.
        return sphere.closest_point(p);

    case GeometryType::Square:
        // Forward to the square closest-point implementation.
        return square.closest_point(p);

    case GeometryType::Triangle:
        // Forward to the triangle closest-point implementation.
        return triangle.closest_point(p);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh closest-point implementation.
        return triangle_mesh.closest_point(p);

    default:
        // Fallback for unexpected discriminator values:
        // return the input point unchanged.
        return p;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch the closest-surface-normal query to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box normal query.
        return box.closest_normal(p);

    case GeometryType::Circle:
        // Forward to the circle normal query.
        return circle.closest_normal(p);

    case GeometryType::Cylinder:
        // Forward to the cylinder normal query.
        return cylinder.closest_normal(p);

    case GeometryType::Plane:
        // Forward to the plane normal query.
        return plane.closest_normal(p);

    case GeometryType::Sphere:
        // Forward to the sphere normal query.
        return sphere.closest_normal(p);

    case GeometryType::Square:
        // Forward to the square normal query.
        return square.closest_normal(p);

    case GeometryType::Triangle:
        // Forward to the triangle normal query.
        return triangle.closest_normal(p);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh normal query.
        return triangle_mesh.closest_normal(p);

    default:
        // Fallback for unexpected discriminator values:
        // return a zero vector to indicate that no valid normal is available.
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
T
GeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Dispatch the signed-distance query to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box signed-distance implementation.
        return box.signed_distance(p);

    case GeometryType::Circle:
        // Forward to the circle signed-distance implementation.
        return circle.signed_distance(p);

    case GeometryType::Cylinder:
        // Forward to the cylinder signed-distance implementation.
        return cylinder.signed_distance(p);

    case GeometryType::Plane:
        // Forward to the plane signed-distance implementation.
        return plane.signed_distance(p);

    case GeometryType::Sphere:
        // Forward to the sphere signed-distance implementation.
        return sphere.signed_distance(p);

    case GeometryType::Square:
        // Forward to the square signed-distance implementation.
        return square.signed_distance(p);

    case GeometryType::Triangle:
        // Forward to the triangle signed-distance implementation.
        return triangle.signed_distance(p);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh signed-distance implementation.
        return triangle_mesh.signed_distance(p);

    default:
        // Fallback for unexpected discriminator values:
        // report an infinite distance so callers can treat the query as invalid.
        return std::numeric_limits<T>::infinity();
    }
}

template <typename T>
bool
GeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Dispatch inside-classification to the active geometry variant.
    //
    // `tolerance` is forwarded unchanged so each concrete operator can apply
    // its own boundary semantics consistently.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box inside test.
        return box.is_inside(p, tolerance);

    case GeometryType::Circle:
        // Forward to the circle inside test.
        return circle.is_inside(p, tolerance);

    case GeometryType::Cylinder:
        // Forward to the cylinder inside test.
        return cylinder.is_inside(p, tolerance);

    case GeometryType::Plane:
        // Forward to the plane inside test.
        return plane.is_inside(p, tolerance);

    case GeometryType::Sphere:
        // Forward to the sphere inside test.
        return sphere.is_inside(p, tolerance);

    case GeometryType::Square:
        // Forward to the square inside test.
        return square.is_inside(p, tolerance);

    case GeometryType::Triangle:
        // Forward to the triangle inside test.
        return triangle.is_inside(p, tolerance);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh inside test.
        return triangle_mesh.is_inside(p, tolerance);

    default:
        // Fallback for unexpected discriminator values:
        // treat the point as not inside.
        return false;
    }
}

template <typename T>
bool
GeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Dispatch surface-band classification to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box surface test.
        return box.is_on_surface(p, tolerance);

    case GeometryType::Circle:
        // Forward to the circle surface test.
        return circle.is_on_surface(p, tolerance);

    case GeometryType::Cylinder:
        // Forward to the cylinder surface test.
        return cylinder.is_on_surface(p, tolerance);

    case GeometryType::Plane:
        // Forward to the plane surface test.
        return plane.is_on_surface(p, tolerance);

    case GeometryType::Sphere:
        // Forward to the sphere surface test.
        return sphere.is_on_surface(p, tolerance);

    case GeometryType::Square:
        // Forward to the square surface test.
        return square.is_on_surface(p, tolerance);

    case GeometryType::Triangle:
        // Forward to the triangle surface test.
        return triangle.is_on_surface(p, tolerance);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh surface test.
        return triangle_mesh.is_on_surface(p, tolerance);

    default:
        // Fallback for unexpected discriminator values:
        // treat the point as not being on the surface.
        return false;
    }
}

template <typename T>
atlas::math::Vector<T, 3>
GeometryOperator<T>::centroid() const noexcept {
    // Dispatch centroid computation to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box centroid implementation.
        return box.centroid();

    case GeometryType::Circle:
        // Forward to the circle centroid implementation.
        return circle.centroid();

    case GeometryType::Cylinder:
        // Forward to the cylinder centroid implementation.
        return cylinder.centroid();

    case GeometryType::Plane:
        // Forward to the plane centroid implementation.
        return plane.centroid();

    case GeometryType::Sphere:
        // Forward to the sphere centroid implementation.
        return sphere.centroid();

    case GeometryType::Square:
        // Forward to the square centroid implementation.
        return square.centroid();

    case GeometryType::Triangle:
        // Forward to the triangle centroid implementation.
        return triangle.centroid();

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh centroid implementation.
        return triangle_mesh.centroid();

    default:
        // Fallback for unexpected discriminator values:
        // return the origin as a neutral default.
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
GeometryOperator<T>::bound() const noexcept {
    // Dispatch axis-aligned bounding-box computation to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box bound computation.
        return box.bound();

    case GeometryType::Circle:
        // Forward to the circle bound computation.
        return circle.bound();

    case GeometryType::Cylinder:
        // Forward to the cylinder bound computation.
        return cylinder.bound();

    case GeometryType::Plane:
        // Forward to the plane bound computation.
        return plane.bound();

    case GeometryType::Sphere:
        // Forward to the sphere bound computation.
        return sphere.bound();

    case GeometryType::Square:
        // Forward to the square bound computation.
        return square.bound();

    case GeometryType::Triangle:
        // Forward to the triangle bound computation.
        return triangle.bound();

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh bound computation.
        return triangle_mesh.bound();

    default:
        // Fallback for unexpected discriminator values:
        // return a default-constructed AABB.
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }
}

template <typename T>
bool
GeometryOperator<T>::is_valid() const noexcept {
    // Dispatch validity testing to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box validity test.
        return box.is_valid();

    case GeometryType::Circle:
        // Forward to the circle validity test.
        return circle.is_valid();

    case GeometryType::Cylinder:
        // Forward to the cylinder validity test.
        return cylinder.is_valid();

    case GeometryType::Plane:
        // Forward to the plane validity test.
        return plane.is_valid();

    case GeometryType::Sphere:
        // Forward to the sphere validity test.
        return sphere.is_valid();

    case GeometryType::Square:
        // Forward to the square validity test.
        return square.is_valid();

    case GeometryType::Triangle:
        // Forward to the triangle validity test.
        return triangle.is_valid();

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh validity test.
        return triangle_mesh.is_valid();

    default:
        // Fallback for unexpected discriminator values:
        // treat the operator as invalid.
        return false;
    }
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Dispatch ray-intersection tracing to the active geometry variant.

    switch (type) {
    case GeometryType::Box:
        // Forward to the box ray-trace implementation.
        return box.trace(ray);

    case GeometryType::Circle:
        // Forward to the circle ray-trace implementation.
        return circle.trace(ray);

    case GeometryType::Cylinder:
        // Forward to the cylinder ray-trace implementation.
        return cylinder.trace(ray);

    case GeometryType::Plane:
        // Forward to the plane ray-trace implementation.
        return plane.trace(ray);

    case GeometryType::Sphere:
        // Forward to the sphere ray-trace implementation.
        return sphere.trace(ray);

    case GeometryType::Square:
        // Forward to the square ray-trace implementation.
        return square.trace(ray);

    case GeometryType::Triangle:
        // Forward to the triangle ray-trace implementation.
        return triangle.trace(ray);

    case GeometryType::TriangleMesh:
        // Forward to the triangle-mesh ray-trace implementation.
        return triangle_mesh.trace(ray);

    default:
        // Fallback for unexpected discriminator values:
        // report no intersection.
        return HitSurface<T> {};
    }
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Convenience call-operator wrapper that forwards directly to `trace()`.
    //
    // This allows the generic geometry operator to be used like a callable
    // intersection functor.
    return trace(ray);
}

} // namespace atlas::geometry
