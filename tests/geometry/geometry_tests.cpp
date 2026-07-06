#include <atlas/geometry/geometry.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

#include <cmath>
#include <gtest/gtest.h>

namespace {

using atlas::Box;
using atlas::Circle;
using atlas::Cylinder;
using atlas::Geometry;
using atlas::GeometryType;
using atlas::HostBuffer;
using atlas::Plane;
using atlas::Sphere;
using atlas::Triangle;
using atlas::TriangleContainer4;
using atlas::TriangleMesh;
using atlas::Vector3;
using atlas::eps;

bool
is_finite_vec(const Vector3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void
expect_operator_interface(const Geometry& geometry,
                          const GeometryType expected_type,
                          const Vector3& query_point) {
    EXPECT_EQ(geometry.type, expected_type);
    EXPECT_TRUE(geometry.is_valid());

    const auto closest_point  = geometry.closest_point(query_point);
    const auto closest_normal = geometry.closest_normal(query_point);
    const auto center         = geometry.centroid();
    const auto bounds         = geometry.bound();
    const auto distance       = geometry.signed_distance(query_point);
    const auto is_inside      = geometry.is_inside(query_point, eps);
    const auto is_on_surface  = geometry.is_on_surface(query_point, eps);

    EXPECT_TRUE(is_finite_vec(closest_point));
    EXPECT_TRUE(is_finite_vec(closest_normal));
    EXPECT_TRUE(is_finite_vec(center));
    EXPECT_TRUE(is_finite_vec(bounds.lower_corner));
    EXPECT_TRUE(is_finite_vec(bounds.upper_corner));
    EXPECT_TRUE(std::isfinite(distance));
    EXPECT_TRUE(is_inside || !is_inside);
    EXPECT_TRUE(is_on_surface || !is_on_surface);
}

}

TEST(Geometry, BoxDispatchesToBoxShape) {
    const auto box = Box::builder()
                         .with_lower_corner(Vector3(-1.0f, -2.0f, -3.0f))
                         .with_upper_corner(Vector3(1.0f, 2.0f, 3.0f))
                         .build();

    expect_operator_interface(
        Geometry(box),
        GeometryType::box,
        Vector3(3.0f, 0.0f, 0.0f));
}

TEST(Geometry, CircleDispatchesToCircleShape) {
    const auto circle = Circle::builder()
                            .with_center(Vector3(0.0f, 0.0f, 0.0f))
                            .with_normal(Vector3(0.0f, 1.0f, 0.0f))
                            .with_radius(2.0f)
                            .build();

    expect_operator_interface(
        Geometry(circle),
        GeometryType::circle,
        Vector3(1.0f, 3.0f, 0.0f));
}

TEST(Geometry, CylinderDispatchesToCylinderShape) {
    const auto cylinder = Cylinder::builder()
                              .with_center(Vector3(0.0f, 0.0f, 0.0f))
                              .with_radius(1.5f)
                              .with_height(4.0f)
                              .build();

    expect_operator_interface(
        Geometry(cylinder),
        GeometryType::cylinder,
        Vector3(2.0f, 0.0f, 0.0f));
}

TEST(Geometry, PlaneDispatchesToPlaneShape) {
    const auto plane = Plane::builder()
                           .with_point_normal(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f))
                           .build();

    expect_operator_interface(
        Geometry(plane),
        GeometryType::plane,
        Vector3(0.0f, 2.0f, 0.0f));
}

TEST(Geometry, SphereDispatchesToSphereShape) {
    const auto sphere = Sphere::builder()
                            .with_center(Vector3(0.0f, 0.0f, 0.0f))
                            .with_radius(2.0f)
                            .build();

    expect_operator_interface(
        Geometry(sphere),
        GeometryType::sphere,
        Vector3(4.0f, 0.0f, 0.0f));
}

TEST(Geometry, TriangleDispatchesToTriangleShape) {
    const auto triangle = Triangle::builder()
                              .with_vertices(
                                  Vector3(0.0f, 0.0f, 0.0f),
                                  Vector3(1.0f, 0.0f, 0.0f),
                                  Vector3(0.0f, 1.0f, 0.0f))
                              .build();

    expect_operator_interface(
        Geometry(triangle),
        GeometryType::triangle,
        Vector3(0.25f, 0.25f, 1.0f));
}

TEST(Geometry, TriangleMeshDispatchesToMeshShape) {
    TriangleContainer4 triangle {};
    triangle[0] = Vector3(0.0f, 0.0f, 0.0f);
    triangle[1] = Vector3(1.0f, 0.0f, 0.0f);
    triangle[2] = Vector3(0.0f, 1.0f, 0.0f);
    triangle[3] = Vector3(0.0f, 0.0f, 0.0f);

    const auto mesh = TriangleMesh::builder()
                          .with_triangles(HostBuffer<TriangleContainer4> { triangle })
                          .build();

    expect_operator_interface(
        mesh.make_device_geometry_view(),
        GeometryType::triangle_mesh,
        Vector3(0.2f, 0.2f, 0.5f));
}
