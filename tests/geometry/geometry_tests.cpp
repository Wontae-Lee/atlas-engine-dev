#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::Circle;
using atlas::Cylinder;
using atlas::Geometry;
using atlas::GeometryHostPtr;
using atlas::HostBuffer;
using atlas::Plane;
using atlas::Sphere;
using atlas::Triangle;
using atlas::TriangleContainer4;
using atlas::TriangleMesh;
using atlas::Vector3F;
using atlas::eps;
using atlas::GeometryType;
using atlas::test::is_finite_vec;

template <typename T>
void
expect_geometry_interface(const GeometryHostPtr<T>& geometry,
                          const GeometryType expected_type,
                          const atlas::Vector3<T>& query_point) {
    ASSERT_NE(geometry, nullptr);

    const Geometry<T>& base = *geometry;

    EXPECT_EQ(base.type(), expected_type);
    EXPECT_TRUE(base.is_valid());

    const auto operator_view = atlas::make_device_geometry_view(base);
    (void)operator_view;

    const auto closest_point = base.closest_point(query_point);
    const auto closest_normal = base.closest_normal(query_point);
    const auto center = base.centroid();
    const auto bounds = base.bound();
    const auto distance = base.signed_distance(query_point);
    const auto is_inside = base.is_inside(query_point, static_cast<T>(eps));
    const auto is_on_surface = base.is_on_surface(query_point, static_cast<T>(eps));

    EXPECT_TRUE(is_finite_vec(closest_point));
    EXPECT_TRUE(is_finite_vec(closest_normal));
    EXPECT_TRUE(is_finite_vec(center));
    EXPECT_TRUE(is_finite_vec(bounds.lower_corner));
    EXPECT_TRUE(is_finite_vec(bounds.upper_corner));
    EXPECT_TRUE(std::isfinite(static_cast<double>(distance)));
    EXPECT_TRUE(is_inside || !is_inside);
    EXPECT_TRUE(is_on_surface || !is_on_surface);
}

} // namespace

TEST(Geometry, BoxImplementsBaseInterface) {
    const auto geometry = Box<float>::builder()
                              .with_lower_corner(Vector3F(-1, -2, -3))
                              .with_upper_corner(Vector3F(1, 2, 3))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Box, Vector3F(3, 0, 0));
}

TEST(Geometry, CircleImplementsBaseInterface) {
    const auto geometry = Circle<float>::builder()
                              .with_center(Vector3F(0, 0, 0))
                              .with_normal(Vector3F(0, 1, 0))
                              .with_radius(static_cast<float>(2))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Circle, Vector3F(1, 3, 0));
}

TEST(Geometry, CylinderImplementsBaseInterface) {
    const auto geometry = Cylinder<float>::builder()
                              .with_center(Vector3F(0, 0, 0))
                              .with_radius(static_cast<float>(1.5))
                              .with_height(static_cast<float>(4))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Cylinder, Vector3F(2, 0, 0));
}

TEST(Geometry, PlaneImplementsBaseInterface) {
    const auto geometry = Plane<float>::builder()
                              .with_point_normal(Vector3F(0, 0, 0), Vector3F(0, 1, 0))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Plane, Vector3F(0, 2, 0));
}

TEST(Geometry, SphereImplementsBaseInterface) {
    const auto geometry = Sphere<float>::builder()
                              .with_center(Vector3F(0, 0, 0))
                              .with_radius(static_cast<float>(2))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Sphere, Vector3F(4, 0, 0));
}

TEST(Geometry, TriangleImplementsBaseInterface) {
    const auto geometry = Triangle<float>::builder()
                              .with_vertices(
                                  Vector3F(0, 0, 0),
                                  Vector3F(1, 0, 0),
                                  Vector3F(0, 1, 0))
                              .make_host_shared();

    expect_geometry_interface<float>(geometry, GeometryType::Triangle, Vector3F(0.25f, 0.25f, 1.0f));
}

TEST(Geometry, TriangleMeshImplementsBaseInterface) {
    TriangleContainer4<float> triangle {};
    triangle[0] = Vector3F(0, 0, 0);
    triangle[1] = Vector3F(1, 0, 0);
    triangle[2] = Vector3F(0, 1, 0);
    triangle[3] = Vector3F(0, 0, 0);

    const auto geometry = TriangleMesh<float>::builder()
                              .with_triangles(HostBuffer<TriangleContainer4<float>> { triangle })
                              .make_host_shared();

    expect_geometry_interface<float>(
        geometry,
        GeometryType::TriangleMesh,
        Vector3F(0.2f, 0.2f, 0.5f));
}
