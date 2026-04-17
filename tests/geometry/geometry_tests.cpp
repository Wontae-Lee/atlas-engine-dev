#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

#include <gtest/gtest.h>

namespace {

template <typename T>
void
expect_geometry_interface(const atlas::GeometryHostPtr<T>& geometry,
                          const atlas::geometry::GeometryType expected_type,
                          const atlas::Vector3<T>& query_point) {
    ASSERT_NE(geometry, nullptr);

    const atlas::Geometry<T>& base = *geometry;

    EXPECT_EQ(base.type(), expected_type);
    EXPECT_TRUE(base.is_valid());

    const auto operator_view = base.make_geometry_operator();
    (void)operator_view;

    const auto closest_point = base.closest_point(query_point);
    const auto closest_normal = base.closest_normal(query_point);
    const auto center = base.centroid();
    const auto bounds = base.bound();
    const auto distance = base.signed_distance(query_point);
    const auto is_inside = base.is_inside(query_point, static_cast<T>(atlas::eps));
    const auto is_on_surface = base.is_on_surface(query_point, static_cast<T>(atlas::eps));

    EXPECT_TRUE(atlas::test::is_finite_vec(closest_point));
    EXPECT_TRUE(atlas::test::is_finite_vec(closest_normal));
    EXPECT_TRUE(atlas::test::is_finite_vec(center));
    EXPECT_TRUE(atlas::test::is_finite_vec(bounds.lower_corner));
    EXPECT_TRUE(atlas::test::is_finite_vec(bounds.upper_corner));
    EXPECT_TRUE(std::isfinite(static_cast<double>(distance)));
    EXPECT_TRUE(is_inside || !is_inside);
    EXPECT_TRUE(is_on_surface || !is_on_surface);
}

} // namespace

TEST(Geometry, BoxImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(atlas::Vector3<T>(-1, -2, -3))
                              .with_upper_corner(atlas::Vector3<T>(1, 2, 3))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Box, atlas::Vector3<T>(3, 0, 0));
}

TEST(Geometry, CircleImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Circle<T>::builder()
                              .with_center(atlas::Vector3<T>(0, 0, 0))
                              .with_normal(atlas::Vector3<T>(0, 1, 0))
                              .with_radius(static_cast<T>(2))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Circle, atlas::Vector3<T>(1, 3, 0));
}

TEST(Geometry, CylinderImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Cylinder<T>::builder()
                              .with_center(atlas::Vector3<T>(0, 0, 0))
                              .with_radius(static_cast<T>(1.5))
                              .with_height(static_cast<T>(4))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Cylinder, atlas::Vector3<T>(2, 0, 0));
}

TEST(Geometry, PlaneImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Plane<T>::builder()
                              .with_point_normal(atlas::Vector3<T>(0, 0, 0), atlas::Vector3<T>(0, 1, 0))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Plane, atlas::Vector3<T>(0, 2, 0));
}

TEST(Geometry, SphereImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Sphere<T>::builder()
                              .with_center(atlas::Vector3<T>(0, 0, 0))
                              .with_radius(static_cast<T>(2))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Sphere, atlas::Vector3<T>(4, 0, 0));
}

TEST(Geometry, TriangleImplementsBaseInterface) {
    using T = float;

    const auto geometry = atlas::geometry::Triangle<T>::builder()
                              .with_vertices(
                                  atlas::Vector3<T>(0, 0, 0),
                                  atlas::Vector3<T>(1, 0, 0),
                                  atlas::Vector3<T>(0, 1, 0))
                              .make_host_shared();

    expect_geometry_interface<T>(geometry, atlas::geometry::GeometryType::Triangle, atlas::Vector3<T>(0.25f, 0.25f, 1.0f));
}

TEST(Geometry, TriangleMeshImplementsBaseInterface) {
    using T = float;

    atlas::TriangleContainer4<T> triangle {};
    triangle[0] = atlas::Vector3<T>(0, 0, 0);
    triangle[1] = atlas::Vector3<T>(1, 0, 0);
    triangle[2] = atlas::Vector3<T>(0, 1, 0);
    triangle[3] = atlas::Vector3<T>(0, 0, 0);

    const auto geometry = atlas::geometry::TriangleMesh<T>::builder()
                              .with_triangles(atlas::HostBuffer<atlas::TriangleContainer4<T>> { triangle })
                              .make_host_shared();

    expect_geometry_interface<T>(
        geometry,
        atlas::geometry::GeometryType::TriangleMesh,
        atlas::Vector3<T>(0.2f, 0.2f, 0.5f));
}
