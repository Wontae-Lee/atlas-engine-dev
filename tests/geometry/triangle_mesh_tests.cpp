#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/triangle_mesh.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

atlas::HostBuffer<atlas::TriangleContainer4<T>>
make_triangles() {
    atlas::TriangleContainer4<T> triangle {};
    triangle[0] = Vec3(0, 0, 0);
    triangle[1] = Vec3(1, 0, 0);
    triangle[2] = Vec3(0, 1, 0);
    triangle[3] = Vec3(0, 0, 0);
    return { triangle };
}

} // namespace

TEST(TriangleMesh, BuilderConstructsValidMesh) {
    const auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                          .with_triangles(make_triangles())
                          .build();

    EXPECT_EQ(mesh.type(), atlas::geometry::GeometryType::TriangleMesh);
    EXPECT_TRUE(mesh.is_valid());
}

TEST(TriangleMesh, BuilderRejectsEmptyMesh) {
    EXPECT_THROW(
        atlas::geometry::TriangleMesh<T>::builder().build(),
        std::runtime_error);
}

TEST(TriangleMesh, ClosestPointNormalAndDistanceWork) {
    const auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vec3 closest = mesh.closest_point(Vec3(0.25f, 0.25f, 1.0f));
    const Vec3 normal = mesh.closest_normal(Vec3(0.25f, 0.25f, 1.0f));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(0.25f, 0.25f, 0.0f), 1e-4f));
    EXPECT_TRUE(atlas::test::is_finite_vec(normal));
    EXPECT_NEAR(std::abs(mesh.signed_distance(Vec3(0.25f, 0.25f, 1.0f))), 1.0f, 1e-3f);
}

TEST(TriangleMesh, ClassificationCentroidBoundAndOperatorWork) {
    const auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vec3 center = mesh.centroid();
    const auto bounds = mesh.bound();
    const auto geometry_operator = mesh.make_geometry_operator();

    EXPECT_TRUE(atlas::test::is_finite_vec(center));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(mesh.type(), geometry_operator.type);
}

TEST(TriangleMesh, TraceHitsMesh) {
    const auto mesh = atlas::geometry::TriangleMesh<T>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const auto hit = mesh.make_geometry_operator().trace(atlas::spatial::Ray<T>(Vec3(0.25f, 0.25f, 1.0f), Vec3(0, 0, -1)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_TRUE(atlas::test::is_finite_vec(hit.point));
    EXPECT_TRUE(atlas::test::is_finite_vec(hit.normal));
}
