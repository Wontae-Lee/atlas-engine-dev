#include "../utilities/test_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::HostBuffer;
using atlas::Ray;
using atlas::TriangleContainer4;
using atlas::TriangleMesh;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;
using atlas::tol;

HostBuffer<TriangleContainer4<float>>
make_triangles() {
    TriangleContainer4<float> triangle {};
    triangle[0] = Vector3F(0, 0, 0);
    triangle[1] = Vector3F(1, 0, 0);
    triangle[2] = Vector3F(0, 1, 0);
    triangle[3] = Vector3F(0, 0, 0);
    return { triangle };
}

} // namespace

TEST(TriangleMesh, BuilderConstructsValidMesh) {
    const auto mesh = TriangleMesh<float>::builder()
                          .with_triangles(make_triangles())
                          .build();

    EXPECT_EQ(mesh.type(), GeometryType::TriangleMesh);
    EXPECT_TRUE(mesh.is_valid());
}

TEST(TriangleMesh, BuilderRejectsEmptyMesh) {
    EXPECT_THROW(
        TriangleMesh<float>::builder().build(),
        std::runtime_error);
}

TEST(TriangleMesh, ClosestPointNormalAndDistanceWork) {
    const auto mesh = TriangleMesh<float>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vector3F closest = mesh.closest_point(Vector3F(0.25f, 0.25f, 1.0f));
    const Vector3F normal = mesh.closest_normal(Vector3F(0.25f, 0.25f, 1.0f));

    EXPECT_TRUE(vec_near(closest, Vector3F(0.25f, 0.25f, 0.0f), tol));
    EXPECT_TRUE(is_finite_vec(normal));
    EXPECT_NEAR(std::abs(mesh.signed_distance(Vector3F(0.25f, 0.25f, 1.0f))), 1.0f, tol);
}

TEST(TriangleMesh, ClassificationCentroidBoundAndOperatorWork) {
    const auto mesh = TriangleMesh<float>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vector3F center = mesh.centroid();
    const auto bounds = mesh.bound();
    const auto geometry_operator = mesh.make_device_geometry_view();

    EXPECT_TRUE(is_finite_vec(center));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(mesh.type(), geometry_operator.type);
}

TEST(TriangleMesh, TraceHitsMesh) {
    const auto mesh = TriangleMesh<float>::builder()
                          .with_triangles(make_triangles())
                          .build();

    const auto hit = mesh.make_device_geometry_view().trace(Ray<float>(Vector3F(0.25f, 0.25f, 1.0f), Vector3F(0, 0, -1)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_TRUE(is_finite_vec(hit.point));
    EXPECT_TRUE(is_finite_vec(hit.normal));
}
