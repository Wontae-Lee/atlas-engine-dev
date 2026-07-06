#include <atlas/geometry/triangle_mesh.h>

#include <atlas/geometry/geometry.h>

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::GeometryType;
using atlas::HostBuffer;
using atlas::Ray;
using atlas::TriangleContainer4;
using atlas::TriangleMesh;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

bool
is_finite_vec(const Vector3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

HostBuffer<TriangleContainer4>
make_triangles() {
    TriangleContainer4 triangle {};
    triangle[0] = Vector3(0.0f, 0.0f, 0.0f);
    triangle[1] = Vector3(1.0f, 0.0f, 0.0f);
    triangle[2] = Vector3(0.0f, 1.0f, 0.0f);
    triangle[3] = Vector3(0.0f, 0.0f, 0.0f);
    return { triangle };
}

}

TEST(TriangleMesh, BuilderConstructsValidMesh) {
    const auto mesh = TriangleMesh::builder()
                          .with_triangles(make_triangles())
                          .build();

    EXPECT_TRUE(mesh.is_valid());
}

TEST(TriangleMesh, BuilderRejectsEmptyMesh) {
    EXPECT_THROW(
        (void)TriangleMesh::builder().build(),
        std::runtime_error);
}

TEST(TriangleMesh, ClosestPointNormalAndDistanceWork) {
    const auto mesh = TriangleMesh::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vector3 closest = mesh.closest_point(Vector3(0.25f, 0.25f, 1.0f));
    const Vector3 normal  = mesh.closest_normal(Vector3(0.25f, 0.25f, 1.0f));

    expect_vec_near(closest, Vector3(0.25f, 0.25f, 0.0f));
    EXPECT_TRUE(is_finite_vec(normal));
    EXPECT_NEAR(std::abs(mesh.signed_distance(Vector3(0.25f, 0.25f, 1.0f))), 1.0f, tol);
}

TEST(TriangleMesh, ClassificationCentroidBoundAndOperatorWork) {
    const auto mesh = TriangleMesh::builder()
                          .with_triangles(make_triangles())
                          .build();

    const Vector3 center         = mesh.centroid();
    const auto bounds            = mesh.bound();
    const auto geometry_operator = mesh.make_device_geometry_view();

    EXPECT_TRUE(is_finite_vec(center));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(geometry_operator.type, GeometryType::triangle_mesh);
}

TEST(TriangleMesh, TraceHitsMesh) {
    const auto mesh = TriangleMesh::builder()
                          .with_triangles(make_triangles())
                          .build();

    const auto hit = mesh.make_device_geometry_view().trace(Ray(Vector3(0.25f, 0.25f, 1.0f), Vector3(0.0f, 0.0f, -1.0f)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_TRUE(is_finite_vec(hit.point));
    EXPECT_TRUE(is_finite_vec(hit.normal));
}
