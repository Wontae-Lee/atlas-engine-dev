#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(TriangleMeshGeometryOperator, IsValidFalseWhenVerticesNull) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<int> idx = { 0, 1, 2 };
    op.indices                 = idx.data();
    op.triangle_count          = 1;

    EXPECT_FALSE(op.is_valid());
}

TEST(TriangleMeshGeometryOperator, IsValidFalseWhenIndicesNull) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
    };
    op.vertices       = v.data();
    op.triangle_count = 1;

    EXPECT_FALSE(op.is_valid());
}

TEST(TriangleMeshGeometryOperator, IsValidFalseWhenTriangleCountNonPositive) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
    };
    const std::vector<int> idx = { 0, 1, 2 };

    op.vertices = v.data();
    op.indices  = idx.data();

    op.triangle_count = 0;
    EXPECT_FALSE(op.is_valid());

    op.triangle_count = -3;
    EXPECT_FALSE(op.is_valid());
}

TEST(TriangleMeshGeometryOperator, IsValidTrueForNonEmptyBuffersAndPositiveCount) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
    };
    const std::vector<int> idx = { 0, 1, 2 };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 1;

    EXPECT_TRUE(op.is_valid());
}

TEST(TriangleMeshGeometryOperator, ClosestPointReturnsInputWhenInvalid) {
    constexpr geometry::TriangleMeshGeometryOperator<double> op;

    constexpr Vector3<double> p(1.0, 2.0, 3.0);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, p, eps));
}

TEST(TriangleMeshGeometryOperator, ClosestNormalReturnsUpWhenInvalid) {
    constexpr geometry::TriangleMeshGeometryOperator<double> op;

    constexpr Vector3<double> p(1.0, 2.0, 3.0);
    const auto n = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(0.0, 0.0, 1.0), eps));
}

TEST(TriangleMeshGeometryOperator, SignedDistanceReturnsInfWhenInvalid) {
    constexpr geometry::TriangleMeshGeometryOperator<double> op;

    constexpr Vector3<double> p(1.0, 2.0, 3.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(TriangleMeshGeometryOperator, CentroidReturnsZeroWhenInvalid) {
    constexpr geometry::TriangleMeshGeometryOperator<double> op;

    const auto c = op.centroid();
    EXPECT_TRUE(test::vec_near(c, Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(TriangleMeshGeometryOperator, BoundReturnsDefaultWhenInvalid) {
    constexpr geometry::TriangleMeshGeometryOperator<double> op;

    const auto aabb = op.bound();
    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(TriangleMeshGeometryOperator, ClosestPointSelectsNearestTriangleAmongTwo) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),

        Vector3<double>(10.0, 0.0, 0.0),
        Vector3<double>(11.0, 0.0, 0.0),
        Vector3<double>(10.0, 1.0, 0.0),
    };

    const std::vector<int> idx = {
        0,
        1,
        2,
        3,
        4,
        5,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 2;

    constexpr Vector3<double> p(0.2, 0.2, 5.0);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, Vector3<double>(0.2, 0.2, 0.0), eps));
}

TEST(TriangleMeshGeometryOperator, ClosestNormalMatchesTriangleNormalOfNearestTriangle) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),

        Vector3<double>(10.0, 0.0, 0.0),
        Vector3<double>(10.0, 1.0, 0.0),
        Vector3<double>(11.0, 0.0, 0.0),
    };

    const std::vector<int> idx = {
        0,
        1,
        2,
        3,
        4,
        5,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 2;

    constexpr Vector3<double> p(0.25, 0.25, 1.0);
    const auto n = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(0.0, 0.0, 1.0), eps));
    EXPECT_NEAR(n.length(), 1.0, 1e-12);
}

TEST(TriangleMeshGeometryOperator, SignedDistanceUsesWindingForClosedMeshSign) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
        Vector3<double>(0.0, 0.0, 1.0),
    };
    const std::vector<int> idx = {
        0,
        2,
        1,
        0,
        1,
        3,
        0,
        3,
        2,
        1,
        2,
        3,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 4;

    constexpr Vector3<double> p_in(0.1, 0.1, 0.1);
    constexpr Vector3<double> p_out(2.0, 2.0, 2.0);

    const double d_in  = op.signed_distance(p_in);
    const double d_out = op.signed_distance(p_out);

    EXPECT_LT(d_in, 0.0);
    EXPECT_GT(d_out, 0.0);
}

TEST(TriangleMeshGeometryOperator, IsInsideUsesWindingForClosedMeshContainment) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
        Vector3<double>(0.0, 0.0, 1.0),
    };
    const std::vector<int> idx = {
        0,
        2,
        1,
        0,
        1,
        3,
        0,
        3,
        2,
        1,
        2,
        3,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 4;

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.1, 0.1, 0.1), 0.0));
    EXPECT_FALSE(op.is_inside(Vector3<double>(2.0, 2.0, 2.0), 0.0));
    EXPECT_TRUE(op.is_inside(Vector3<double>(0.6, 0.6, 0.1), 0.5));
}

TEST(TriangleMeshGeometryOperator, IsOnSurfaceDetectsSurfaceBand) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0),
    };
    const std::vector<int> idx = { 0, 1, 2 };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 1;

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.0), 0.0));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.3), 0.0));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.1), 0.15));
}

TEST(TriangleMeshGeometryOperator, CentroidIsAverageOfTriangleCentroidsUniformWeight) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 3.0, 0.0),

        Vector3<double>(10.0, 0.0, 0.0),
        Vector3<double>(13.0, 0.0, 0.0),
        Vector3<double>(10.0, 3.0, 0.0),
    };

    const std::vector<int> idx = {
        0,
        1,
        2,
        3,
        4,
        5,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 2;

    constexpr Vector3<double> c0((0.0 + 3.0 + 0.0) / 3.0, (0.0 + 0.0 + 3.0) / 3.0, 0.0);
    constexpr Vector3<double> c1((10.0 + 13.0 + 10.0) / 3.0, (0.0 + 0.0 + 3.0) / 3.0, 0.0);
    constexpr Vector3<double> expected((c0.x + c1.x) * 0.5, (c0.y + c1.y) * 0.5, 0.0);

    const auto c = op.centroid();
    EXPECT_TRUE(test::vec_near(c, expected, eps));
}

TEST(TriangleMeshGeometryOperator, BoundScansAllReferencedVertices) {
    geometry::TriangleMeshGeometryOperator<double> op;

    const std::vector<Vector3<double>> v = {
        Vector3<double>(-2.0, 3.0, 1.0),
        Vector3<double>(5.0, -4.0, 2.0),
        Vector3<double>(1.0, 2.0, -6.0),
        Vector3<double>(100.0, 100.0, 100.0),
    };

    const std::vector<int> idx = {
        0,
        1,
        2,
    };

    op.vertices       = v.data();
    op.indices        = idx.data();
    op.triangle_count = 1;

    const auto aabb = op.bound();

    constexpr Vector3<double> expected_lo(-2.0, -4.0, -6.0);
    constexpr Vector3<double> expected_hi(5.0, 3.0, 2.0);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, expected_lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, expected_hi, eps));
}