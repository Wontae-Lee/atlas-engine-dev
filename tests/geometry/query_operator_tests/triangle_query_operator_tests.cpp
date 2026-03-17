#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(TriangleQueryOperator, ClosestPointReturnsInputWhenVerticesNull) {
    constexpr geometry::TriangleQueryOperator<double> op;

    constexpr Vector3<double> p(1.25, -2.5, 3.75);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, p, eps));
}

TEST(TriangleQueryOperator, ClosestPointVertexRegionA) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(-1.0, -1.0, 0.0);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, a, eps));
}

TEST(TriangleQueryOperator, ClosestPointVertexRegionB) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(2.0, -1.0, 0.0);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, b, eps));
}

TEST(TriangleQueryOperator, ClosestPointVertexRegionC) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(-1.0, 2.0, 0.0);
    const auto cp = op.closest_point(p);

    EXPECT_TRUE(test::vec_near(cp, c, eps));
}

TEST(TriangleQueryOperator, ClosestPointEdgeRegionABProjectsToSegment) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(2.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 2.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(1.0, -1.0, 0.0);
    constexpr Vector3<double> expected(1.0, 0.0, 0.0);

    const auto cp = op.closest_point(p);
    EXPECT_TRUE(test::vec_near(cp, expected, eps));
}

TEST(TriangleQueryOperator, ClosestPointEdgeRegionACProjectsToSegment) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(2.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 2.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(-1.0, 1.0, 0.0);
    constexpr Vector3<double> expected(0.0, 1.0, 0.0);

    const auto cp = op.closest_point(p);
    EXPECT_TRUE(test::vec_near(cp, expected, eps));
}

TEST(TriangleQueryOperator, ClosestPointEdgeRegionBCProjectsToSegment) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(2.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 2.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(2.0, 2.0, 0.0);
    constexpr Vector3<double> expected(1.0, 1.0, 0.0);

    const auto cp = op.closest_point(p);
    EXPECT_TRUE(test::vec_near(cp, expected, eps));
}

TEST(TriangleQueryOperator, ClosestPointFaceRegionUsesBarycentric) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(2.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 2.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(0.25, 0.25, 1.0);
    constexpr Vector3<double> expected(0.25, 0.25, 0.0);

    const auto cp = op.closest_point(p);
    EXPECT_TRUE(test::vec_near(cp, expected, eps));
}

TEST(TriangleQueryOperator, ClosestNormalReturnsExplicitNormalWhenProvided) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    constexpr Vector3<double> n(0.0, 0.0, -1.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);
    op.n = atlas::raw_pointer_cast(&n);

    constexpr Vector3<double> p(0.2, 0.2, 10.0);
    const auto nn = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(nn, n, eps));
}

TEST(TriangleQueryOperator, ClosestNormalComputesGeometricNormalWhenNoExplicitNormal) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p(0.2, 0.2, 10.0);
    const auto nn = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(nn, Vector3<double>(0.0, 0.0, 1.0), eps));
    EXPECT_NEAR(nn.length(), 1.0, eps);
}

TEST(TriangleQueryOperator, ClosestNormalFallbackWhenVerticesNull) {
    constexpr geometry::TriangleQueryOperator<double> op;

    constexpr Vector3<double> p(0.0, 0.0, 0.0);
    const auto nn = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(nn, Vector3<double>(0.0, 0.0, 1.0), eps));
}

TEST(TriangleQueryOperator, SignedDistanceReturnsInfWhenVerticesNull) {
    constexpr geometry::TriangleQueryOperator<double> op;

    constexpr Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(TriangleQueryOperator, SignedDistanceSignMatchesNormalSide) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> p_pos(0.25, 0.25, 2.0);
    constexpr Vector3<double> p_neg(0.25, 0.25, -2.0);

    const double d_pos = op.signed_distance(p_pos);
    const double d_neg = op.signed_distance(p_neg);

    EXPECT_GT(d_pos, 0.0);
    EXPECT_LT(d_neg, 0.0);
    EXPECT_NEAR(std::abs(d_pos), 2.0, eps);
    EXPECT_NEAR(std::abs(d_neg), 2.0, eps);
}

TEST(TriangleQueryOperator, IsInsideUsesSignedSideAndTolerance) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.25, 0.25, -0.1), 0.0));
    EXPECT_FALSE(op.is_inside(Vector3<double>(0.25, 0.25, 0.1), 0.0));
    EXPECT_TRUE(op.is_inside(Vector3<double>(0.25, 0.25, 0.1), 0.15));
}

TEST(TriangleQueryOperator, IsOnSurfaceDetectsSurfaceBand) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.0), 0.0));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.3), 0.0));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.25, 0.25, 0.1), 0.15));
}

TEST(TriangleQueryOperator, CentroidReturnsZeroWhenVerticesNull) {
    constexpr geometry::TriangleQueryOperator<double> op;

    const auto ctd = op.centroid();
    EXPECT_TRUE(test::vec_near(ctd, Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(TriangleQueryOperator, CentroidIsAverageOfVertices) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(3.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 6.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    constexpr Vector3<double> expected(1.0, 2.0, 0.0);

    const auto ctd = op.centroid();
    EXPECT_TRUE(test::vec_near(ctd, expected, eps));
}

TEST(TriangleQueryOperator, BoundReturnsDefaultWhenVerticesNull) {
    constexpr geometry::TriangleQueryOperator<double> op;

    const auto aabb = op.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(TriangleQueryOperator, BoundIsComponentwiseMinMaxOfVertices) {
    constexpr Vector3<double> a(-2.0, 3.0, 1.0);
    constexpr Vector3<double> b(5.0, -4.0, 2.0);
    constexpr Vector3<double> c(1.0, 2.0, -6.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    const auto aabb = op.bound();

    constexpr Vector3<double> expected_lo(-2.0, -4.0, -6.0);
    constexpr Vector3<double> expected_hi(5.0, 3.0, 2.0);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, expected_lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, expected_hi, eps));
}

TEST(TriangleQueryOperator, IsValidFalseWhenAnyVertexNull) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);

    geometry::TriangleQueryOperator<double> op1;
    op1.a = atlas::raw_pointer_cast(&a);
    op1.b = atlas::raw_pointer_cast(&b);
    EXPECT_FALSE(op1.is_valid());

    geometry::TriangleQueryOperator<double> op2;
    op2.a = atlas::raw_pointer_cast(&a);
    op2.c = atlas::raw_pointer_cast(&b);
    EXPECT_FALSE(op2.is_valid());

    geometry::TriangleQueryOperator<double> op3;
    op3.b = atlas::raw_pointer_cast(&a);
    op3.c = atlas::raw_pointer_cast(&b);
    EXPECT_FALSE(op3.is_valid());
}

TEST(TriangleQueryOperator, IsValidFalseForDegenerateTriangle) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(2.0, 0.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    EXPECT_FALSE(op.is_valid());
}

TEST(TriangleQueryOperator, IsValidTrueForNonDegenerateTriangle) {
    constexpr Vector3<double> a(0.0, 0.0, 0.0);
    constexpr Vector3<double> b(1.0, 0.0, 0.0);
    constexpr Vector3<double> c(0.0, 1.0, 0.0);

    geometry::TriangleQueryOperator<double> op;
    op.a = atlas::raw_pointer_cast(&a);
    op.b = atlas::raw_pointer_cast(&b);
    op.c = atlas::raw_pointer_cast(&c);

    EXPECT_TRUE(op.is_valid());
}
