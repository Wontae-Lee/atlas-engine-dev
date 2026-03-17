#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SphereQueryOperator, ClosestPointReturnsInputWhenPointersNull) {
    constexpr geometry::SphereQueryOperator<double> op;

    const Vector3<double> p(1.25, -2.5, 3.75);
    EXPECT_TRUE(test::vec_near(op.closest_point(p), p, eps));
}

TEST(SphereQueryOperator, ClosestPointAtCenterReturnsPlusXOnSurface) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 4.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    const Vector3<double> p = c;
    const Vector3<double> expected(c.x + r, c.y, c.z);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(SphereQueryOperator, ClosestPointProjectsToSurfaceAlongRadialDirection) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    const Vector3<double> p(3.0, 4.0, 0.0);
    const Vector3<double> expected(1.2, 1.6, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(SphereQueryOperator, ClosestNormalReturnsZeroWhenPointersNull) {
    constexpr geometry::SphereQueryOperator<double> op;

    const Vector3<double> p(1.0, 2.0, 3.0);
    EXPECT_TRUE(test::vec_near(op.closest_normal(p), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(SphereQueryOperator, ClosestNormalAtCenterReturnsPlusX) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 4.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    const Vector3<double> p = c;
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(SphereQueryOperator, ClosestNormalIsNormalizedRadialVector) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 10.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    const Vector3<double> p(3.0, 4.0, 0.0);
    const Vector3<double> expected(0.6, 0.8, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(SphereQueryOperator, SignedDistanceReturnsInfWhenPointersNull) {
    constexpr geometry::SphereQueryOperator<double> op;

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(SphereQueryOperator, SignedDistanceMatchesDefinition) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    EXPECT_NEAR(op.signed_distance(Vector3<double>(0.0, 0.0, 0.0)), -2.0, eps);
    EXPECT_NEAR(op.signed_distance(Vector3<double>(2.0, 0.0, 0.0)), 0.0, eps);
    EXPECT_NEAR(op.signed_distance(Vector3<double>(5.0, 0.0, 0.0)), 3.0, eps);
}

TEST(SphereQueryOperator, IsInsideClassifiesInteriorAndToleranceBand) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_inside(Vector3<double>(2.2, 0.0, 0.0)));
    EXPECT_TRUE(op.is_inside(Vector3<double>(2.2, 0.0, 0.0), 0.25));
}

TEST(SphereQueryOperator, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(2.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(2.1, 0.0, 0.0), 0.15));
}

TEST(SphereQueryOperator, CentroidReturnsZeroWhenCenterNull) {
    constexpr geometry::SphereQueryOperator<double> op;
    EXPECT_TRUE(test::vec_near(op.centroid(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(SphereQueryOperator, CentroidReturnsCenterWhenValid) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 4.0;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    EXPECT_TRUE(test::vec_near(op.centroid(), c, eps));
}

TEST(SphereQueryOperator, BoundReturnsDefaultWhenPointersNull) {
    constexpr geometry::SphereQueryOperator<double> op;

    const auto aabb = op.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(SphereQueryOperator, BoundIsCenterPlusMinusRadiusVector) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 2.5;

    geometry::SphereQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);

    const auto aabb = op.bound();

    const Vector3<double> dr(r, r, r);
    const Vector3<double> lo = c - dr;
    const Vector3<double> hi = c + dr;

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(SphereQueryOperator, IsValidFalseWhenRadiusNull) {
    constexpr geometry::SphereQueryOperator<double> op;
    EXPECT_FALSE(op.is_valid());
}

TEST(SphereQueryOperator, IsValidTrueForPositiveRadius) {
    constexpr double r = 1.0;

    geometry::SphereQueryOperator<double> op;
    op.radius = atlas::raw_pointer_cast(&r);

    EXPECT_TRUE(op.is_valid());
}

TEST(SphereQueryOperator, IsValidFalseForNonPositiveRadius) {
    constexpr double r0 = 0.0;
    constexpr double rn = -1.0;

    geometry::SphereQueryOperator<double> op0;
    op0.radius = atlas::raw_pointer_cast(&r0);

    geometry::SphereQueryOperator<double> opn;
    opn.radius = atlas::raw_pointer_cast(&rn);

    EXPECT_FALSE(op0.is_valid());
    EXPECT_FALSE(opn.is_valid());
}
