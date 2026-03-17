#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(CylinderQueryOperator, SignedDistanceReturnsInfWhenPointersNull) {
    constexpr geometry::CylinderQueryOperator<double> op;

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(CylinderQueryOperator, SignedDistanceIsNegativeInside) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_LT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), -2.0, eps);
}

TEST(CylinderQueryOperator, SignedDistanceIsPositiveOutsideRadially) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(5.0, 0.0, 0.0);
    EXPECT_GT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), 3.0, eps);
}

TEST(CylinderQueryOperator, SignedDistanceIsPositiveOutsideAboveCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 5.0);
    EXPECT_GT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), 3.0, eps);
}

TEST(CylinderQueryOperator, IsInsideClassifiesInteriorAndToleranceBand) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0)));
    EXPECT_TRUE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(CylinderQueryOperator, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.0, 0.0, 1.1), 0.15));
}

TEST(CylinderQueryOperator, ClosestPointReturnsInputWhenPointersNull) {
    constexpr geometry::CylinderQueryOperator<double> op;

    const Vector3<double> p(1.25, -2.5, 3.75);
    EXPECT_TRUE(test::vec_near(op.closest_point(p), p, eps));
}

TEST(CylinderQueryOperator, ClosestPointOutsideProjectsToSideAndClampsZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(10.0, 0.0, 10.0);
    const Vector3<double> expected(2.0, 0.0, 2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestPointInsidePushesToNearestSideWall) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.9, 0.0, 0.0);
    const Vector3<double> expected(2.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestPointInsideOnAxisChoosesPlusXSideWall) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(c.x, c.y, c.z);
    const Vector3<double> expected(c.x + r, c.y, c.z);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestPointInsidePushesToNearestBottomCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 1.0, -1.9);
    const Vector3<double> expected(1.0, 1.0, -2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestPointInsidePushesToNearestTopCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 1.0, 1.9);
    const Vector3<double> expected(1.0, 1.0, 2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalReturnsZeroWhenPointersNull) {
    constexpr geometry::CylinderQueryOperator<double> op;

    const Vector3<double> p(1.0, 2.0, 3.0);
    EXPECT_TRUE(test::vec_near(op.closest_normal(p), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(CylinderQueryOperator, ClosestNormalInsideSideWallIsRadialUnit) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 0.0, 0.0);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalInsideOnAxisReturnsPlusX) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 0.0);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalInsideBottomCapIsMinusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, -1.9);
    const Vector3<double> expected(0.0, 0.0, -1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalInsideTopCapIsPlusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 1.9);
    const Vector3<double> expected(0.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalOutsideAboveCapIsPlusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.5, 0.5, 10.0);
    const Vector3<double> expected(0.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, ClosestNormalOutsideSideWallIsRadialFromCenterToClosestPoint) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 6.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(c.x + 10.0, c.y, c.z);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderQueryOperator, CentroidReturnsZeroWhenCenterNull) {
    constexpr geometry::CylinderQueryOperator<double> op;
    EXPECT_TRUE(test::vec_near(op.centroid(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(CylinderQueryOperator, CentroidReturnsCenterWhenValid) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(test::vec_near(op.centroid(), c, eps));
}

TEST(CylinderQueryOperator, BoundReturnsDefaultWhenPointersNull) {
    constexpr geometry::CylinderQueryOperator<double> op;

    const auto aabb = op.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(CylinderQueryOperator, BoundMatchesAxisAlignedExtents) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 6.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const auto aabb = op.bound();

    const Vector3<double> lo(c.x - r, c.y - r, c.z - h * 0.5);
    const Vector3<double> hi(c.x + r, c.y + r, c.z + h * 0.5);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(CylinderQueryOperator, IsValidFalseWhenRadiusOrHeightNull) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderQueryOperator<double> op_rnull;
    op_rnull.center = atlas::raw_pointer_cast(&c);
    op_rnull.height = atlas::raw_pointer_cast(&h);
    EXPECT_FALSE(op_rnull.is_valid());

    geometry::CylinderQueryOperator<double> op_hnull;
    op_hnull.center = atlas::raw_pointer_cast(&c);
    op_hnull.radius = atlas::raw_pointer_cast(&r);
    EXPECT_FALSE(op_hnull.is_valid());
}

TEST(CylinderQueryOperator, IsValidTrueForPositiveRadiusAndHeight) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderQueryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_valid());
}

TEST(CylinderQueryOperator, IsValidFalseForNonPositiveRadiusOrHeight) {
    const Vector3<double> c(0.0, 0.0, 0.0);

    constexpr double r0 = 0.0;
    constexpr double rn = -1.0;
    constexpr double h0 = 0.0;
    constexpr double hn = -2.0;

    constexpr double r_ok = 1.0;
    constexpr double h_ok = 2.0;

    geometry::CylinderQueryOperator<double> op_r0;
    op_r0.center = atlas::raw_pointer_cast(&c);
    op_r0.radius = atlas::raw_pointer_cast(&r0);
    op_r0.height = atlas::raw_pointer_cast(&h_ok);
    EXPECT_FALSE(op_r0.is_valid());

    geometry::CylinderQueryOperator<double> op_rn;
    op_rn.center = atlas::raw_pointer_cast(&c);
    op_rn.radius = atlas::raw_pointer_cast(&rn);
    op_rn.height = atlas::raw_pointer_cast(&h_ok);
    EXPECT_FALSE(op_rn.is_valid());

    geometry::CylinderQueryOperator<double> op_h0;
    op_h0.center = atlas::raw_pointer_cast(&c);
    op_h0.radius = atlas::raw_pointer_cast(&r_ok);
    op_h0.height = atlas::raw_pointer_cast(&h0);
    EXPECT_FALSE(op_h0.is_valid());

    geometry::CylinderQueryOperator<double> op_hn;
    op_hn.center = atlas::raw_pointer_cast(&c);
    op_hn.radius = atlas::raw_pointer_cast(&r_ok);
    op_hn.height = atlas::raw_pointer_cast(&hn);
    EXPECT_FALSE(op_hn.is_valid());
}
