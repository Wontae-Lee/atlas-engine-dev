#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(CylinderGeometryOperator, SignedDistanceReturnsInfWhenPointersNull) {
    constexpr geometry::CylinderGeometryOperator<double> op;

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(CylinderGeometryOperator, SignedDistanceIsNegativeInside) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_LT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), -2.0, eps);
}

TEST(CylinderGeometryOperator, SignedDistanceIsPositiveOutsideRadially) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(5.0, 0.0, 0.0);
    EXPECT_GT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), 3.0, eps);
}

TEST(CylinderGeometryOperator, SignedDistanceIsPositiveOutsideAboveCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 5.0);
    EXPECT_GT(op.signed_distance(p), 0.0);
    EXPECT_NEAR(op.signed_distance(p), 3.0, eps);
}

TEST(CylinderGeometryOperator, IsInsideClassifiesInteriorAndToleranceBand) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0)));
    EXPECT_TRUE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(CylinderGeometryOperator, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(0.0, 0.0, 1.1), 0.15));
}

TEST(CylinderGeometryOperator, ClosestPointReturnsInputWhenPointersNull) {
    constexpr geometry::CylinderGeometryOperator<double> op;

    const Vector3<double> p(1.25, -2.5, 3.75);
    EXPECT_TRUE(test::vec_near(op.closest_point(p), p, eps));
}

TEST(CylinderGeometryOperator, ClosestPointOutsideProjectsToSideAndClampsZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(10.0, 0.0, 10.0);
    const Vector3<double> expected(2.0, 0.0, 2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestPointInsidePushesToNearestSideWall) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.9, 0.0, 0.0);
    const Vector3<double> expected(2.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestPointInsideOnAxisChoosesPlusXSideWall) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(c.x, c.y, c.z);
    const Vector3<double> expected(c.x + r, c.y, c.z);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestPointInsidePushesToNearestBottomCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 1.0, -1.9);
    const Vector3<double> expected(1.0, 1.0, -2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestPointInsidePushesToNearestTopCap) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 1.0, 1.9);
    const Vector3<double> expected(1.0, 1.0, 2.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalReturnsZeroWhenPointersNull) {
    constexpr geometry::CylinderGeometryOperator<double> op;

    const Vector3<double> p(1.0, 2.0, 3.0);
    EXPECT_TRUE(test::vec_near(op.closest_normal(p), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(CylinderGeometryOperator, ClosestNormalInsideSideWallIsRadialUnit) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(1.0, 0.0, 0.0);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalInsideOnAxisReturnsPlusX) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 10.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 0.0);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalInsideBottomCapIsMinusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, -1.9);
    const Vector3<double> expected(0.0, 0.0, -1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalInsideTopCapIsPlusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 5.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.0, 0.0, 1.9);
    const Vector3<double> expected(0.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalOutsideAboveCapIsPlusZ) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(0.5, 0.5, 10.0);
    const Vector3<double> expected(0.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, ClosestNormalOutsideSideWallIsRadialFromCenterToClosestPoint) {
    const Vector3<double> c(1.0, 2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 6.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const Vector3<double> p(c.x + 10.0, c.y, c.z);
    const Vector3<double> expected(1.0, 0.0, 0.0);

    EXPECT_TRUE(test::vec_near(op.closest_normal(p), expected, eps));
}

TEST(CylinderGeometryOperator, CentroidReturnsZeroWhenCenterNull) {
    constexpr geometry::CylinderGeometryOperator<double> op;
    EXPECT_TRUE(test::vec_near(op.centroid(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(CylinderGeometryOperator, CentroidReturnsCenterWhenValid) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 4.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(test::vec_near(op.centroid(), c, eps));
}

TEST(CylinderGeometryOperator, BoundReturnsDefaultWhenPointersNull) {
    constexpr geometry::CylinderGeometryOperator<double> op;

    const auto aabb = op.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(CylinderGeometryOperator, BoundMatchesAxisAlignedExtents) {
    const Vector3<double> c(1.0, -2.0, 3.0);
    constexpr double r = 2.0;
    constexpr double h = 6.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    const auto aabb = op.bound();

    const Vector3<double> lo(c.x - r, c.y - r, c.z - h * 0.5);
    const Vector3<double> hi(c.x + r, c.y + r, c.z + h * 0.5);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(CylinderGeometryOperator, IsValidFalseWhenRadiusOrHeightNull) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderGeometryOperator<double> op_rnull;
    op_rnull.center = atlas::raw_pointer_cast(&c);
    op_rnull.height = atlas::raw_pointer_cast(&h);
    EXPECT_FALSE(op_rnull.is_valid());

    geometry::CylinderGeometryOperator<double> op_hnull;
    op_hnull.center = atlas::raw_pointer_cast(&c);
    op_hnull.radius = atlas::raw_pointer_cast(&r);
    EXPECT_FALSE(op_hnull.is_valid());
}

TEST(CylinderGeometryOperator, IsValidTrueForPositiveRadiusAndHeight) {
    const Vector3<double> c(0.0, 0.0, 0.0);
    constexpr double r = 1.0;
    constexpr double h = 2.0;

    geometry::CylinderGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&c);
    op.radius = atlas::raw_pointer_cast(&r);
    op.height = atlas::raw_pointer_cast(&h);

    EXPECT_TRUE(op.is_valid());
}

TEST(CylinderGeometryOperator, IsValidFalseForNonPositiveRadiusOrHeight) {
    const Vector3<double> c(0.0, 0.0, 0.0);

    constexpr double r0 = 0.0;
    constexpr double rn = -1.0;
    constexpr double h0 = 0.0;
    constexpr double hn = -2.0;

    constexpr double r_ok = 1.0;
    constexpr double h_ok = 2.0;

    geometry::CylinderGeometryOperator<double> op_r0;
    op_r0.center = atlas::raw_pointer_cast(&c);
    op_r0.radius = atlas::raw_pointer_cast(&r0);
    op_r0.height = atlas::raw_pointer_cast(&h_ok);
    EXPECT_FALSE(op_r0.is_valid());

    geometry::CylinderGeometryOperator<double> op_rn;
    op_rn.center = atlas::raw_pointer_cast(&c);
    op_rn.radius = atlas::raw_pointer_cast(&rn);
    op_rn.height = atlas::raw_pointer_cast(&h_ok);
    EXPECT_FALSE(op_rn.is_valid());

    geometry::CylinderGeometryOperator<double> op_h0;
    op_h0.center = atlas::raw_pointer_cast(&c);
    op_h0.radius = atlas::raw_pointer_cast(&r_ok);
    op_h0.height = atlas::raw_pointer_cast(&h0);
    EXPECT_FALSE(op_h0.is_valid());

    geometry::CylinderGeometryOperator<double> op_hn;
    op_hn.center = atlas::raw_pointer_cast(&c);
    op_hn.radius = atlas::raw_pointer_cast(&r_ok);
    op_hn.height = atlas::raw_pointer_cast(&hn);
    EXPECT_FALSE(op_hn.is_valid());
}