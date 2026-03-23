#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(BoxGeometryOperator, ClosestPointReturnsInputWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;

    const Vector3<double> p(1.25, -2.5, 3.75);
    EXPECT_TRUE(test::vec_near(op.closest_point(p), p, eps));
}

TEST(BoxGeometryOperator, ClosestPointOutsideClampsToBounds) {
    const Vector3<double> lo(-1.0, -2.0, -3.0);
    const Vector3<double> hi(1.0, 2.0, 3.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(5.0, -5.0, 0.5);
    const Vector3<double> expected(1.0, -2.0, 0.5);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(BoxGeometryOperator, ClosestPointInsidePushesToNearestFace) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(0.2, 0.7, 0.9);
    const Vector3<double> expected(0.2, 0.7, 1.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), expected, eps));
}

TEST(BoxGeometryOperator, ClosestNormalReturnsZeroWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;

    const Vector3<double> p(1.0, 2.0, 3.0);
    EXPECT_TRUE(test::vec_near(op.closest_normal(p), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(BoxGeometryOperator, ClosestNormalInsideReturnsAxisUnitVectorOutward) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(-0.9, 0.2, 0.3);
    const Vector3<double> n = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(-1.0, 0.0, 0.0), eps));
}

TEST(BoxGeometryOperator, ClosestNormalOutsideUsesMajorAxisOfDisplacement) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(3.0, 2.0, 0.5);
    const Vector3<double> n = op.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(BoxGeometryOperator, SignedDistanceReturnsInfWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;

    const Vector3<double> p(0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(op.signed_distance(p)));
}

TEST(BoxGeometryOperator, SignedDistanceIsNegativeInside) {
    const Vector3<double> lo(-1.0, -2.0, -3.0);
    const Vector3<double> hi(1.0, 2.0, 3.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(0.0, 0.0, 0.0);

    const double d = op.signed_distance(p);
    EXPECT_LT(d, 0.0);
    EXPECT_NEAR(d, -1.0, eps);
}

TEST(BoxGeometryOperator, SignedDistanceIsPositiveOutside) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> p(4.0, 0.0, 0.0);

    const double d = op.signed_distance(p);
    EXPECT_GT(d, 0.0);
    EXPECT_NEAR(d, 3.0, eps);
}

TEST(BoxGeometryOperator, IsInsideClassifiesInteriorAndToleranceBand) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_FALSE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0)));
    EXPECT_TRUE(op.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(BoxGeometryOperator, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const Vector3<double> lo(-1.0, -1.0, -1.0);
    const Vector3<double> hi(1.0, 1.0, 1.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 0.25, 0.0)));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(0.0, 0.0, 0.0)));
    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.1, 0.0, 0.0), 0.15));
}

TEST(BoxGeometryOperator, CentroidReturnsZeroWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;
    EXPECT_TRUE(test::vec_near(op.centroid(), Vector3<double>(0.0, 0.0, 0.0), eps));
}

TEST(BoxGeometryOperator, CentroidIsMidpointOfBounds) {
    const Vector3<double> lo(-2.0, -4.0, -6.0);
    const Vector3<double> hi(6.0, 2.0, 4.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const Vector3<double> expected(2.0, -1.0, -1.0);

    EXPECT_TRUE(test::vec_near(op.centroid(), expected, eps));
}

TEST(BoxGeometryOperator, BoundReturnsDefaultWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;

    const auto aabb = op.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));
}

TEST(BoxGeometryOperator, BoundReturnsSameBoundsWhenValid) {
    const Vector3<double> lo(-2.0, -3.0, -4.0);
    const Vector3<double> hi(5.0, 6.0, 7.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    const auto aabb = op.bound();

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(BoxGeometryOperator, IsValidFalseWhenPointersNull) {
    constexpr geometry::BoxGeometryOperator<double> op;
    EXPECT_FALSE(op.is_valid());
}

TEST(BoxGeometryOperator, IsValidTrueForOrderedBounds) {
    const Vector3<double> lo(-1.0, -2.0, -3.0);
    const Vector3<double> hi(1.0, 2.0, 3.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    EXPECT_TRUE(op.is_valid());
}

TEST(BoxGeometryOperator, IsValidFalseForInvertedBounds) {
    const Vector3<double> lo(1.0, 0.0, 0.0);
    const Vector3<double> hi(-1.0, 0.0, 0.0);

    geometry::BoxGeometryOperator<double> op;
    op.lower_corner = atlas::raw_pointer_cast(&lo);
    op.upper_corner = atlas::raw_pointer_cast(&hi);

    EXPECT_FALSE(op.is_valid());
}