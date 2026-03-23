#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Box, DefaultConstructorSetsCanonicalBounds) {

    const geometry::Box<double> b;

    EXPECT_TRUE(test::vec_near(b.lower_corner, Vector3<double>(-1.0, -1.0, -1.0), eps));
    EXPECT_TRUE(test::vec_near(b.upper_corner, Vector3<double>(+1.0, +1.0, +1.0), eps));
}

TEST(Box, BoundsConstructorCopiesMemberBounds) {

    const Vector3<double> lo(-2.0, -3.0, -4.0);
    const Vector3<double> hi(+5.0, +6.0, +7.0);

    const geometry::Box<double> b(lo, hi);

    EXPECT_TRUE(test::vec_near(b.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(b.upper_corner, hi, eps));
}

TEST(Box, LowerCornerMemberIsWritable) {

    geometry::Box<double> b;

    const Vector3<double> lo(-9.0, -8.0, -7.0);
    b.lower_corner = lo;

    EXPECT_TRUE(test::vec_near(b.lower_corner, lo, eps));
}

TEST(Box, UpperCornerMemberIsWritable) {

    geometry::Box<double> b;

    const Vector3<double> hi(9.0, 8.0, 7.0);
    b.upper_corner = hi;

    EXPECT_TRUE(test::vec_near(b.upper_corner, hi, eps));
}

TEST(Box, MakeGeometryOperatorReturnsBoxGeometryOperatorVariant) {

    const geometry::Box<double> b;

    const auto op = b.make_geometry_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Box);
}

TEST(Box, ClosestPointDelegatesToGeometryOperator) {

    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    const Vector3<double> p(2.0, 0.5, -3.0);
    const Vector3<double> expected(1.0, 0.5, -1.0);

    EXPECT_TRUE(test::vec_near(b.closest_point(p), expected, eps));
}

TEST(Box, ClosestNormalReturnsAxisUnitNormalForOutsidePoint) {

    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    const Vector3<double> p(2.0, 0.25, 0.0);

    const Vector3<double> n = b.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(Box, SignedDistanceIsPositiveOutsideAndNegativeInside) {

    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    const Vector3<double> p_out(3.0, 0.0, 0.0);

    const Vector3<double> p_in(0.0, 0.0, 0.0);

    EXPECT_GT(b.signed_distance(p_out), 0.0);

    EXPECT_LT(b.signed_distance(p_in), 0.0);
}

TEST(Box, IsInsideClassifiesInteriorAndToleranceBand) {
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0));

    EXPECT_TRUE(b.is_inside(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(b.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.0));
    EXPECT_TRUE(b.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(Box, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0));

    EXPECT_TRUE(b.is_on_surface(Vector3<double>(1.0, 0.25, 0.0), 0.0));
    EXPECT_FALSE(b.is_on_surface(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_TRUE(b.is_on_surface(Vector3<double>(1.1, 0.0, 0.0), 0.15));
}

TEST(Box, CentroidIsMidpointOfBounds) {

    const geometry::Box<double> b(
        Vector3<double>(-2.0, -4.0, -6.0),
        Vector3<double>(+6.0, +2.0, +4.0));

    const Vector3<double> expected(2.0, -1.0, -1.0);

    EXPECT_TRUE(test::vec_near(b.centroid(), expected, eps));
}

TEST(Box, BoundReturnsAxisAlignedBoundingBoxWithSameBounds) {

    const Vector3<double> lo(-2.0, -3.0, -4.0);
    const Vector3<double> hi(+5.0, +6.0, +7.0);

    const geometry::Box<double> b(lo, hi);

    const auto aabb = b.bound();

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(Box, IsValidTrueForOrderedBounds) {

    const geometry::Box<double> b(
        Vector3<double>(-1.0, -2.0, -3.0),
        Vector3<double>(+1.0, +2.0, +3.0));

    EXPECT_TRUE(b.is_valid());
}

TEST(Box, IsValidFalseForInvertedBounds) {

    const geometry::Box<double> b(
        Vector3<double>(+1.0, 0.0, 0.0),
        Vector3<double>(-1.0, 0.0, 0.0));

    EXPECT_FALSE(b.is_valid());
}

TEST(Box, TypeReturnsBoxGeometryType) {

    const geometry::Box<double> b;

    EXPECT_EQ(b.type(), geometry::GeometryType::Box);
}