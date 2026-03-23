#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Cylinder, DefaultConstructorSetsCanonicalParams) {

    const geometry::Cylinder<double> c;

    EXPECT_TRUE(test::vec_near(c.center, Vector3<double>(0.0, 0.0, 0.0), eps));
    EXPECT_NEAR(c.radius, 1.0, eps);
    EXPECT_NEAR(c.height, 1.0, eps);
}

TEST(Cylinder, ParamsConstructorCopiesMemberParams) {

    const Vector3<double> center(1.0, -2.0, 3.0);
    constexpr double radius = 2.5;
    constexpr double height = 7.0;

    const geometry::Cylinder<double> c(center, radius, height);

    EXPECT_TRUE(test::vec_near(c.center, center, eps));
    EXPECT_NEAR(c.radius, radius, eps);
    EXPECT_NEAR(c.height, height, eps);
}

TEST(Cylinder, CenterMemberIsWritable) {

    geometry::Cylinder<double> c;

    const Vector3<double> center(9.0, 8.0, 7.0);
    c.center = center;

    EXPECT_TRUE(test::vec_near(c.center, center, eps));
}

TEST(Cylinder, RadiusMemberIsWritable) {

    geometry::Cylinder<double> c;

    c.radius = 3.0;

    EXPECT_NEAR(c.radius, 3.0, eps);
}

TEST(Cylinder, HeightMemberIsWritable) {

    geometry::Cylinder<double> c;

    c.height = 4.0;

    EXPECT_NEAR(c.height, 4.0, eps);
}

TEST(Cylinder, MakeGeometryOperatorReturnsCylinderGeometryOperatorVariant) {

    const geometry::Cylinder<double> c;

    const auto op = c.make_geometry_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Cylinder);
}

TEST(Cylinder, ClosestPointClampsToSideWallInXYAndCapsInZ) {

    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    const Vector3<double> p(2.0, 0.0, 5.0);

    const Vector3<double> expected(1.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(c.closest_point(p), expected, eps));
}

TEST(Cylinder, ClosestNormalIsRadialForSideWallOutsidePoint) {

    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    const Vector3<double> p(2.0, 0.0, 0.25);

    const Vector3<double> n = c.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(Cylinder, SignedDistanceIsPositiveOutsideAndNegativeInside) {

    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    const Vector3<double> p_out(3.0, 0.0, 0.0);

    const Vector3<double> p_in(0.0, 0.0, 0.0);

    EXPECT_GT(c.signed_distance(p_out), 0.0);
    EXPECT_LT(c.signed_distance(p_in), 0.0);
}

TEST(Cylinder, IsInsideClassifiesInteriorAndToleranceBand) {
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    EXPECT_TRUE(c.is_inside(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(c.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.0));
    EXPECT_TRUE(c.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(Cylinder, IsOnSurfaceDetectsSideWallAndCapTolerance) {
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    EXPECT_TRUE(c.is_on_surface(Vector3<double>(1.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(c.is_on_surface(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_TRUE(c.is_on_surface(Vector3<double>(0.0, 0.0, 1.1), 0.15));
}

TEST(Cylinder, CentroidEqualsCenter) {

    const Vector3<double> center(1.0, -2.0, 3.0);
    const geometry::Cylinder<double> c(center, 2.0, 4.0);

    EXPECT_TRUE(test::vec_near(c.centroid(), center, eps));
}

TEST(Cylinder, BoundMatchesAxisAlignedExtents) {

    const Vector3<double> center(1.0, -2.0, 3.0);
    constexpr double radius = 2.0;
    constexpr double height = 6.0;

    const geometry::Cylinder<double> c(center, radius, height);

    const auto aabb = c.bound();

    const Vector3<double> lo(center.x - radius, center.y - radius, center.z - height * 0.5);
    const Vector3<double> hi(center.x + radius, center.y + radius, center.z + height * 0.5);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(Cylinder, IsValidTrueForPositiveRadiusAndHeight) {

    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    EXPECT_TRUE(c.is_valid());
}

TEST(Cylinder, IsValidFalseForNonPositiveRadiusOrHeight) {

    const geometry::Cylinder<double> r0(Vector3<double>(0.0, 0.0, 0.0), 0.0, 2.0);
    const geometry::Cylinder<double> h0(Vector3<double>(0.0, 0.0, 0.0), 1.0, 0.0);
    const geometry::Cylinder<double> rn(Vector3<double>(0.0, 0.0, 0.0), -1.0, 2.0);
    const geometry::Cylinder<double> hn(Vector3<double>(0.0, 0.0, 0.0), 1.0, -2.0);

    EXPECT_FALSE(r0.is_valid());
    EXPECT_FALSE(h0.is_valid());
    EXPECT_FALSE(rn.is_valid());
    EXPECT_FALSE(hn.is_valid());
}

TEST(Cylinder, TypeReturnsCylinderGeometryType) {

    const geometry::Cylinder<double> c;

    EXPECT_EQ(c.type(), geometry::GeometryType::Cylinder);
}