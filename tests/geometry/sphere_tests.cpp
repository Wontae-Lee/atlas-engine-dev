#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Sphere, DefaultConstructorSetsCanonicalParams) {

    // Default-constructed Sphere<T> should be the unit sphere at the origin.
    const geometry::Sphere<double> s;

    EXPECT_TRUE(test::vec_near(s.center, Vector3<double>(0.0, 0.0, 0.0), eps));
    EXPECT_NEAR(s.radius, 1.0, eps);
}

TEST(Sphere, ConstructorCopiesCenterAndRadius) {

    // Explicit constructor should copy the provided center/radius verbatim.
    const geometry::Sphere<double> s(Vector3<double>(1.0, -2.0, 3.0), 4.0);

    EXPECT_TRUE(test::vec_near(s.center, Vector3<double>(1.0, -2.0, 3.0), eps));
    EXPECT_NEAR(s.radius, 4.0, eps);
}

TEST(Sphere, CenterMemberIsWritable) {

    geometry::Sphere<double> s;

    const Vector3<double> center(9.0, 8.0, 7.0);
    s.center = center;

    EXPECT_TRUE(test::vec_near(s.center, center, eps));
}

TEST(Sphere, RadiusMemberIsWritable) {

    geometry::Sphere<double> s;
    s.radius = 3.5;

    EXPECT_NEAR(s.radius, 3.5, eps);
}

TEST(Sphere, MakeTraceOperatorReturnsSphereTraceOperatorVariant) {

    const geometry::Sphere<double> s;
    const auto op = s.make_trace_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Sphere);
}

TEST(Sphere, MakeQueryOperatorReturnsSphereVariant) {

    const geometry::Sphere<double> s;
    const auto op = s.make_query_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Sphere);
}

TEST(Sphere, ClosestPointProjectsToSurfaceAlongRadialDirection) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    // (3,4,0) lies 5 units from center, so the closest point on radius 2 sphere is
    // 2/5 of the displacement vector.
    const Vector3<double> p(3.0, 4.0, 0.0);
    const Vector3<double> expected(1.2, 1.6, 0.0);

    EXPECT_TRUE(test::vec_near(s.closest_point(p), expected, eps));
}

TEST(Sphere, ClosestNormalIsNormalizedRadialDirection) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 10.0);

    const Vector3<double> p(3.0, 4.0, 0.0);
    const Vector3<double> expected(0.6, 0.8, 0.0);

    EXPECT_TRUE(test::vec_near(s.closest_normal(p), expected, eps));
}

TEST(Sphere, SignedDistanceIsPositiveOutsideAndNegativeInside) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    EXPECT_GT(s.signed_distance(Vector3<double>(5.0, 0.0, 0.0)), 0.0);
    EXPECT_LT(s.signed_distance(Vector3<double>(0.0, 0.0, 0.0)), 0.0);
    EXPECT_NEAR(s.signed_distance(Vector3<double>(2.0, 0.0, 0.0)), 0.0, eps);
}

TEST(Sphere, IsInsideClassifiesInteriorAndToleranceBand) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    EXPECT_TRUE(s.is_inside(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(s.is_inside(Vector3<double>(2.2, 0.0, 0.0), 0.0));
    EXPECT_TRUE(s.is_inside(Vector3<double>(2.2, 0.0, 0.0), 0.25));
}

TEST(Sphere, IsOnSurfaceDetectsBoundaryWithTolerance) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 2.0);

    EXPECT_TRUE(s.is_on_surface(Vector3<double>(2.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(s.is_on_surface(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_TRUE(s.is_on_surface(Vector3<double>(2.1, 0.0, 0.0), 0.15));
}

TEST(Sphere, CentroidEqualsCenter) {

    const Vector3<double> center(1.0, -2.0, 3.0);
    const geometry::Sphere<double> s(center, 2.0);

    EXPECT_TRUE(test::vec_near(s.centroid(), center, eps));
}

TEST(Sphere, BoundIsCenterPlusMinusRadiusVector) {

    const Vector3<double> center(1.0, -2.0, 3.0);
    constexpr double radius = 2.5;
    const geometry::Sphere<double> s(center, radius);

    const auto aabb = s.bound();
    const Vector3<double> dr(radius, radius, radius);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, center - dr, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, center + dr, eps));
}

TEST(Sphere, IsValidTrueForPositiveRadius) {

    const geometry::Sphere<double> s(Vector3<double>(0.0, 0.0, 0.0), 1.0);
    EXPECT_TRUE(s.is_valid());
}

TEST(Sphere, IsValidFalseForNonPositiveRadius) {

    const geometry::Sphere<double> s0(Vector3<double>(0.0, 0.0, 0.0), 0.0);
    const geometry::Sphere<double> sn(Vector3<double>(0.0, 0.0, 0.0), -1.0);

    EXPECT_FALSE(s0.is_valid());
    EXPECT_FALSE(sn.is_valid());
}

TEST(Sphere, TypeReturnsSphereGeometryType) {

    const geometry::Sphere<double> s;
    EXPECT_EQ(s.type(), geometry::GeometryType::Sphere);
}
