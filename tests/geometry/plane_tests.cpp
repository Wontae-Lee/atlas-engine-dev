#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Plane, DefaultConstructorSetsCanonicalParams) {

    const geometry::Plane<double> p;

    EXPECT_TRUE(test::vec_near(p.normal, Vector3<double>(0.0, 0.0, 1.0), eps));
    EXPECT_NEAR(p.offset, 0.0, eps);
}

TEST(Plane, NormalOffsetConstructorCopiesMemberParams) {

    const Vector3<double> n(0.0, 2.0, 0.0);
    constexpr double d = -3.5;

    const geometry::Plane<double> p(n, d);

    EXPECT_TRUE(test::vec_near(p.normal, n, eps));
    EXPECT_NEAR(p.offset, d, eps);
}

TEST(Plane, PointNormalConstructorSetsOffsetAsNegativeDot) {

    const Vector3<double> point(0.0, 0.0, 2.0);
    const Vector3<double> n(0.0, 0.0, 3.0);

    const geometry::Plane<double> p(point, n);

    EXPECT_TRUE(test::vec_near(p.normal, n, eps));

    EXPECT_NEAR(p.offset, -(n.dot(point)), eps);
}

TEST(Plane, NormalMemberIsWritable) {

    geometry::Plane<double> p;

    const Vector3<double> n(1.0, 0.0, 0.0);
    p.normal = n;

    EXPECT_TRUE(test::vec_near(p.normal, n, eps));
}

TEST(Plane, OffsetMemberIsWritable) {

    geometry::Plane<double> p;

    p.offset = 4.25;

    EXPECT_NEAR(p.offset, 4.25, eps);
}

TEST(Plane, MakeGeometryOperatorReturnsPlaneGeometryOperatorVariant) {

    const geometry::Plane<double> p;

    const auto op = p.make_geometry_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Plane);
}

TEST(Plane, ClosestPointProjectsToPlaneForZPlane) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> q(1.25, -3.5, 7.0);
    const Vector3<double> expected(1.25, -3.5, 0.0);

    EXPECT_TRUE(test::vec_near(p.closest_point(q), expected, eps));
}

TEST(Plane, ClosestNormalIsParallelToStoredNormalForZPlane) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> q(3.0, 4.0, 5.0);
    const Vector3<double> n = p.closest_normal(q);

    const double nx = std::abs(n.x);
    const double ny = std::abs(n.y);
    const double nz = std::abs(n.z);

    EXPECT_NEAR(nx, 0.0, eps);
    EXPECT_NEAR(ny, 0.0, eps);
    EXPECT_NEAR(nz, 1.0, eps);
}

TEST(Plane, SignedDistanceHasCorrectSignForZPlane) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> above(0.0, 0.0, 2.0);
    const Vector3<double> below(0.0, 0.0, -2.0);

    EXPECT_GT(p.signed_distance(above), 0.0);
    EXPECT_LT(p.signed_distance(below), 0.0);
}

TEST(Plane, IsInsideUsesHalfSpaceAndTolerance) {
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    EXPECT_TRUE(p.is_inside(Vector3<double>(0.0, 0.0, -1.0), 0.0));
    EXPECT_FALSE(p.is_inside(Vector3<double>(0.0, 0.0, 0.2), 0.0));
    EXPECT_TRUE(p.is_inside(Vector3<double>(0.0, 0.0, 0.2), 0.25));
}

TEST(Plane, IsOnSurfaceDetectsPlaneBand) {
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    EXPECT_TRUE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.0), 0.0));
    EXPECT_FALSE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.5), 0.0));
    EXPECT_TRUE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.1), 0.15));
}

TEST(Plane, CentroidLiesOnPlaneForZPlane) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> c = p.centroid();

    EXPECT_NEAR(p.normal.dot(c) + p.offset, 0.0, 1e-10);
}

TEST(Plane, BoundIsWellFormedForInfinitePrimitive) {

    const geometry::Plane<double> p;

    const auto aabb = p.bound();

    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));

    EXPECT_LE(aabb.lower_corner.x, aabb.upper_corner.x);
    EXPECT_LE(aabb.lower_corner.y, aabb.upper_corner.y);
    EXPECT_LE(aabb.lower_corner.z, aabb.upper_corner.z);
}

TEST(Plane, IsValidTrueForNonZeroFiniteNormalAndFiniteOffset) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 2.0), 1.0);

    EXPECT_TRUE(p.is_valid());
}

TEST(Plane, IsValidFalseForZeroNormal) {

    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 0.0), 0.0);

    EXPECT_FALSE(p.is_valid());
}

TEST(Plane, TypeReturnsPlaneGeometryType) {

    const geometry::Plane<double> p;

    EXPECT_EQ(p.type(), geometry::GeometryType::Plane);
}