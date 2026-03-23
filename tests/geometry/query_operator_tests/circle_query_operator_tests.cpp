#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(CircleGeometryOperator, ClosestPointReturnsInputWhenStorageMissing) {
    constexpr geometry::CircleGeometryOperator<double> op;
    const Vector3<double> p(1.0, 2.0, 3.0);

    EXPECT_TRUE(test::vec_near(op.closest_point(p), p, eps));
}

TEST(CircleGeometryOperator, ClosestPointProjectsToDiskInterior) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    EXPECT_TRUE(test::vec_near(
        op.closest_point(Vector3<double>(1.0, 1.0, 3.0)),
        Vector3<double>(1.0, 1.0, 0.0),
        eps));
}

TEST(CircleGeometryOperator, ClosestPointClampsOutsideDiskToBoundary) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    EXPECT_TRUE(test::vec_near(
        op.closest_point(Vector3<double>(3.0, 0.0, 1.0)),
        Vector3<double>(2.0, 0.0, 0.0),
        eps));
}

TEST(CircleGeometryOperator, ClosestNormalReturnsNormalizedStoredNormal) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 2.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    EXPECT_TRUE(test::vec_near(
        op.closest_normal(Vector3<double>(1.0, 0.0, 0.0)),
        Vector3<double>(0.0, 0.0, 1.0),
        eps));
}

TEST(CircleGeometryOperator, SignedDistanceUsesPlaneSideSign) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    EXPECT_GT(op.signed_distance(Vector3<double>(0.0, 0.0, 1.0)), 0.0);
    EXPECT_LT(op.signed_distance(Vector3<double>(0.0, 0.0, -1.0)), 0.0);
}

TEST(CircleGeometryOperator, IsOnSurfaceAndIsInsideFollowSignedDistance) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    EXPECT_TRUE(op.is_on_surface(Vector3<double>(1.0, 1.0, 0.0), 0.0));
    EXPECT_FALSE(op.is_on_surface(Vector3<double>(1.0, 1.0, 0.3), 0.0));
    EXPECT_TRUE(op.is_inside(Vector3<double>(0.0, 0.0, -0.1), 0.0));
    EXPECT_FALSE(op.is_inside(Vector3<double>(0.0, 0.0, 0.1), 0.0));
}

TEST(CircleGeometryOperator, BoundReflectsDiskOrientation) {
    const Vector3<double> center(1.0, 2.0, 3.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    const auto bound = op.bound();
    EXPECT_TRUE(test::vec_near(bound.lower_corner, Vector3<double>(-1.0, 0.0, 3.0), eps));
    EXPECT_TRUE(test::vec_near(bound.upper_corner, Vector3<double>(3.0, 4.0, 3.0), eps));
}

TEST(CircleGeometryOperator, TraceHitsDiskWhenRayIntersectsPlaneInsideRadius) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    const Ray<double> ray(Vector3<double>(0.5, 0.5, 5.0), Vector3<double>(0.0, 0.0, -1.0));
    const auto hit = op.trace(ray);

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0, eps);
    EXPECT_TRUE(test::vec_near(hit.point, Vector3<double>(0.5, 0.5, 0.0), eps));
    EXPECT_TRUE(test::vec_near(hit.normal, Vector3<double>(0.0, 0.0, 1.0), eps));
}

TEST(CircleGeometryOperator, TraceMissesDiskWhenPlaneHitIsOutsideRadius) {
    const Vector3<double> center(0.0, 0.0, 0.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const double radius = 2.0;

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.normal = atlas::raw_pointer_cast(&normal);
    op.radius = atlas::raw_pointer_cast(&radius);

    const Ray<double> ray(Vector3<double>(3.0, 0.0, 5.0), Vector3<double>(0.0, 0.0, -1.0));
    const auto hit = op.trace(ray);

    EXPECT_FALSE(hit.is_intersecting);
}