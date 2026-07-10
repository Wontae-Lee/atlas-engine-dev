#include <atlas/geometry/circle.h>

#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Circle;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Ray;

constexpr float kTol = 1e-5f;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, kTol);
    EXPECT_NEAR(actual.y, expected.y, kTol);
    EXPECT_NEAR(actual.z, expected.z, kTol);
}

}

TEST(Circle, DefaultIsUnitDiskInZPlane) {
    const Circle circle;

    expect_vec_near(circle.center, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(circle.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(circle.radius, 1.0f);
    EXPECT_TRUE(circle.is_valid());
}

TEST(Circle, ConstructorStoresParameters) {
    const Circle circle(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 1.0f, 0.0f), 2.0f);

    expect_vec_near(circle.center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(circle.normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_FLOAT_EQ(circle.radius, 2.0f);
}

TEST(Circle, IsValidRejectsZeroNormal) {
    EXPECT_FALSE(Circle(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 1.0f).is_valid());
}

TEST(Circle, IsValidRejectsNonPositiveRadius) {
    EXPECT_FALSE(Circle(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f), 0.0f).is_valid());
}

TEST(Circle, ClosestPointOnFaceDropsOutOfPlaneComponent) {
    const Circle circle; // unit disk in z = 0.

    // Projection lands within the rim, so only the z offset is removed.
    expect_vec_near(circle.closest_point(Float3(0.5f, 0.0f, 2.0f)),
                    Float3(0.5f, 0.0f, 0.0f));
}

TEST(Circle, ClosestPointOutsideRimSnapsToRim) {
    const Circle circle;

    // Projection is 3 units out; it is pushed back onto the unit rim.
    expect_vec_near(circle.closest_point(Float3(3.0f, 0.0f, 1.0f)),
                    Float3(1.0f, 0.0f, 0.0f));
}

TEST(Circle, ClosestNormalIsUnitFaceNormal) {
    const Circle circle;

    const Float3 n = circle.closest_normal(Float3(4.0f, -2.0f, 7.0f));

    expect_vec_near(n, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(n.length(), 1.0f, kTol);
}

TEST(Circle, SignedDistanceSignFollowsNormalSide) {
    const Circle circle;

    // Same face point, opposite sides of the plane.
    EXPECT_NEAR(circle.signed_distance(Float3(0.5f, 0.0f, 2.0f)), 2.0f, kTol);
    EXPECT_NEAR(circle.signed_distance(Float3(0.5f, 0.0f, -2.0f)), -2.0f, kTol);
}

TEST(Circle, IsInsideIsTheBackHalfSpace) {
    const Circle circle;

    // Behind the plane (z < 0) counts as interior at the default tolerance.
    EXPECT_TRUE(circle.is_inside(Float3(0.0f, 0.0f, -1.0f)));
    // In front of the plane and outside the tolerance shell is not interior.
    EXPECT_FALSE(circle.is_inside(Float3(0.0f, 0.0f, 1.0f)));
}

TEST(Circle, IsInsideToleranceAdmitsFrontShell) {
    const Circle circle;

    // A point 0.05 in front of the disk falls inside the tolerance shell.
    EXPECT_TRUE(circle.is_inside(Float3(0.0f, 0.0f, 0.05f), 0.1f));
}

TEST(Circle, IsInsideRejectsInvalidDisk) {
    const Circle degenerate(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f), 0.0f);

    EXPECT_FALSE(degenerate.is_inside(Float3(0.0f, 0.0f, -1.0f)));
}

TEST(Circle, IsOnSurfaceShellInsideAndOutside) {
    const Circle circle;

    EXPECT_TRUE(circle.is_on_surface(Float3(0.0f, 0.0f, 0.05f), 0.1f));
    EXPECT_FALSE(circle.is_on_surface(Float3(0.0f, 0.0f, 0.2f), 0.1f));
}

TEST(Circle, IsOnSurfaceUsesRadialOvershootPastRim) {
    const Circle circle;

    // Overshoot 0.05 past the rim, still within the 0.1 shell.
    EXPECT_TRUE(circle.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.1f));
    // Overshoot 0.2 past the rim, outside the shell.
    EXPECT_FALSE(circle.is_on_surface(Float3(1.2f, 0.0f, 0.0f), 0.1f));
}

TEST(Circle, IsOnSurfaceRejectsNegativeTolerance) {
    const Circle circle;

    EXPECT_FALSE(circle.is_on_surface(Float3(0.0f, 0.0f, 0.0f), -0.1f));
}

TEST(Circle, CentroidIsTheCenter) {
    const Circle circle(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);

    expect_vec_near(circle.centroid(), Float3(1.0f, 2.0f, 3.0f));
}

TEST(Circle, BoundIsTightForAxisAlignedDisk) {
    const Circle circle; // z-facing unit disk collapses flat in z.
    const AABB box = circle.bound();

    expect_vec_near(box.lower_corner, Float3(-1.0f, -1.0f, 0.0f));
    expect_vec_near(box.upper_corner, Float3(1.0f, 1.0f, 0.0f));
    EXPECT_TRUE(box.contains(Float3(1.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(box.contains(Float3(0.0f, -1.0f, 0.0f)));
}

TEST(Circle, BoundCollapsesForDegenerateDisk) {
    const Circle degenerate(Float3(2.0f, 3.0f, 4.0f), Float3(0.0f, 0.0f, 0.0f), 1.0f);
    const AABB box = degenerate.bound();

    expect_vec_near(box.lower_corner, Float3(2.0f, 3.0f, 4.0f));
    expect_vec_near(box.upper_corner, Float3(2.0f, 3.0f, 4.0f));
}

TEST(Circle, TraceHitsDiskFaceFromOutside) {
    const Circle circle;

    const HitSurface hit = circle.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                            Float3(0.0f, 0.0f, -1.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, kTol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Circle, TraceMissesWhenHitFallsOutsideRim) {
    const Circle circle;

    // Crosses the plane at (3, 0, 0), well beyond the unit rim.
    const HitSurface hit = circle.trace(Ray(Float3(3.0f, 0.0f, 5.0f),
                                            Float3(0.0f, 0.0f, -1.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Circle, TraceMissesWhenPointingAway) {
    const Circle circle;

    const HitSurface hit = circle.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                            Float3(0.0f, 0.0f, 1.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Circle, TraceGrazesWhenParallelAndInsideRim) {
    const Circle circle;

    // Ray lies in the disk plane and starts inside the rim: grazing hit at t = 0.
    const HitSurface hit = circle.trace(Ray(Float3(0.0f, 0.0f, 0.0f),
                                            Float3(1.0f, 0.0f, 0.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 0.0f, kTol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Circle, TraceMissesWhenParallelAndOutsideRim) {
    const Circle circle;

    // In-plane ray but origin lies outside the rim.
    const HitSurface hit = circle.trace(Ray(Float3(5.0f, 0.0f, 0.0f),
                                            Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Circle, TraceMissesForInvalidDisk) {
    const Circle degenerate(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 1.0f);

    const HitSurface hit = degenerate.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                                Float3(0.0f, 0.0f, -1.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Circle, BuilderBuildsValidatedDisk) {
    const Circle circle = Circle::builder()
                              .with_center(Float3(1.0f, 2.0f, 3.0f))
                              .with_normal(Float3(0.0f, 1.0f, 0.0f))
                              .with_radius(2.0f)
                              .build();

    expect_vec_near(circle.center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(circle.normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_FLOAT_EQ(circle.radius, 2.0f);
}

TEST(Circle, BuilderRejectsNonPositiveRadius) {
    EXPECT_THROW(
        static_cast<void>(Circle::builder().with_radius(0.0f).build()),
        std::runtime_error);
}

TEST(Circle, BuilderRejectsZeroNormal) {
    EXPECT_THROW(
        static_cast<void>(Circle::builder().with_normal(Float3(0.0f, 0.0f, 0.0f)).build()),
        std::runtime_error);
}
