#include <atlas/geometry/cylinder.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Cylinder;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Ray;

/** Per-component approximate comparison for Float3. */
void
expect_vec_near(const Float3& value, const Float3& expected, const float eps = 1.0e-6f) {
    EXPECT_NEAR(value.x, expected.x, eps);
    EXPECT_NEAR(value.y, expected.y, eps);
    EXPECT_NEAR(value.z, expected.z, eps);
}

/** Default capped cylinder: origin, radius 1, height 1, z in [-0.5, 0.5]. */
Cylinder
unit_cylinder() {
    return Cylinder(Float3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f);
}

/** Open (uncapped) tube counterpart of unit_cylinder(). */
Cylinder
open_tube() {
    Cylinder c = unit_cylinder();
    c.open     = true;
    return c;
}

}

TEST(Cylinder, DefaultIsAUnitCappedCylinderOnTheOrigin) {
    const Cylinder cylinder;

    expect_vec_near(cylinder.center, Float3(0.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(cylinder.radius, 1.0f);
    EXPECT_FLOAT_EQ(cylinder.height, 1.0f);
    EXPECT_FALSE(cylinder.open);
    EXPECT_TRUE(cylinder.is_valid());
}

TEST(Cylinder, ConstructorProducesACappedCylinder) {
    const Cylinder cylinder(Float3(1.0f, 2.0f, 3.0f), 2.0f, 4.0f);

    expect_vec_near(cylinder.center, Float3(1.0f, 2.0f, 3.0f));
    EXPECT_FLOAT_EQ(cylinder.radius, 2.0f);
    EXPECT_FLOAT_EQ(cylinder.height, 4.0f);
    EXPECT_FALSE(cylinder.open);
}

TEST(Cylinder, IsValidRejectsNonPositiveRadiusOrHeight) {
    EXPECT_FALSE(Cylinder(Float3(0.0f), 0.0f, 1.0f).is_valid());
    EXPECT_FALSE(Cylinder(Float3(0.0f), -1.0f, 1.0f).is_valid());
    EXPECT_FALSE(Cylinder(Float3(0.0f), 1.0f, 0.0f).is_valid());
    EXPECT_FALSE(Cylinder(Float3(0.0f), 1.0f, -1.0f).is_valid());
}

TEST(Cylinder, SignedDistanceIsNegativeInside) {
    const Cylinder cylinder = unit_cylinder();
    // Deepest penetration is min(radius, height/2) = 0.5 toward a cap.
    EXPECT_FLOAT_EQ(cylinder.signed_distance(Float3(0.0f, 0.0f, 0.0f)), -0.5f);
}

TEST(Cylinder, SignedDistanceIsZeroOnLateralWallAndCaps) {
    const Cylinder cylinder = unit_cylinder();
    EXPECT_FLOAT_EQ(cylinder.signed_distance(Float3(1.0f, 0.0f, 0.0f)), 0.0f);
    EXPECT_FLOAT_EQ(cylinder.signed_distance(Float3(0.0f, 0.0f, 0.5f)), 0.0f);
}

TEST(Cylinder, SignedDistanceIsPositiveOutside) {
    const Cylinder cylinder = unit_cylinder();

    // Radially outside the wall.
    EXPECT_FLOAT_EQ(cylinder.signed_distance(Float3(2.0f, 0.0f, 0.0f)), 1.0f);
    // Axially above the top cap.
    EXPECT_FLOAT_EQ(cylinder.signed_distance(Float3(0.0f, 0.0f, 1.0f)), 0.5f);
    // Past the rim (both excesses positive).
    EXPECT_NEAR(cylinder.signed_distance(Float3(2.0f, 0.0f, 1.0f)), std::sqrt(1.25f), 1.0e-6f);
}

TEST(Cylinder, ClosestPointForInteriorPointsExitsNearestFeature) {
    const Cylinder cylinder = unit_cylinder();

    // Nearest feature is the lateral wall.
    expect_vec_near(cylinder.closest_point(Float3(0.9f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    // Nearest feature is the top cap.
    expect_vec_near(cylinder.closest_point(Float3(0.0f, 0.0f, 0.4f)), Float3(0.0f, 0.0f, 0.5f));
    // Nearest feature is the bottom cap.
    expect_vec_near(cylinder.closest_point(Float3(0.0f, 0.0f, -0.4f)), Float3(0.0f, 0.0f, -0.5f));
}

TEST(Cylinder, ClosestPointForExteriorPoints) {
    const Cylinder cylinder = unit_cylinder();

    // Radially outside, within the height band -> lateral wall.
    expect_vec_near(cylinder.closest_point(Float3(2.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    // Above the axis -> top cap.
    expect_vec_near(cylinder.closest_point(Float3(0.0f, 0.0f, 2.0f)), Float3(0.0f, 0.0f, 0.5f));
    // Past the rim -> the top edge circle.
    expect_vec_near(cylinder.closest_point(Float3(2.0f, 0.0f, 2.0f)), Float3(1.0f, 0.0f, 0.5f));
}

TEST(Cylinder, ClosestNormalIsUnitAndSelectsTheFeature) {
    const Cylinder cylinder = unit_cylinder();

    const Float3 side = cylinder.closest_normal(Float3(0.9f, 0.0f, 0.0f));
    expect_vec_near(side, Float3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(side.length(), 1.0f, 1.0e-6f);

    expect_vec_near(cylinder.closest_normal(Float3(0.0f, 0.0f, 0.4f)), Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(cylinder.closest_normal(Float3(0.0f, 0.0f, -0.4f)), Float3(0.0f, 0.0f, -1.0f));

    // Exterior features.
    expect_vec_near(cylinder.closest_normal(Float3(2.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(cylinder.closest_normal(Float3(0.0f, 0.0f, 2.0f)), Float3(0.0f, 0.0f, 1.0f));
}

TEST(Cylinder, IsInsideRespectsTolerance) {
    const Cylinder cylinder = unit_cylinder();

    EXPECT_TRUE(cylinder.is_inside(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(cylinder.is_inside(Float3(0.95f, 0.0f, 0.0f)));

    EXPECT_FALSE(cylinder.is_inside(Float3(1.05f, 0.0f, 0.0f)));
    EXPECT_TRUE(cylinder.is_inside(Float3(1.05f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(cylinder.is_inside(Float3(1.05f, 0.0f, 0.0f), 0.01f));
}

TEST(Cylinder, IsOnSurfaceInteriorShellUsesTolerance) {
    const Cylinder cylinder = unit_cylinder();

    EXPECT_TRUE(cylinder.is_on_surface(Float3(1.0f, 0.0f, 0.0f)));
    // 0.05 inside the wall.
    EXPECT_TRUE(cylinder.is_on_surface(Float3(0.95f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(cylinder.is_on_surface(Float3(0.95f, 0.0f, 0.0f), 0.01f));
}

TEST(Cylinder, IsOnSurfaceExteriorShellUsesTolerance) {
    const Cylinder cylinder = unit_cylinder();

    // 0.05 outside the wall.
    EXPECT_TRUE(cylinder.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(cylinder.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.01f));
}

TEST(Cylinder, IsOnSurfaceRejectsDegenerateOrNegativeTolerance) {
    const Cylinder cylinder = unit_cylinder();
    EXPECT_FALSE(cylinder.is_on_surface(Float3(1.0f, 0.0f, 0.0f), -0.1f));

    const Cylinder degenerate(Float3(0.0f), 0.0f, 1.0f);
    EXPECT_FALSE(degenerate.is_on_surface(Float3(0.0f, 0.0f, 0.0f), 0.1f));
}

TEST(Cylinder, CentroidIsTheCenter) {
    const Cylinder cylinder(Float3(1.0f, 2.0f, 3.0f), 2.0f, 4.0f);
    expect_vec_near(cylinder.centroid(), Float3(1.0f, 2.0f, 3.0f));
}

TEST(Cylinder, BoundExtendsRadiusInXYAndHalfHeightInZ) {
    const Cylinder cylinder(Float3(1.0f, 2.0f, 3.0f), 2.0f, 4.0f);
    const AABB bound = cylinder.bound();

    expect_vec_near(bound.lower_corner, Float3(-1.0f, 0.0f, 1.0f));
    expect_vec_near(bound.upper_corner, Float3(3.0f, 4.0f, 5.0f));
    EXPECT_TRUE(bound.contains(cylinder.centroid()));
}

TEST(Cylinder, TraceHitsTheLateralWallFromOutside) {
    const Cylinder cylinder = unit_cylinder();
    const Ray ray(Float3(-5.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = cylinder.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 4.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(-1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(-1.0f, 0.0f, 0.0f));
}

TEST(Cylinder, TraceHitsTheTopCapFromAbove) {
    const Cylinder cylinder = unit_cylinder();
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, -1.0f));

    const HitSurface hit = cylinder.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 4.5f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.5f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Cylinder, TraceMisses) {
    const Cylinder cylinder = unit_cylinder();
    const Ray ray(Float3(-5.0f, 5.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_FALSE(cylinder.trace(ray).is_intersecting);
}

TEST(Cylinder, TraceFromInsideExitsTheWall) {
    const Cylinder cylinder = unit_cylinder();
    const Ray ray(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = cylinder.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Cylinder, TraceGrazingTangentToTheWallStillIntersects) {
    const Cylinder cylinder = unit_cylinder();
    // Tangent line x-parallel through y = 1 (== radius): a single grazing hit.
    const Ray ray(Float3(-5.0f, 1.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = cylinder.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(0.0f, 1.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 1.0f, 0.0f));
}

TEST(Cylinder, OpenTubeSignedDistanceIsRadialWithinTheBand) {
    const Cylinder tube = open_tube();
    // On the axis inside the band: signed radial distance to the wall.
    EXPECT_FLOAT_EQ(tube.signed_distance(Float3(0.0f, 0.0f, 0.0f)), -1.0f);
    EXPECT_FLOAT_EQ(tube.signed_distance(Float3(0.5f, 0.0f, 0.0f)), -0.5f);
}

TEST(Cylinder, OpenTubeClosestPointIsAlwaysOnTheWall) {
    const Cylinder tube = open_tube();
    // On the axis, an arbitrary +x wall direction is chosen.
    expect_vec_near(tube.closest_point(Float3(0.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(tube.closest_point(Float3(0.5f, 0.0f, 0.2f)), Float3(1.0f, 0.0f, 0.2f));
}

TEST(Cylinder, OpenTubeIsOnSurfaceUsesTheWallOnly) {
    const Cylinder tube = open_tube();
    EXPECT_TRUE(tube.is_on_surface(Float3(1.0f, 0.0f, 0.0f)));
    // The cap disk region is not part of an open tube's surface.
    EXPECT_FALSE(tube.is_on_surface(Float3(0.0f, 0.0f, 0.5f)));
}

TEST(Cylinder, OpenTubeTraceMissesAnAxialRayThroughTheOpenEnds) {
    const Cylinder tube = open_tube();
    // Straight down the axis: no caps to hit and the wall is parallel.
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, -1.0f));

    EXPECT_FALSE(tube.trace(ray).is_intersecting);
}

TEST(Cylinder, OpenTubeTraceHitsTheLateralWall) {
    const Cylinder tube = open_tube();
    const Ray ray(Float3(-5.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = tube.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 4.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(-1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(-1.0f, 0.0f, 0.0f));
}

TEST(Cylinder, BuilderBuildsValidatedCappedCylinder) {
    const Cylinder cylinder = Cylinder::builder()
                                  .with_center(Float3(1.0f, 2.0f, 3.0f))
                                  .with_radius(2.0f)
                                  .with_height(4.0f)
                                  .build();

    expect_vec_near(cylinder.center, Float3(1.0f, 2.0f, 3.0f));
    EXPECT_FLOAT_EQ(cylinder.radius, 2.0f);
    EXPECT_FLOAT_EQ(cylinder.height, 4.0f);
    EXPECT_FALSE(cylinder.open); // capped by default
    EXPECT_TRUE(cylinder.is_valid());
}

TEST(Cylinder, BuilderWithOpenProducesAnUncappedTube) {
    // Unlike the three-argument constructor (which forces open = false), the builder's
    // with_open(true) carries the flag through to the built cylinder.
    const Cylinder tube = Cylinder::builder()
                              .with_radius(1.0f)
                              .with_height(2.0f)
                              .with_open(true)
                              .build();

    EXPECT_TRUE(tube.open);
}

TEST(Cylinder, BuilderRejectsNonPositiveRadiusOrHeight) {
    EXPECT_THROW(static_cast<void>(Cylinder::builder().with_radius(0.0f).build()),
                 std::runtime_error);
    EXPECT_THROW(static_cast<void>(Cylinder::builder().with_height(-1.0f).build()),
                 std::runtime_error);
}

TEST(Cylinder, BuilderMakeHostSharedBuildsCylinder) {
    const auto cylinder = Cylinder::builder()
                              .with_radius(3.0f)
                              .with_height(6.0f)
                              .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(cylinder));
    EXPECT_FLOAT_EQ(cylinder->radius, 3.0f);
    EXPECT_FLOAT_EQ(cylinder->height, 6.0f);
}
