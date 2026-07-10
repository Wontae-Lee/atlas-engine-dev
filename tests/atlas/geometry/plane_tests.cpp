#include <atlas/geometry/plane.h>

#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Plane;
using atlas::Ray;

constexpr float kTol = 1e-5f;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, kTol);
    EXPECT_NEAR(actual.y, expected.y, kTol);
    EXPECT_NEAR(actual.z, expected.z, kTol);
}

}

TEST(Plane, DefaultIsZPlaneFacingPositiveZ) {
    const Plane plane;

    expect_vec_near(plane.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(plane.offset, 0.0f);
    EXPECT_TRUE(plane.is_valid());
}

TEST(Plane, NormalOffsetConstructorStoresParameters) {
    const Plane plane(Float3(0.0f, 0.0f, 1.0f), -2.0f);

    expect_vec_near(plane.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(plane.offset, -2.0f);

    // Plane z = 2, so the origin sits two units on the far (interior) side.
    EXPECT_NEAR(plane.signed_distance(Float3(0.0f, 0.0f, 0.0f)), -2.0f, kTol);
}

TEST(Plane, PointNormalConstructorDerivesOffset) {
    // offset = -(normal . point) = -(1 * 5) = -5.
    const Plane plane(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_FLOAT_EQ(plane.offset, -5.0f);
    EXPECT_NEAR(plane.signed_distance(Float3(0.0f, 0.0f, 5.0f)), 0.0f, kTol);
}

TEST(Plane, IsValidRejectsZeroNormal) {
    const Plane degenerate(Float3(0.0f, 0.0f, 0.0f), 0.0f);

    EXPECT_FALSE(degenerate.is_valid());
}

TEST(Plane, IsValidRejectsNonFiniteOffset) {
    const Plane bad(Float3(0.0f, 0.0f, 1.0f),
                    std::numeric_limits<float>::infinity());

    EXPECT_FALSE(bad.is_valid());
}

TEST(Plane, SignedDistanceIsPositiveOnNormalSide) {
    const Plane plane; // z = 0, normal +z.

    // Positive on the +z (normal-facing) side, negative behind.
    EXPECT_NEAR(plane.signed_distance(Float3(1.0f, 2.0f, 3.0f)), 3.0f, kTol);
    EXPECT_NEAR(plane.signed_distance(Float3(1.0f, 2.0f, -4.0f)), -4.0f, kTol);
}

TEST(Plane, ClosestPointProjectsOntoPlane) {
    const Plane plane;

    expect_vec_near(plane.closest_point(Float3(1.0f, 2.0f, 3.0f)),
                    Float3(1.0f, 2.0f, 0.0f));
}

TEST(Plane, ClosestNormalIsTheStoredUnitNormal) {
    const Plane plane;

    const Float3 n = plane.closest_normal(Float3(7.0f, -3.0f, 4.0f));

    expect_vec_near(n, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(n.length(), 1.0f, kTol);
}

TEST(Plane, IsInsideIsTheFarHalfSpace) {
    const Plane plane; // interior is normal . p + offset <= 0.

    EXPECT_TRUE(plane.is_inside(Float3(0.0f, 0.0f, -1.0f)));
    EXPECT_FALSE(plane.is_inside(Float3(0.0f, 0.0f, 1.0f)));
}

TEST(Plane, IsInsideToleranceAdmitsPointsSlightlyInFront) {
    const Plane plane;

    // A point 0.05 in front is outside at tol 0 but inside once tol >= 0.05.
    EXPECT_FALSE(plane.is_inside(Float3(0.0f, 0.0f, 0.05f)));
    EXPECT_TRUE(plane.is_inside(Float3(0.0f, 0.0f, 0.05f), 0.1f));
}

TEST(Plane, IsOnSurfaceHonorsToleranceShell) {
    const Plane plane;

    EXPECT_TRUE(plane.is_on_surface(Float3(0.0f, 0.0f, 0.05f), 0.1f));
    EXPECT_FALSE(plane.is_on_surface(Float3(0.0f, 0.0f, 0.2f), 0.1f));
}

TEST(Plane, IsOnSurfaceRejectsNegativeTolerance) {
    const Plane plane;

    EXPECT_FALSE(plane.is_on_surface(Float3(0.0f, 0.0f, 0.0f), -0.1f));
}

TEST(Plane, CentroidIsTheOrigin) {
    const Plane plane;

    expect_vec_near(plane.centroid(), Float3(0.0f, 0.0f, 0.0f));
}

TEST(Plane, BoundSpansTheWholeFloatRange) {
    const Plane plane;
    const AABB box = plane.bound();

    EXPECT_TRUE(box.is_valid());
    EXPECT_TRUE(box.contains(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(box.contains(Float3(1e30f, -1e30f, 1e30f)));
    EXPECT_FLOAT_EQ(box.lower_corner.z, std::numeric_limits<float>::lowest());
    EXPECT_FLOAT_EQ(box.upper_corner.z, std::numeric_limits<float>::max());
}

TEST(Plane, TraceHitsFromOutside) {
    const Plane plane; // z = 0.

    const HitSurface hit = plane.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                           Float3(0.0f, 0.0f, -1.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, kTol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Plane, TraceMissesWhenPointingAway) {
    const Plane plane;

    // Crossing lies behind the origin (t < 0): reported as a miss.
    const HitSurface hit = plane.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                           Float3(0.0f, 0.0f, 1.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Plane, TraceMissesWhenParallelAndOffPlane) {
    const Plane plane;

    const HitSurface hit = plane.trace(Ray(Float3(0.0f, 0.0f, 5.0f),
                                           Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Plane, TraceGrazesWhenParallelAndOnPlane) {
    const Plane plane;

    const HitSurface hit = plane.trace(Ray(Float3(0.0f, 0.0f, 0.0f),
                                           Float3(1.0f, 0.0f, 0.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 0.0f, kTol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Plane, BuilderBuildsValidatedPlane) {
    const Plane plane = Plane::builder()
                            .with_normal_offset(Float3(0.0f, 0.0f, 1.0f), -2.0f)
                            .build();

    expect_vec_near(plane.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(plane.offset, -2.0f);
}

TEST(Plane, BuilderRejectsZeroNormal) {
    EXPECT_THROW(
        static_cast<void>(Plane::builder().with_normal(Float3(0.0f, 0.0f, 0.0f)).build()),
        std::runtime_error);
}
