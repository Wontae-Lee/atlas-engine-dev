#include <atlas/geometry/box.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Box;
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

}

TEST(Box, DefaultIsTheUnitCubeCenteredOnTheOrigin) {
    const Box box;

    expect_vec_near(box.lower_corner, Float3(-1.0f, -1.0f, -1.0f));
    expect_vec_near(box.upper_corner, Float3(1.0f, 1.0f, 1.0f));
    EXPECT_TRUE(box.is_valid());
}

TEST(Box, ConstructorStoresCornersVerbatim) {
    const Box box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 4.0f, 6.0f));

    expect_vec_near(box.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(box.upper_corner, Float3(2.0f, 4.0f, 6.0f));
}

TEST(Box, IsValidRejectsInvertedCorners) {
    const Box inverted(Float3(1.0f, 1.0f, 1.0f), Float3(-1.0f, -1.0f, -1.0f));
    EXPECT_FALSE(inverted.is_valid());
}

TEST(Box, IsValidRejectsNonFiniteCorners) {
    const float inf = std::numeric_limits<float>::infinity();
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(inf, 1.0f, 1.0f));
    EXPECT_FALSE(box.is_valid());
}

TEST(Box, IsValidRejectsAZeroExtentBox) {
    // A box collapsed to a point encloses no volume.
    const Box point(Float3(1.0f, 2.0f, 3.0f), Float3(1.0f, 2.0f, 3.0f));
    EXPECT_FALSE(point.is_valid());
}

TEST(Box, IsValidRejectsABoxCollapsedOnASingleAxis) {
    // Flat on z: still no volume, so is_inside could never be true.
    const Box slab(Float3(0.0f, 0.0f, 5.0f), Float3(2.0f, 3.0f, 5.0f));
    EXPECT_FALSE(slab.is_valid());
}

TEST(Box, SignedDistanceIsNegativeInside) {
    const Box box;

    EXPECT_FLOAT_EQ(box.signed_distance(Float3(0.0f, 0.0f, 0.0f)), -1.0f);
    EXPECT_FLOAT_EQ(box.signed_distance(Float3(0.5f, 0.0f, 0.0f)), -0.5f);
}

TEST(Box, SignedDistanceIsZeroOnTheSurface) {
    const Box box;
    EXPECT_FLOAT_EQ(box.signed_distance(Float3(1.0f, 0.0f, 0.0f)), 0.0f);
}

TEST(Box, SignedDistanceIsEuclideanOutside) {
    const Box box;

    EXPECT_FLOAT_EQ(box.signed_distance(Float3(2.0f, 0.0f, 0.0f)), 1.0f);
    EXPECT_NEAR(box.signed_distance(Float3(2.0f, 2.0f, 2.0f)), std::sqrt(3.0f), 1.0e-6f);
}

TEST(Box, ClosestPointPushesInteriorPointToNearestFace) {
    const Box box;
    expect_vec_near(box.closest_point(Float3(0.5f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
}

TEST(Box, ClosestPointClampsExteriorPoints) {
    const Box box;

    expect_vec_near(box.closest_point(Float3(2.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(box.closest_point(Float3(2.0f, 2.0f, 0.0f)), Float3(1.0f, 1.0f, 0.0f));
    expect_vec_near(box.closest_point(Float3(2.0f, 2.0f, 2.0f)), Float3(1.0f, 1.0f, 1.0f));
}

TEST(Box, ClosestNormalIsUnitAndSelectsTheFace) {
    const Box box;

    const Float3 inside_n = box.closest_normal(Float3(0.5f, 0.0f, 0.0f));
    expect_vec_near(inside_n, Float3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(inside_n.length(), 1.0f, 1.0e-6f);

    // Off a face: the offset axis dominates.
    expect_vec_near(box.closest_normal(Float3(2.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));

    // Off an edge (equal x/y overshoot): ties resolve toward the lower axis (x).
    expect_vec_near(box.closest_normal(Float3(2.0f, 2.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));

    // Off a corner with a dominant z overshoot.
    expect_vec_near(box.closest_normal(Float3(2.0f, 3.0f, 4.0f)), Float3(0.0f, 0.0f, 1.0f));
}

TEST(Box, IsInsideRespectsTolerance) {
    const Box box;

    EXPECT_TRUE(box.is_inside(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_FALSE(box.is_inside(Float3(1.05f, 0.0f, 0.0f)));
    EXPECT_TRUE(box.is_inside(Float3(1.05f, 0.0f, 0.0f), 0.1f));
    // Negative tolerance shrinks the box.
    EXPECT_FALSE(box.is_inside(Float3(0.95f, 0.0f, 0.0f), -0.1f));
}

TEST(Box, IsOnSurfaceInteriorShellUsesTolerance) {
    const Box box;

    EXPECT_TRUE(box.is_on_surface(Float3(1.0f, 0.0f, 0.0f)));
    // 0.05 in from the +x face.
    EXPECT_TRUE(box.is_on_surface(Float3(0.95f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(box.is_on_surface(Float3(0.95f, 0.0f, 0.0f), 0.01f));
}

TEST(Box, IsOnSurfaceExteriorShellUsesTolerance) {
    const Box box;

    // 0.05 outside the +x face.
    EXPECT_TRUE(box.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(box.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.01f));
}

TEST(Box, IsOnSurfaceRejectsNegativeTolerance) {
    const Box box;
    EXPECT_FALSE(box.is_on_surface(Float3(1.0f, 0.0f, 0.0f), -0.1f));
}

TEST(Box, CentroidIsTheMidpoint) {
    const Box box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 4.0f, 6.0f));
    expect_vec_near(box.centroid(), Float3(1.0f, 2.0f, 3.0f));
}

TEST(Box, BoundCoincidesWithTheBox) {
    const Box box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 3.0f, 4.0f));
    const AABB bound = box.bound();

    expect_vec_near(bound.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(bound.upper_corner, Float3(2.0f, 3.0f, 4.0f));
    EXPECT_TRUE(bound.contains(box.centroid()));
}

TEST(Box, TraceHitsFromOutside) {
    const Box box;
    const Ray ray(Float3(-5.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = box.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 4.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(-1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(-1.0f, 0.0f, 0.0f));
}

TEST(Box, TraceMissesWhenOffset) {
    const Box box;
    const Ray ray(Float3(-5.0f, 5.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_FALSE(box.trace(ray).is_intersecting);
}

TEST(Box, TraceFromInsideReportsTheExitFace) {
    const Box box;
    const Ray ray(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = box.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Box, TraceGrazingAlongAFaceStillIntersects) {
    const Box box;
    // Travels in the plane y = 1 (the top face), parallel to +x.
    const Ray ray(Float3(-5.0f, 1.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    const HitSurface hit = box.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 4.0f, 1.0e-6f);
    EXPECT_NEAR(hit.point.x, -1.0f, 1.0e-6f);
}

TEST(Box, BuilderBuildsValidatedBox) {
    const Box box = Box::builder()
                        .with_lower_corner(Float3(0.0f, 1.0f, 2.0f))
                        .with_upper_corner(Float3(3.0f, 4.0f, 5.0f))
                        .build();

    expect_vec_near(box.lower_corner, Float3(0.0f, 1.0f, 2.0f));
    expect_vec_near(box.upper_corner, Float3(3.0f, 4.0f, 5.0f));
    EXPECT_TRUE(box.is_valid());
}

TEST(Box, BuilderRejectsInvertedCorners) {
    // upper not strictly greater than lower on every axis is an invalid box.
    EXPECT_THROW(static_cast<void>(Box::builder()
                                       .with_lower_corner(Float3(1.0f, 1.0f, 1.0f))
                                       .with_upper_corner(Float3(0.0f, 2.0f, 2.0f))
                                       .build()),
                 std::runtime_error);
}

TEST(Box, BuilderRejectsCollapsedBox) {
    // A box flat on a single axis encloses no volume and is rejected.
    EXPECT_THROW(static_cast<void>(Box::builder()
                                       .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
                                       .with_upper_corner(Float3(1.0f, 1.0f, 0.0f))
                                       .build()),
                 std::runtime_error);
}

TEST(Box, BuilderMakeHostSharedBuildsBox) {
    const auto box = Box::builder()
                         .with_lower_corner(Float3(-2.0f, -2.0f, -2.0f))
                         .with_upper_corner(Float3(2.0f, 2.0f, 2.0f))
                         .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(box));
    expect_vec_near(box->lower_corner, Float3(-2.0f, -2.0f, -2.0f));
    expect_vec_near(box->upper_corner, Float3(2.0f, 2.0f, 2.0f));
}
