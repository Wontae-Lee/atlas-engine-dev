#include <atlas/geometry/square.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Ray;
using atlas::Square;

/** Per-component approximate comparison for Float3. */
void
expect_vec_near(const Float3& value, const Float3& expected, const float eps = 1.0e-6f) {
    EXPECT_NEAR(value.x, expected.x, eps);
    EXPECT_NEAR(value.y, expected.y, eps);
    EXPECT_NEAR(value.z, expected.z, eps);
}

/**
 * The default square (normal +z) resolves through orthonormal_basis to the
 * in-plane axes tangent = +x, bitangent = +y, so it spans x,y in [-0.5, 0.5]
 * at z = 0.
 */
Square
unit_square() {
    return Square(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f), 1.0f);
}

}

TEST(Square, DefaultIsAUnitPatchInTheZPlaneFacingPlusZ) {
    const Square square;

    expect_vec_near(square.center, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(square.normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(square.side_length, 1.0f);
    EXPECT_TRUE(square.is_valid());
}

TEST(Square, IsValidRejectsNonPositiveSideLength) {
    EXPECT_FALSE(Square(Float3(0.0f), Float3(0.0f, 0.0f, 1.0f), 0.0f).is_valid());
    EXPECT_FALSE(Square(Float3(0.0f), Float3(0.0f, 0.0f, 1.0f), -2.0f).is_valid());
}

TEST(Square, IsValidRejectsAZeroNormal) {
    EXPECT_FALSE(Square(Float3(0.0f), Float3(0.0f, 0.0f, 0.0f), 1.0f).is_valid());
}

TEST(Square, IsValidRejectsNonFiniteParameters) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(Square(Float3(0.0f), Float3(0.0f, 0.0f, 1.0f), inf).is_valid());
}

TEST(Square, ClosestNormalIsTheUnitFaceNormal) {
    const Square square = unit_square();
    const Float3 n      = square.closest_normal(Float3(3.0f, -2.0f, 5.0f));

    expect_vec_near(n, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(n.length(), 1.0f, 1.0e-6f);
}

TEST(Square, ClosestPointProjectsOntoThePatch) {
    const Square square = unit_square();

    // Directly above the center projects to the center.
    expect_vec_near(square.closest_point(Float3(0.0f, 0.0f, 3.0f)), Float3(0.0f, 0.0f, 0.0f));

    // Above an interior point projects straight down.
    expect_vec_near(square.closest_point(Float3(0.25f, 0.25f, 1.0f)), Float3(0.25f, 0.25f, 0.0f));
}

TEST(Square, ClosestPointClampsToTheExtent) {
    const Square square = unit_square();

    // Outside a face along +x.
    expect_vec_near(square.closest_point(Float3(2.0f, 0.0f, 1.0f)), Float3(0.5f, 0.0f, 0.0f));

    // Near an edge.
    expect_vec_near(square.closest_point(Float3(2.0f, 0.1f, 0.0f)), Float3(0.5f, 0.1f, 0.0f));

    // Near a corner.
    expect_vec_near(square.closest_point(Float3(2.0f, 2.0f, 0.0f)), Float3(0.5f, 0.5f, 0.0f));
}

TEST(Square, SignedDistanceIsPositiveOnTheNormalSide) {
    const Square square = unit_square();
    EXPECT_FLOAT_EQ(square.signed_distance(Float3(0.0f, 0.0f, 1.0f)), 1.0f);
}

TEST(Square, SignedDistanceIsNegativeOnTheFarSide) {
    const Square square = unit_square();
    EXPECT_FLOAT_EQ(square.signed_distance(Float3(0.0f, 0.0f, -1.0f)), -1.0f);
}

TEST(Square, SignedDistanceMeasuresToTheClampedPoint) {
    const Square square = unit_square();
    // In-plane but outside the extent: nearest point is the +x edge midpoint.
    EXPECT_FLOAT_EQ(square.signed_distance(Float3(2.0f, 0.0f, 0.0f)), 1.5f);
}

TEST(Square, IsInsideIsASlabTestWithTolerance) {
    const Square square = unit_square();

    EXPECT_TRUE(square.is_inside(Float3(0.4f, 0.4f, 0.0f)));

    // Just off the plane.
    EXPECT_TRUE(square.is_inside(Float3(0.0f, 0.0f, 0.05f), 0.1f));
    EXPECT_FALSE(square.is_inside(Float3(0.0f, 0.0f, 0.05f), 0.01f));
}

TEST(Square, IsOnSurfaceMatchesIsInside) {
    const Square square = unit_square();

    // Just beyond the in-plane extent.
    EXPECT_TRUE(square.is_on_surface(Float3(0.55f, 0.0f, 0.0f), 0.1f));
    EXPECT_FALSE(square.is_on_surface(Float3(0.55f, 0.0f, 0.0f), 0.01f));

    EXPECT_EQ(square.is_on_surface(Float3(0.4f, 0.4f, 0.02f), 0.05f),
              square.is_inside(Float3(0.4f, 0.4f, 0.02f), 0.05f));
}

TEST(Square, IsInsideIsFalseForADegeneratePatch) {
    const Square degenerate(Float3(0.0f), Float3(0.0f, 0.0f, 1.0f), 0.0f);
    EXPECT_FALSE(degenerate.is_inside(Float3(0.0f, 0.0f, 0.0f)));
}

TEST(Square, CentroidIsTheCenter) {
    const Square square(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);
    expect_vec_near(square.centroid(), Float3(1.0f, 2.0f, 3.0f));
}

TEST(Square, BoundOfAnAxisAlignedPatchIsFlatInZ) {
    const Square square = unit_square();
    const AABB bound    = square.bound();

    expect_vec_near(bound.lower_corner, Float3(-0.5f, -0.5f, 0.0f));
    expect_vec_near(bound.upper_corner, Float3(0.5f, 0.5f, 0.0f));
}

TEST(Square, BoundOfATiltedPatchIsTheRotatedBox) {
    // Normal +x: basis resolves to tangent = +y, bitangent = +z, so the side-2
    // patch spans y,z in [-1, 1] at x = 0 and is flat in x.
    const Square square(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f), 2.0f);
    const AABB bound = square.bound();

    expect_vec_near(bound.lower_corner, Float3(0.0f, -1.0f, -1.0f));
    expect_vec_near(bound.upper_corner, Float3(0.0f, 1.0f, 1.0f));
}

TEST(Square, SignedDistanceOfATiltedPatchUsesTheNormalDirection) {
    const Square square(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f), 2.0f);
    // One unit along +x (the normal side) from the patch.
    EXPECT_FLOAT_EQ(square.signed_distance(Float3(1.0f, 0.0f, 0.0f)), 1.0f);
}

TEST(Square, TraceHitsThePatchFromTheFront) {
    const Square square = unit_square();
    const Ray ray(Float3(0.0f, 0.0f, 3.0f), Float3(0.0f, 0.0f, -1.0f));

    const HitSurface hit = square.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 3.0f, 1.0e-6f);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(Square, TraceMissesOutsideThePatchExtent) {
    const Square square = unit_square();
    const Ray ray(Float3(2.0f, 0.0f, 3.0f), Float3(0.0f, 0.0f, -1.0f));

    EXPECT_FALSE(square.trace(ray).is_intersecting);
}

TEST(Square, TraceMissesWhenParallelToThePlane) {
    const Square square = unit_square();
    const Ray ray(Float3(0.0f, 0.0f, 3.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_FALSE(square.trace(ray).is_intersecting);
}

TEST(Square, TraceMissesWhenPointingAway) {
    const Square square = unit_square();
    // Behind the patch, travelling further away.
    const Ray ray(Float3(0.0f, 0.0f, 3.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_FALSE(square.trace(ray).is_intersecting);
}

TEST(Square, BuilderBuildsValidatedSquare) {
    const Square square = Square::builder()
                              .with_center(Float3(1.0f, 2.0f, 3.0f))
                              .with_normal(Float3(0.0f, 1.0f, 0.0f))
                              .with_side_length(4.0f)
                              .build();

    expect_vec_near(square.center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(square.normal, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_FLOAT_EQ(square.side_length, 4.0f);
    EXPECT_TRUE(square.is_valid());
}

TEST(Square, BuilderRejectsNonPositiveSideLength) {
    EXPECT_THROW(static_cast<void>(Square::builder().with_side_length(0.0f).build()),
                 std::runtime_error);
    EXPECT_THROW(static_cast<void>(Square::builder().with_side_length(-1.0f).build()),
                 std::runtime_error);
}

TEST(Square, BuilderRejectsZeroNormal) {
    EXPECT_THROW(static_cast<void>(Square::builder().with_normal(Float3(0.0f, 0.0f, 0.0f)).build()),
                 std::runtime_error);
}

TEST(Square, BuilderMakeHostSharedBuildsSquare) {
    const auto square = Square::builder()
                            .with_center(Float3(0.0f, 0.0f, 5.0f))
                            .with_side_length(2.0f)
                            .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(square));
    expect_vec_near(square->center, Float3(0.0f, 0.0f, 5.0f));
    EXPECT_FLOAT_EQ(square->side_length, 2.0f);
}
