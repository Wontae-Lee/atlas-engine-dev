#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

namespace {

using atlas::AABB;
using atlas::Float3;

// Compare two vectors component-wise with float tolerance; operator== on Float3
// is exact and unsuitable for arithmetic results.
void
expect_float3_eq(const Float3& actual, const Float3& expected) {
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
}

// A box whose corners are deliberately inverted, so size queries go negative.
AABB
inverted_box() {
    AABB box;
    box.lower_corner = Float3(5.0f, 5.0f, 5.0f);
    box.upper_corner = Float3(2.0f, 2.0f, 2.0f);
    return box;
}

}

TEST(AABB, DefaultConstructsCanonicalEmptyBox) {
    const AABB box;

    // Empty box seeds lower = +inf, upper = -inf so the first merge snaps on.
    EXPECT_TRUE(std::isinf(box.lower_corner.x));
    EXPECT_GT(box.lower_corner.x, 0.0f);
    EXPECT_TRUE(std::isinf(box.upper_corner.x));
    EXPECT_LT(box.upper_corner.x, 0.0f);

    EXPECT_FALSE(box.is_valid());
    EXPECT_TRUE(box.is_empty());
}

TEST(AABB, TwoPointConstructorSortsCornersRegardlessOfOrder) {
    const AABB box(Float3(2.0f, -1.0f, 4.0f), Float3(-3.0f, 5.0f, 0.0f));

    expect_float3_eq(box.lower_corner, Float3(-3.0f, -1.0f, 0.0f));
    expect_float3_eq(box.upper_corner, Float3(2.0f, 5.0f, 4.0f));
}

TEST(AABB, MakeAabbCollapsesOntoAPoint) {
    const Float3 p(1.0f, 2.0f, 3.0f);
    const AABB   box = atlas::make_aabb(p);

    expect_float3_eq(box.lower_corner, p);
    expect_float3_eq(box.upper_corner, p);
    // A point box has zero volume, so it is empty but still finite/valid.
    EXPECT_TRUE(box.is_empty());
    EXPECT_TRUE(box.is_valid());
}

TEST(AABB, ExtentsAndSideLengths) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 3.0f, 4.0f));

    EXPECT_FLOAT_EQ(box.width(), 2.0f);
    EXPECT_FLOAT_EQ(box.height(), 3.0f);
    EXPECT_FLOAT_EQ(box.depth(), 4.0f);
    EXPECT_FLOAT_EQ(box.length(0), 2.0f);
    EXPECT_FLOAT_EQ(box.length(1), 3.0f);
    EXPECT_FLOAT_EQ(box.length(2), 4.0f);
    expect_float3_eq(box.extents(), Float3(2.0f, 3.0f, 4.0f));
}

TEST(AABB, WidthIsNegativeOnAnInvertedBox) {
    EXPECT_FLOAT_EQ(inverted_box().width(), -3.0f);
}

TEST(AABB, SurfaceAreaMatchesTheSahMetric) {
    const AABB unit(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));
    EXPECT_FLOAT_EQ(unit.area(), 6.0f);

    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 3.0f, 4.0f));
    // 2 * (w*h + w*d + h*d) = 2 * (6 + 8 + 12) = 52.
    EXPECT_FLOAT_EQ(box.area(), 52.0f);
}

TEST(AABB, CenterIsTheMidpointOfTheCorners) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 4.0f, 6.0f));
    expect_float3_eq(box.center(), Float3(1.0f, 2.0f, 3.0f));
}

TEST(AABB, DiagonalLength) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 3.0f, 6.0f));

    EXPECT_FLOAT_EQ(box.diagonal_length_squared(), 49.0f);
    EXPECT_FLOAT_EQ(box.diagonal_length(), 7.0f);
}

TEST(AABB, MergePointGrowsTheBound) {
    AABB box;
    box.merge(Float3(1.0f, 2.0f, 3.0f));
    box.merge(Float3(-1.0f, 5.0f, 0.0f));

    expect_float3_eq(box.lower_corner, Float3(-1.0f, 2.0f, 0.0f));
    expect_float3_eq(box.upper_corner, Float3(1.0f, 5.0f, 3.0f));
}

TEST(AABB, MergeBoxProducesTheUnionBound) {
    AABB       box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));
    const AABB other(Float3(2.0f, 2.0f, 2.0f), Float3(3.0f, 3.0f, 3.0f));

    box.merge(other);

    expect_float3_eq(box.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_eq(box.upper_corner, Float3(3.0f, 3.0f, 3.0f));
}

TEST(AABB, MergeAabbFreeFunctionDoesNotMutateInputs) {
    const AABB a(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));
    const AABB b(Float3(2.0f, 2.0f, 2.0f), Float3(3.0f, 3.0f, 3.0f));

    const AABB u = atlas::merge_aabb(a, b);

    expect_float3_eq(u.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_eq(u.upper_corner, Float3(3.0f, 3.0f, 3.0f));
    // Inputs are untouched.
    expect_float3_eq(a.upper_corner, Float3(1.0f, 1.0f, 1.0f));
    expect_float3_eq(b.lower_corner, Float3(2.0f, 2.0f, 2.0f));
}

TEST(AABB, ResetReturnsToTheEmptyBox) {
    AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));
    box.reset();

    EXPECT_FALSE(box.is_valid());
    EXPECT_TRUE(box.is_empty());
}

TEST(AABB, ExpandInflatesEveryFaceAndNegativeDeltaShrinks) {
    AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f));

    box.expand(0.5f);
    expect_float3_eq(box.lower_corner, Float3(-0.5f, -0.5f, -0.5f));
    expect_float3_eq(box.upper_corner, Float3(2.5f, 2.5f, 2.5f));

    box.expand(-0.5f);
    expect_float3_eq(box.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_eq(box.upper_corner, Float3(2.0f, 2.0f, 2.0f));
}

TEST(AABB, ContainsIsBoundaryInclusive) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f));

    EXPECT_TRUE(box.contains(Float3(1.0f, 1.0f, 1.0f)));
    EXPECT_TRUE(box.contains(Float3(0.0f, 1.0f, 2.0f)));  // on the boundary
    EXPECT_FALSE(box.contains(Float3(3.0f, 1.0f, 1.0f)));
    EXPECT_FALSE(box.contains(Float3(-0.001f, 1.0f, 1.0f)));
}

TEST(AABB, OverlapsCountsTouchingFacesAndRejectsSeparation) {
    const AABB a(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f));

    EXPECT_TRUE(a.overlaps(AABB(Float3(1.0f, 1.0f, 1.0f), Float3(3.0f, 3.0f, 3.0f))));
    // Faces touch exactly at x = 2; strict separation test counts this as overlap.
    EXPECT_TRUE(a.overlaps(AABB(Float3(2.0f, 0.0f, 0.0f), Float3(4.0f, 2.0f, 2.0f))));
    EXPECT_FALSE(a.overlaps(AABB(Float3(3.0f, 3.0f, 3.0f), Float3(4.0f, 4.0f, 4.0f))));
}

TEST(AABB, CornerSelectsByBitIndex) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 2.0f, 3.0f));

    expect_float3_eq(box.corner(0), Float3(0.0f, 0.0f, 0.0f));  // all lower
    expect_float3_eq(box.corner(1), Float3(1.0f, 0.0f, 0.0f));  // upper x
    expect_float3_eq(box.corner(2), Float3(0.0f, 2.0f, 0.0f));  // upper y
    expect_float3_eq(box.corner(4), Float3(0.0f, 0.0f, 3.0f));  // upper z
    expect_float3_eq(box.corner(7), Float3(1.0f, 2.0f, 3.0f));  // all upper
}

TEST(AABB, ClampProjectsOntoTheBox) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));

    expect_float3_eq(box.clamp(Float3(3.0f, -1.0f, 0.5f)), Float3(1.0f, 0.0f, 0.5f));
    // A point already inside is left untouched.
    expect_float3_eq(box.clamp(Float3(0.25f, 0.75f, 0.5f)), Float3(0.25f, 0.75f, 0.5f));
}

TEST(AABB, DistanceSquaredIsZeroInsideAndPositiveOutside) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));

    EXPECT_FLOAT_EQ(atlas::aabb_distance_squared(box, Float3(0.5f, 0.5f, 0.5f)), 0.0f);
    EXPECT_FLOAT_EQ(atlas::aabb_distance_squared(box, Float3(3.0f, 0.0f, 0.0f)), 4.0f);
}

TEST(AABB, IsEmptyTreatsAFlatSlabAsEmptyButStillValid) {
    // Zero extent along z: valid (finite, lower <= upper) yet empty (no volume).
    const AABB slab(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 0.0f));

    EXPECT_TRUE(slab.is_valid());
    EXPECT_TRUE(slab.is_empty());
}

TEST(AABB, IsValidAndIsEmptyOnAProperBox) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 2.0f, 3.0f));

    EXPECT_TRUE(box.is_valid());
    EXPECT_FALSE(box.is_empty());
}

TEST(AABB, IsValidRejectsNonFiniteCorners) {
    AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f));
    box.upper_corner = Float3(atlas::inf, 1.0f, 1.0f);

    EXPECT_FALSE(box.is_valid());
}

TEST(AABB, TransformAabbIdentityPreservesTheBox) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 2.0f, 3.0f));

    const AABB out = atlas::transform_aabb(box, [](const Float3& p) { return p; });

    expect_float3_eq(out.lower_corner, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_eq(out.upper_corner, Float3(1.0f, 2.0f, 3.0f));
}

TEST(AABB, TransformAabbTranslatesEveryCorner) {
    const AABB box(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 2.0f, 3.0f));

    const AABB out = atlas::transform_aabb(box, [](const Float3& p) {
        return p + Float3(10.0f, 0.0f, 0.0f);
    });

    expect_float3_eq(out.lower_corner, Float3(10.0f, 0.0f, 0.0f));
    expect_float3_eq(out.upper_corner, Float3(11.0f, 2.0f, 3.0f));
}

TEST(AABB, TransformAabbOnAnInvalidBoxYieldsTheEmptyBox) {
    const AABB empty;  // default, non-finite corners => not valid

    const AABB out = atlas::transform_aabb(empty, [](const Float3& p) { return p; });

    EXPECT_FALSE(out.is_valid());
    EXPECT_TRUE(out.is_empty());
}
