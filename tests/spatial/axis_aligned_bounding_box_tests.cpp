#include "../utilities/test_utils.h"

#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <testkit/testkit.h>

namespace {

using atlas::AABBF;
using atlas::RayF;
using atlas::Vector3F;
using atlas::make_aabb;
using atlas::merge_aabb;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(AxisAlignedBoundingBox, DefaultConstructorCreatesEmptyInvalidBox) {
    // Arrange: create a default bounding box.
    const AABBF box;

    // Assert: default construction represents an invalid empty box.
    EXPECT_FALSE(box.is_valid());
    EXPECT_GT(box.lower_corner.x, box.upper_corner.x);
    EXPECT_GT(box.lower_corner.y, box.upper_corner.y);
    EXPECT_GT(box.lower_corner.z, box.upper_corner.z);
}

TEST(AxisAlignedBoundingBox, TwoPointConstructorOrdersCorners) {
    // Arrange and act: construct a box from unordered corner points.
    const AABBF box(Vector3F(3, -1, 5), Vector3F(-2, 4, 1));

    // Assert: constructor canonicalizes lower and upper corners.
    EXPECT_TRUE(box.is_valid());
    EXPECT_TRUE(vec_near(box.lower_corner, Vector3F(-2, -1, 1), tol));
    EXPECT_TRUE(vec_near(box.upper_corner, Vector3F(3, 4, 5), tol));
}

TEST(AxisAlignedBoundingBox, SizeQueriesMatchCorners) {
    // Arrange: create a valid box with equal edge lengths.
    const AABBF box(Vector3F(-1, -2, -3), Vector3F(3, 2, 1));

    // Assert: scalar size queries match the corner deltas.
    EXPECT_NEAR(box.width(), 4.0f, tol);
    EXPECT_NEAR(box.height(), 4.0f, tol);
    EXPECT_NEAR(box.depth(), 4.0f, tol);
    EXPECT_NEAR(box.length(0), 4.0f, tol);
    EXPECT_NEAR(box.length(1), 4.0f, tol);
    EXPECT_NEAR(box.length(2), 4.0f, tol);
    EXPECT_NEAR(box.area(), 96.0f, tol);
}

TEST(AxisAlignedBoundingBox, OverlapAndContainmentQueriesWork) {
    // Arrange: create one base box, one overlapping box, and one separate box.
    const AABBF box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));
    const AABBF overlap(Vector3F(0, 0, 0), Vector3F(2, 2, 2));
    const AABBF separate(Vector3F(3, 3, 3), Vector3F(4, 4, 4));

    // Assert: overlap queries distinguish intersecting and separate boxes.
    EXPECT_TRUE(box.overlaps(overlap));
    EXPECT_FALSE(box.overlaps(separate));

    // Assert: containment includes boundary points and rejects exterior points.
    EXPECT_TRUE(box.contains(Vector3F(0, 0, 0)));
    EXPECT_TRUE(box.contains(Vector3F(1, 1, 1)));
    EXPECT_FALSE(box.contains(Vector3F(2, 0, 0)));
}

TEST(AxisAlignedBoundingBox, TraceAndIntersectsWorkForRay) {
    // Arrange: create a unit box and rays that hit and miss it.
    const AABBF box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));
    const RayF hit_ray(Vector3F(-3, 0, 0), Vector3F(1, 0, 0));
    const RayF miss_ray(Vector3F(-3, 3, 0), Vector3F(1, 0, 0));

    // Assert: broad ray intersection checks agree with expected hit state.
    EXPECT_TRUE(box.intersects(hit_ray));
    EXPECT_FALSE(box.intersects(miss_ray));

    // Act: trace the ray through the box.
    const auto hit = box.trace(hit_ray);

    // Assert: trace returns the entry and exit distances.
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.enter, 2.0f, tol);
    EXPECT_NEAR(hit.exit, 4.0f, tol);
}

TEST(AxisAlignedBoundingBox, CenterExtentsAndDiagonalQueriesWork) {
    // Arrange: create a valid box with known center and extents.
    const AABBF box(Vector3F(-1, -2, -3), Vector3F(3, 2, 1));

    // Assert: vector and scalar derived queries match the box geometry.
    EXPECT_TRUE(vec_near(box.center(), Vector3F(1, 0, -1), tol));
    EXPECT_TRUE(vec_near(box.extents(), Vector3F(4, 4, 4), tol));
    EXPECT_NEAR(box.diagonal_length_squared(), 48.0f, tol);
    EXPECT_NEAR(box.diagonal_length(), std::sqrt(48.0f), tol);
}

TEST(AxisAlignedBoundingBox, ResetMergeExpandAndCornerWork) {
    // Arrange: start from an invalid empty box.
    AABBF box;

    // Act: merge points into the box.
    box.merge(Vector3F(1, 2, 3));
    box.merge(Vector3F(-1, -2, -3));

    // Assert: point merging creates the expected bounds.
    EXPECT_TRUE(box.is_valid());
    EXPECT_TRUE(vec_near(box.lower_corner, Vector3F(-1, -2, -3), tol));
    EXPECT_TRUE(vec_near(box.upper_corner, Vector3F(1, 2, 3), tol));

    // Act: expand the box uniformly.
    box.expand(1.0f);

    // Assert: expansion and corner lookup match the expected bounds.
    EXPECT_TRUE(vec_near(box.lower_corner, Vector3F(-2, -3, -4), tol));
    EXPECT_TRUE(vec_near(box.upper_corner, Vector3F(2, 3, 4), tol));
    EXPECT_TRUE(vec_near(box.corner(0), Vector3F(-2, -3, -4), tol));
    EXPECT_TRUE(vec_near(box.corner(7), Vector3F(2, 3, 4), tol));

    // Act and assert: reset returns the box to the invalid empty state.
    box.reset();
    EXPECT_FALSE(box.is_valid());
}

TEST(AxisAlignedBoundingBox, FreeHelpersMakeAndMergeAabbWork) {
    // Arrange: create degenerate boxes through helper functions.
    auto a = make_aabb(Vector3F(-1, -1, -1));
    auto b = make_aabb(Vector3F(3, 4, 5));

    // Act: expand each box and merge them into a union.
    a.merge(Vector3F(1, 1, 1));
    b.merge(Vector3F(0, 2, 0));

    const auto merged = merge_aabb(a, b);

    // Assert: helpers produce valid boxes and the expected merged bounds.
    EXPECT_TRUE(a.is_valid());
    EXPECT_TRUE(b.is_valid());
    EXPECT_TRUE(merged.is_valid());
    EXPECT_TRUE(vec_near(merged.lower_corner, Vector3F(-1, -1, -1), tol));
    EXPECT_TRUE(vec_near(merged.upper_corner, Vector3F(3, 4, 5), tol));
}
