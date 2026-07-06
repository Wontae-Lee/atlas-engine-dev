#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <cmath>
#include <gtest/gtest.h>

namespace {

using atlas::AABB;
using atlas::Ray;
using atlas::Float3;
using atlas::make_aabb;
using atlas::merge_aabb;
using atlas::transform_aabb;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(AABB, DefaultConstructorCreatesEmptyInvalidBox) {
    const AABB box;

    EXPECT_FALSE(box.is_valid());
    EXPECT_GT(box.lower_corner.x, box.upper_corner.x);
    EXPECT_GT(box.lower_corner.y, box.upper_corner.y);
    EXPECT_GT(box.lower_corner.z, box.upper_corner.z);
}

TEST(AABB, TwoPointConstructorOrdersCorners) {
    const AABB box(Float3(3.0f, -1.0f, 5.0f), Float3(-2.0f, 4.0f, 1.0f));

    EXPECT_TRUE(box.is_valid());
    expect_vec_near(box.lower_corner, Float3(-2.0f, -1.0f, 1.0f));
    expect_vec_near(box.upper_corner, Float3(3.0f, 4.0f, 5.0f));
}

TEST(AABB, SizeQueriesMatchCorners) {
    const AABB box(Float3(-1.0f, -2.0f, -3.0f), Float3(3.0f, 2.0f, 1.0f));

    EXPECT_NEAR(box.width(), 4.0f, tol);
    EXPECT_NEAR(box.height(), 4.0f, tol);
    EXPECT_NEAR(box.depth(), 4.0f, tol);
    EXPECT_NEAR(box.length(0), 4.0f, tol);
    EXPECT_NEAR(box.length(1), 4.0f, tol);
    EXPECT_NEAR(box.length(2), 4.0f, tol);
    EXPECT_NEAR(box.area(), 96.0f, tol);
}

TEST(AABB, OverlapAndContainmentQueriesWork) {
    const AABB box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));
    const AABB overlap(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f));
    const AABB separate(Float3(3.0f, 3.0f, 3.0f), Float3(4.0f, 4.0f, 4.0f));

    EXPECT_TRUE(box.overlaps(overlap));
    EXPECT_FALSE(box.overlaps(separate));

    EXPECT_TRUE(box.contains(Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(box.contains(Float3(1.0f, 1.0f, 1.0f)));
    EXPECT_FALSE(box.contains(Float3(2.0f, 0.0f, 0.0f)));
}

TEST(AABB, TraceAndIntersectsWorkForRay) {
    const AABB box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));
    const Ray hit_ray(Float3(-3.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));
    const Ray miss_ray(Float3(-3.0f, 3.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_TRUE(box.intersects(hit_ray));
    EXPECT_FALSE(box.intersects(miss_ray));

    const auto hit = box.trace(hit_ray);

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.enter, 2.0f, tol);
    EXPECT_NEAR(hit.exit, 4.0f, tol);
}

TEST(AABB, CenterExtentsAndDiagonalQueriesWork) {
    const AABB box(Float3(-1.0f, -2.0f, -3.0f), Float3(3.0f, 2.0f, 1.0f));

    expect_vec_near(box.center(), Float3(1.0f, 0.0f, -1.0f));
    expect_vec_near(box.extents(), Float3(4.0f, 4.0f, 4.0f));
    EXPECT_NEAR(box.diagonal_length_squared(), 48.0f, tol);
    EXPECT_NEAR(box.diagonal_length(), std::sqrt(48.0f), tol);
}

TEST(AABB, ResetMergeExpandAndCornerWork) {
    AABB box;

    box.merge(Float3(1.0f, 2.0f, 3.0f));
    box.merge(Float3(-1.0f, -2.0f, -3.0f));

    EXPECT_TRUE(box.is_valid());
    expect_vec_near(box.lower_corner, Float3(-1.0f, -2.0f, -3.0f));
    expect_vec_near(box.upper_corner, Float3(1.0f, 2.0f, 3.0f));

    box.expand(1.0f);

    expect_vec_near(box.lower_corner, Float3(-2.0f, -3.0f, -4.0f));
    expect_vec_near(box.upper_corner, Float3(2.0f, 3.0f, 4.0f));
    expect_vec_near(box.corner(0), Float3(-2.0f, -3.0f, -4.0f));
    expect_vec_near(box.corner(7), Float3(2.0f, 3.0f, 4.0f));

    box.reset();
    EXPECT_FALSE(box.is_valid());
}

TEST(AABB, FreeHelpersMakeAndMergeAabbWork) {
    auto a = make_aabb(Float3(-1.0f, -1.0f, -1.0f));
    auto b = make_aabb(Float3(3.0f, 4.0f, 5.0f));

    a.merge(Float3(1.0f, 1.0f, 1.0f));
    b.merge(Float3(0.0f, 2.0f, 0.0f));

    const auto merged = merge_aabb(a, b);

    EXPECT_TRUE(a.is_valid());
    EXPECT_TRUE(b.is_valid());
    EXPECT_TRUE(merged.is_valid());
    expect_vec_near(merged.lower_corner, Float3(-1.0f, -1.0f, -1.0f));
    expect_vec_near(merged.upper_corner, Float3(3.0f, 4.0f, 5.0f));
}

TEST(AABB, TransformAabbEnclosesTransformedCorners) {
    const AABB box(Float3(-1.0f, -2.0f, 0.0f), Float3(2.0f, 1.0f, 3.0f));

    const auto transformed = transform_aabb(
        box,
        [](const Float3& p) noexcept {
            return Float3(-p.x + 1.0f, p.y * 2.0f, p.z + 4.0f);
        });

    EXPECT_TRUE(transformed.is_valid());
    expect_vec_near(transformed.lower_corner, Float3(-1.0f, -4.0f, 4.0f));
    expect_vec_near(transformed.upper_corner, Float3(2.0f, 2.0f, 7.0f));
}

TEST(AABB, TransformAabbKeepsInvalidInputInvalid) {
    const AABB box;

    const auto transformed = transform_aabb(
        box,
        [](const Float3& p) noexcept {
            return p;
        });

    EXPECT_FALSE(transformed.is_valid());
}
