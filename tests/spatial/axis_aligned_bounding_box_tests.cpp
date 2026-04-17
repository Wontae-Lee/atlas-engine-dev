#include "../utilities/tests_utils.h"

#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Aabb = atlas::spatial::AxisAlignedBoundingBox<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(AxisAlignedBoundingBox, DefaultConstructorCreatesEmptyInvalidBox) {
    const Aabb box;

    EXPECT_FALSE(box.is_valid());
    EXPECT_GT(box.lower_corner.x, box.upper_corner.x);
    EXPECT_GT(box.lower_corner.y, box.upper_corner.y);
    EXPECT_GT(box.lower_corner.z, box.upper_corner.z);
}

TEST(AxisAlignedBoundingBox, TwoPointConstructorOrdersCorners) {
    const Aabb box(Vec3(3, -1, 5), Vec3(-2, 4, 1));

    EXPECT_TRUE(box.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(box.lower_corner, Vec3(-2, -1, 1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.upper_corner, Vec3(3, 4, 5), kEps));
}

TEST(AxisAlignedBoundingBox, SizeQueriesMatchCorners) {
    const Aabb box(Vec3(-1, -2, -3), Vec3(3, 2, 1));

    EXPECT_NEAR(box.width(), 4.0f, kEps);
    EXPECT_NEAR(box.height(), 4.0f, kEps);
    EXPECT_NEAR(box.depth(), 4.0f, kEps);
    EXPECT_NEAR(box.length(0), 4.0f, kEps);
    EXPECT_NEAR(box.length(1), 4.0f, kEps);
    EXPECT_NEAR(box.length(2), 4.0f, kEps);
    EXPECT_NEAR(box.area(), 96.0f, kEps);
}

TEST(AxisAlignedBoundingBox, OverlapAndContainmentQueriesWork) {
    const Aabb box(Vec3(-1, -1, -1), Vec3(1, 1, 1));
    const Aabb overlap(Vec3(0, 0, 0), Vec3(2, 2, 2));
    const Aabb separate(Vec3(3, 3, 3), Vec3(4, 4, 4));

    EXPECT_TRUE(box.overlaps(overlap));
    EXPECT_FALSE(box.overlaps(separate));

    EXPECT_TRUE(box.contains(Vec3(0, 0, 0)));
    EXPECT_TRUE(box.contains(Vec3(1, 1, 1)));
    EXPECT_FALSE(box.contains(Vec3(2, 0, 0)));
}

TEST(AxisAlignedBoundingBox, TraceAndIntersectsWorkForRay) {
    const Aabb box(Vec3(-1, -1, -1), Vec3(1, 1, 1));
    const atlas::spatial::Ray<T> hit_ray(Vec3(-3, 0, 0), Vec3(1, 0, 0));
    const atlas::spatial::Ray<T> miss_ray(Vec3(-3, 3, 0), Vec3(1, 0, 0));

    EXPECT_TRUE(box.intersects(hit_ray));
    EXPECT_FALSE(box.intersects(miss_ray));

    const auto hit = box.trace(hit_ray);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.enter, 2.0f, kEps);
    EXPECT_NEAR(hit.exit, 4.0f, kEps);
}

TEST(AxisAlignedBoundingBox, CenterExtentsAndDiagonalQueriesWork) {
    const Aabb box(Vec3(-1, -2, -3), Vec3(3, 2, 1));

    EXPECT_TRUE(atlas::test::vec_near(box.center(), Vec3(1, 0, -1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.extents(), Vec3(4, 4, 4), kEps));
    EXPECT_NEAR(box.diagonal_length_squared(), 48.0f, kEps);
    EXPECT_NEAR(box.diagonal_length(), std::sqrt(48.0f), kEps);
}

TEST(AxisAlignedBoundingBox, ResetMergeExpandAndCornerWork) {
    Aabb box;

    box.merge(Vec3(1, 2, 3));
    box.merge(Vec3(-1, -2, -3));

    EXPECT_TRUE(box.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(box.lower_corner, Vec3(-1, -2, -3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.upper_corner, Vec3(1, 2, 3), kEps));

    box.expand(1.0f);
    EXPECT_TRUE(atlas::test::vec_near(box.lower_corner, Vec3(-2, -3, -4), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.upper_corner, Vec3(2, 3, 4), kEps));

    EXPECT_TRUE(atlas::test::vec_near(box.corner(0), Vec3(-2, -3, -4), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.corner(7), Vec3(2, 3, 4), kEps));

    box.reset();
    EXPECT_FALSE(box.is_valid());
}

TEST(AxisAlignedBoundingBox, FreeHelpersMakeAndMergeAabbWork) {
    auto a = atlas::spatial::make_aabb(Vec3(-1, -1, -1));
    auto b = atlas::spatial::make_aabb(Vec3(3, 4, 5));

    a.merge(Vec3(1, 1, 1));
    b.merge(Vec3(0, 2, 0));

    const auto merged = atlas::spatial::merge_aabb(a, b);

    EXPECT_TRUE(a.is_valid());
    EXPECT_TRUE(b.is_valid());
    EXPECT_TRUE(merged.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(merged.lower_corner, Vec3(-1, -1, -1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(merged.upper_corner, Vec3(3, 4, 5), kEps));
}
