#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

TEST(AxisAlignedBoundingBox, DefaultConstructorIsEmptyAndResetExtremes) {

    atlas::AxisAlignedBoundingBox<double> aabb;

    EXPECT_TRUE(aabb.is_empty());

    constexpr double M = std::numeric_limits<double>::max();

    EXPECT_DOUBLE_EQ(aabb.lower_corner[0], M);
    EXPECT_DOUBLE_EQ(aabb.lower_corner[1], M);
    EXPECT_DOUBLE_EQ(aabb.lower_corner[2], M);

    EXPECT_DOUBLE_EQ(aabb.upper_corner[0], -M);
    EXPECT_DOUBLE_EQ(aabb.upper_corner[1], -M);
    EXPECT_DOUBLE_EQ(aabb.upper_corner[2], -M);
}

TEST(AxisAlignedBoundingBox, TwoPointConstructorOrdersCorners) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> p1(2.0, -1.0, 5.0);
    const atlas::math::Vector<double, 3> p2(-3.0, 4.0, 1.0);

    const atlas::AxisAlignedBoundingBox<double> aabb(p1, p2);

    EXPECT_TRUE(atlas::test::vec_near(aabb.lower_corner,
                                      atlas::math::Vector<double, 3>(-3.0, -1.0, 1.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.upper_corner,
                                      atlas::math::Vector<double, 3>(2.0, 4.0, 5.0),
                                      eps));
    EXPECT_FALSE(aabb.is_empty());
}

TEST(AxisAlignedBoundingBox, CopyConstructorCopiesCorners) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> a(
        atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0),
        atlas::math::Vector<double, 3>(4.0, 5.0, 6.0));

    const atlas::AxisAlignedBoundingBox<double>& b(a);

    EXPECT_TRUE(atlas::test::vec_near(b.lower_corner, a.lower_corner, eps));
    EXPECT_TRUE(atlas::test::vec_near(b.upper_corner, a.upper_corner, eps));
}

TEST(AxisAlignedBoundingBox, WidthHeightDepthLength) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, 2.0, 10.0),
        atlas::math::Vector<double, 3>(3.0, 7.0, 13.0));

    EXPECT_TRUE(atlas::test::near(aabb.width(), 4.0, eps));
    EXPECT_TRUE(atlas::test::near(aabb.height(), 5.0, eps));
    EXPECT_TRUE(atlas::test::near(aabb.depth(), 3.0, eps));

    EXPECT_TRUE(atlas::test::near(aabb.length(0), 4.0, eps));
    EXPECT_TRUE(atlas::test::near(aabb.length(1), 5.0, eps));
    EXPECT_TRUE(atlas::test::near(aabb.length(2), 3.0, eps));
}

TEST(AxisAlignedBoundingBox, AreaMatchesFormula) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, 2.0, 10.0),
        atlas::math::Vector<double, 3>(3.0, 7.0, 13.0));

    EXPECT_TRUE(atlas::test::near(aabb.area(), 94.0, eps));
}

TEST(AxisAlignedBoundingBox, ContainsIsInclusiveOnBoundaries) {
    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0),
        atlas::math::Vector<double, 3>(4.0, 5.0, 6.0));

    EXPECT_TRUE(aabb.contains(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0)));
    EXPECT_TRUE(aabb.contains(atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0)));
    EXPECT_TRUE(aabb.contains(atlas::math::Vector<double, 3>(4.0, 5.0, 6.0)));

    EXPECT_FALSE(aabb.contains(atlas::math::Vector<double, 3>(-1.0 - 1e-12, 0.0, 0.0)));
    EXPECT_FALSE(aabb.contains(atlas::math::Vector<double, 3>(0.0, 5.0 + 1e-12, 0.0)));
    EXPECT_FALSE(aabb.contains(atlas::math::Vector<double, 3>(0.0, 0.0, 6.0 + 1e-12)));
}

TEST(AxisAlignedBoundingBox, OverlapsBasicCases) {
    const atlas::AxisAlignedBoundingBox<double> a(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::AxisAlignedBoundingBox<double> b(
        atlas::math::Vector<double, 3>(0.5, 0.5, 0.5),
        atlas::math::Vector<double, 3>(2.0, 2.0, 2.0));

    const atlas::AxisAlignedBoundingBox<double> c(
        atlas::math::Vector<double, 3>(2.0, 2.0, 2.0),
        atlas::math::Vector<double, 3>(3.0, 3.0, 3.0));

    EXPECT_TRUE(a.overlaps(b));
    EXPECT_TRUE(b.overlaps(a));

    EXPECT_FALSE(a.overlaps(c));
    EXPECT_FALSE(c.overlaps(a));
}

TEST(AxisAlignedBoundingBox, CenterAndExtents) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, 2.0, 10.0),
        atlas::math::Vector<double, 3>(3.0, 7.0, 13.0));

    EXPECT_TRUE(atlas::test::vec_near(aabb.center(),
                                      atlas::math::Vector<double, 3>(1.0, 4.5, 11.5),
                                      eps));

    EXPECT_TRUE(atlas::test::vec_near(aabb.extents(),
                                      atlas::math::Vector<double, 3>(4.0, 5.0, 3.0),
                                      eps));
}

TEST(AxisAlignedBoundingBox, DiagonalLengths) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, 2.0, 10.0),
        atlas::math::Vector<double, 3>(3.0, 7.0, 13.0));

    constexpr double diag2 = 4.0 * 4.0 + 5.0 * 5.0 + 3.0 * 3.0;
    EXPECT_TRUE(atlas::test::near(aabb.diagonal_length_squared(), diag2, eps));
    EXPECT_TRUE(atlas::test::near(aabb.diagonal_length(), std::sqrt(diag2), eps));
}

TEST(AxisAlignedBoundingBox, MergePointInitializesFromEmpty) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::AxisAlignedBoundingBox<double> aabb;
    EXPECT_TRUE(aabb.is_empty());

    aabb.merge(atlas::math::Vector<double, 3>(2.0, -1.0, 5.0));
    EXPECT_TRUE(aabb.is_empty());

    EXPECT_TRUE(atlas::test::vec_near(aabb.lower_corner,
                                      atlas::math::Vector<double, 3>(2.0, -1.0, 5.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.upper_corner,
                                      atlas::math::Vector<double, 3>(2.0, -1.0, 5.0),
                                      eps));

    aabb.merge(atlas::math::Vector<double, 3>(-3.0, 4.0, 1.0));

    EXPECT_TRUE(atlas::test::vec_near(aabb.lower_corner,
                                      atlas::math::Vector<double, 3>(-3.0, -1.0, 1.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.upper_corner,
                                      atlas::math::Vector<double, 3>(2.0, 4.0, 5.0),
                                      eps));
}

TEST(AxisAlignedBoundingBox, MergeBoxUnionsBounds) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::AxisAlignedBoundingBox<double> a(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 2.0, 3.0));

    const atlas::AxisAlignedBoundingBox<double> b(
        atlas::math::Vector<double, 3>(-2.0, 1.0, 2.0),
        atlas::math::Vector<double, 3>(0.5, 5.0, 10.0));

    a.merge(b);

    EXPECT_TRUE(atlas::test::vec_near(a.lower_corner,
                                      atlas::math::Vector<double, 3>(-2.0, 0.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(a.upper_corner,
                                      atlas::math::Vector<double, 3>(1.0, 5.0, 10.0),
                                      eps));
}

TEST(AxisAlignedBoundingBox, ExpandGrowsUniformly) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 1.0, 2.0),
        atlas::math::Vector<double, 3>(3.0, 4.0, 5.0));

    aabb.expand(2.0);

    EXPECT_TRUE(atlas::test::vec_near(aabb.lower_corner,
                                      atlas::math::Vector<double, 3>(-2.0, -1.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.upper_corner,
                                      atlas::math::Vector<double, 3>(5.0, 6.0, 7.0),
                                      eps));
}

TEST(AxisAlignedBoundingBox, CornerIndexingReturnsExpectedCorners) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0),
        atlas::math::Vector<double, 3>(4.0, 5.0, 6.0));

    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(0), atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(1), atlas::math::Vector<double, 3>(4.0, -2.0, -3.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(2), atlas::math::Vector<double, 3>(-1.0, 5.0, -3.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(3), atlas::math::Vector<double, 3>(4.0, 5.0, -3.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(4), atlas::math::Vector<double, 3>(-1.0, -2.0, 6.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(5), atlas::math::Vector<double, 3>(4.0, -2.0, 6.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(6), atlas::math::Vector<double, 3>(-1.0, 5.0, 6.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(aabb.corner(7), atlas::math::Vector<double, 3>(4.0, 5.0, 6.0), eps));
}

TEST(AxisAlignedBoundingBox, ClampClipsPointToBounds) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(-1.0, -2.0, -3.0),
        atlas::math::Vector<double, 3>(4.0, 5.0, 6.0));

    const atlas::math::Vector<double, 3> p(-10.0, 3.0, 100.0);
    const atlas::math::Vector<double, 3> c = aabb.clamp(p);

    EXPECT_TRUE(atlas::test::vec_near(c,
                                      atlas::math::Vector<double, 3>(-1.0, 3.0, 6.0),
                                      eps));
}

TEST(AxisAlignedBoundingBox, RayIntersectsBasicHitAndMiss) {

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::Ray<double> hit(atlas::math::Vector<double, 3>(-1.0, 0.5, 0.5),
                                 atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));
    EXPECT_TRUE(aabb.intersects(hit));

    const atlas::Ray<double> miss(atlas::math::Vector<double, 3>(-1.0, 2.0, 0.5),
                                  atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));
    EXPECT_FALSE(aabb.intersects(miss));
}

TEST(AxisAlignedBoundingBox, RayParallelToAxisInsideSlabIntersects) {

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.5, -2.0, 0.5),
                               atlas::math::Vector<double, 3>(0.0, 1.0, 0.0));
    EXPECT_TRUE(aabb.intersects(r));
}

TEST(AxisAlignedBoundingBox, RayParallelToAxisOutsideSlabMisses) {
    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(2.0, -2.0, 0.5),
                               atlas::math::Vector<double, 3>(0.0, 1.0, 0.0));
    EXPECT_FALSE(aabb.intersects(r));
}

TEST(AxisAlignedBoundingBox, TraceReturnsEntryExitForSimpleHit) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-2.0, 0.5, 0.5),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto isect = aabb.trace(r);

    EXPECT_TRUE(isect.is_intersecting);
    EXPECT_TRUE(atlas::test::near(isect.enter, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(isect.exit, 3.0, eps));
}

TEST(AxisAlignedBoundingBox, TraceFromInsideForcesEnterZero) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.25, 0.25, 0.25),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto isect = aabb.trace(r);

    EXPECT_TRUE(isect.is_intersecting);
    EXPECT_TRUE(atlas::test::near(isect.enter, 0.0, eps));

    EXPECT_TRUE(atlas::test::near(isect.exit, 0.75, eps));
}

TEST(AxisAlignedBoundingBox, ResetRestoresEmptyState) {
    atlas::AxisAlignedBoundingBox<double> aabb(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 1.0, 1.0));

    EXPECT_FALSE(aabb.is_empty());

    aabb.reset();
    EXPECT_TRUE(aabb.is_empty());
}