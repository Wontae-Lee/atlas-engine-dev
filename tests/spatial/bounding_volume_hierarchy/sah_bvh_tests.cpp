#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cstddef>
#include <gtest/gtest.h>

static ATLAS_FORCE_INLINE atlas::TriangleContainer4<double>
make_tri(const atlas::math::Vector<double, 3>& a,
         const atlas::math::Vector<double, 3>& b,
         const atlas::math::Vector<double, 3>& c,
         const atlas::math::Vector<double, 3>& n) {

    return atlas::TriangleContainer4<double>(a, b, c, n);
}

TEST(SAHBVH, BuildSingleTriangleCreatesSingleLeafRoot) {
    atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<double> bvh;

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.resize(1);
    tris[0] = make_tri(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 1.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    bvh.build(tris);

    EXPECT_EQ(bvh.root(), 0);

    ASSERT_EQ(bvh.nodes().size(), static_cast<std::size_t>(1));
    EXPECT_TRUE(bvh.nodes()[0].is_leaf);
    EXPECT_EQ(bvh.nodes()[0].left, -1);
    EXPECT_EQ(bvh.nodes()[0].right, -1);
    EXPECT_EQ(bvh.nodes()[0].count, 1);

    ASSERT_EQ(bvh.indices().size(), static_cast<std::size_t>(1));
    EXPECT_EQ(bvh.indices()[0], 0);
}

TEST(SAHBVH, BuildTwoTrianglesCreatesInternalRootAndTwoLeaves) {
    atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<double> bvh;
    bvh.set_leaf_size(1);
    bvh.set_num_of_bins(8);

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.resize(2);

    tris[0] = make_tri(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 1.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    tris[1] = make_tri(
        atlas::math::Vector<double, 3>(10.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(11.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(10.0, 1.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    bvh.build(tris);

    const int root = bvh.root();
    EXPECT_GE(root, 0);

    ASSERT_GE(bvh.nodes().size(), static_cast<std::size_t>(3));

    const auto& nodes = bvh.nodes();
    ASSERT_LT(static_cast<std::size_t>(root), nodes.size());

    EXPECT_FALSE(nodes[root].is_leaf);
    EXPECT_GE(nodes[root].left, 0);
    EXPECT_GE(nodes[root].right, 0);

    ASSERT_LT(static_cast<std::size_t>(nodes[root].left), nodes.size());
    ASSERT_LT(static_cast<std::size_t>(nodes[root].right), nodes.size());

    EXPECT_TRUE(nodes[nodes[root].left].is_leaf);
    EXPECT_TRUE(nodes[nodes[root].right].is_leaf);

    EXPECT_EQ(nodes[nodes[root].left].count, 1);
    EXPECT_EQ(nodes[nodes[root].right].count, 1);

    ASSERT_EQ(bvh.indices().size(), static_cast<std::size_t>(2));
}

TEST(SAHBVH, TraceOperatorFromBVHHitsNearTriangle) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<double> bvh;
    bvh.set_leaf_size(1);
    bvh.set_num_of_bins(8);

    atlas::HostBuffer<atlas::TriangleContainer4<double>> tris;
    tris.resize(2);

    tris[0] = make_tri(
        atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 1.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    tris[1] = make_tri(
        atlas::math::Vector<double, 3>(10.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(11.0, 0.0, 0.0),
        atlas::math::Vector<double, 3>(10.0, 1.0, 0.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    bvh.build(tris);

    const auto op = bvh.make_trace_operator();

    const atlas::Ray<double> r(
        atlas::math::Vector<double, 3>(0.25, 0.25, 1.0),
        atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 1.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(
        point,
        atlas::math::Vector<double, 3>(0.25, 0.25, 0.0),
        eps));

    EXPECT_TRUE(atlas::test::near(normal.length(), 1.0, eps));
}