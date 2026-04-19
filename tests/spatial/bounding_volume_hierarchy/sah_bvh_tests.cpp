#include "../../utilities/tests_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using SahBvh = atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy<T>;

atlas::HostBuffer<atlas::TriangleContainer4<T>>
make_triangles() {
    atlas::TriangleContainer4<T> triangle {};
    triangle[0] = Vec3(0, 0, 0);
    triangle[1] = Vec3(1, 0, 0);
    triangle[2] = Vec3(0, 1, 0);
    triangle[3] = Vec3(0, 0, 1);
    return { triangle };
}

atlas::HostBuffer<atlas::TriangleContainer4<T>>
make_split_triangles() {
    atlas::TriangleContainer4<T> left {};
    left[0] = Vec3(0, 0, 0);
    left[1] = Vec3(1, 0, 0);
    left[2] = Vec3(0, 1, 0);
    left[3] = Vec3(0, 0, 1);

    atlas::TriangleContainer4<T> right {};
    right[0] = Vec3(10, 0, 0);
    right[1] = Vec3(11, 0, 0);
    right[2] = Vec3(10, 1, 0);
    right[3] = Vec3(0, 0, 1);

    atlas::TriangleContainer4<T> top {};
    top[0] = Vec3(0, 10, 0);
    top[1] = Vec3(1, 10, 0);
    top[2] = Vec3(0, 11, 0);
    top[3] = Vec3(0, 0, 1);

    return { left, right, top };
}

} // namespace

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, DefaultStateIsEmpty) {
    const SahBvh bvh;

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_EQ(bvh.leaf_size(), 32);
    EXPECT_EQ(bvh.num_of_bins(), 100);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
    EXPECT_TRUE(bvh.bounds().empty());
    EXPECT_TRUE(bvh.centroids().empty());
    EXPECT_TRUE(bvh.device_nodes().empty());
    EXPECT_TRUE(bvh.device_indices().empty());
    EXPECT_TRUE(bvh.device_triangles().empty());
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, SettersClampToSupportedRange) {
    SahBvh bvh;

    bvh.set_leaf_size(0);
    bvh.set_num_of_bins(0);
    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.num_of_bins(), 4);

    bvh.set_leaf_size(8);
    bvh.set_num_of_bins(300);
    EXPECT_EQ(bvh.leaf_size(), 8);
    EXPECT_EQ(bvh.num_of_bins(), 256);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildPopulatesHierarchyBuffers) {
    SahBvh bvh;
    const auto triangles = make_triangles();

    bvh.build(triangles);

    EXPECT_GE(bvh.root(), 0);
    EXPECT_FALSE(bvh.nodes().empty());
    EXPECT_FALSE(bvh.indices().empty());
    EXPECT_FALSE(bvh.bounds().empty());
    EXPECT_FALSE(bvh.centroids().empty());
    EXPECT_FALSE(bvh.device_nodes().empty());
    EXPECT_FALSE(bvh.device_indices().empty());
    EXPECT_FALSE(bvh.device_triangles().empty());
    EXPECT_EQ(bvh.indices().size(), triangles.size());
    EXPECT_EQ(bvh.bounds().size(), triangles.size());
    EXPECT_EQ(bvh.centroids().size(), triangles.size());
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildCreatesLeafForSingleTriangle) {
    SahBvh bvh;
    bvh.build(make_triangles());

    ASSERT_GE(bvh.root(), 0);
    ASSERT_LT(bvh.root(), static_cast<int>(bvh.nodes().size()));

    const auto& root = bvh.nodes()[static_cast<std::size_t>(bvh.root())];
    EXPECT_TRUE(root.is_leaf);
    EXPECT_EQ(root.left, -1);
    EXPECT_EQ(root.right, -1);
    EXPECT_EQ(root.start, 0);
    EXPECT_EQ(root.count, 1);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildCreatesInternalNodeWhenSplitIsNeeded) {
    SahBvh bvh;
    bvh.set_leaf_size(1);

    const auto triangles = make_split_triangles();
    bvh.build(triangles);

    ASSERT_GE(bvh.root(), 0);
    ASSERT_LT(bvh.root(), static_cast<int>(bvh.nodes().size()));

    const auto& root = bvh.nodes()[static_cast<std::size_t>(bvh.root())];
    EXPECT_FALSE(root.is_leaf);
    EXPECT_GE(root.left, 0);
    EXPECT_GE(root.right, 0);
    EXPECT_EQ(root.start, -1);
    EXPECT_EQ(root.count, 0);
    ASSERT_LT(root.left, static_cast<int>(bvh.nodes().size()));
    ASSERT_LT(root.right, static_cast<int>(bvh.nodes().size()));

    const auto& left_child = bvh.nodes()[static_cast<std::size_t>(root.left)];
    const auto& right_child = bvh.nodes()[static_cast<std::size_t>(root.right)];

    EXPECT_GE(left_child.count + right_child.count, 1);
    EXPECT_TRUE(left_child.is_leaf || right_child.is_leaf);
    EXPECT_GE(static_cast<int>(bvh.nodes().size()), 3);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildPreservesAllPrimitiveIndicesAfterPartition) {
    SahBvh bvh;
    bvh.set_leaf_size(1);

    const auto triangles = make_split_triangles();
    bvh.build(triangles);

    ASSERT_EQ(bvh.indices().size(), triangles.size());

    std::vector<int> sorted_indices(bvh.indices().begin(), bvh.indices().end());
    std::sort(sorted_indices.begin(), sorted_indices.end());

    EXPECT_EQ(sorted_indices, (std::vector<int> { 0, 1, 2 }));
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, ResetClearsBuiltState) {
    SahBvh bvh;
    bvh.build(make_triangles());

    bvh.reset();

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
    EXPECT_TRUE(bvh.bounds().empty());
    EXPECT_TRUE(bvh.centroids().empty());
    EXPECT_TRUE(bvh.device_nodes().empty());
    EXPECT_TRUE(bvh.device_indices().empty());
    EXPECT_TRUE(bvh.device_triangles().empty());
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, GeometryOperatorReferencesCurrentBuffers) {
    SahBvh bvh;
    bvh.build(make_triangles());

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
