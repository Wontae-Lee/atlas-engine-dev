#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <atlas/geometry/triangle_mesh.h>

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

using atlas::HostBuffer;
using atlas::SAHBVH;
using atlas::TriangleContainer4;
using atlas::Vector3;


HostBuffer<TriangleContainer4>
make_triangles() {
    const TriangleContainer4 triangle(Vector3(0.0f, 0.0f, 0.0f),
                                      Vector3(1.0f, 0.0f, 0.0f),
                                      Vector3(0.0f, 1.0f, 0.0f),
                                      Vector3(0.0f, 0.0f, 1.0f));

    HostBuffer<TriangleContainer4> triangles;
    triangles.push_back(triangle);

    return triangles;
}

HostBuffer<TriangleContainer4>
make_split_triangles() {
    const TriangleContainer4 left(Vector3(0.0f, 0.0f, 0.0f),
                                  Vector3(1.0f, 0.0f, 0.0f),
                                  Vector3(0.0f, 1.0f, 0.0f),
                                  Vector3(0.0f, 0.0f, 1.0f));

    const TriangleContainer4 right(Vector3(10.0f, 0.0f, 0.0f),
                                   Vector3(11.0f, 0.0f, 0.0f),
                                   Vector3(10.0f, 1.0f, 0.0f),
                                   Vector3(0.0f, 0.0f, 1.0f));

    const TriangleContainer4 top(Vector3(0.0f, 10.0f, 0.0f),
                                 Vector3(1.0f, 10.0f, 0.0f),
                                 Vector3(0.0f, 11.0f, 0.0f),
                                 Vector3(0.0f, 0.0f, 1.0f));

    HostBuffer<TriangleContainer4> triangles;
    triangles.push_back(left);
    triangles.push_back(right);
    triangles.push_back(top);

    return triangles;
}

}

TEST(SAHBVH, DefaultStateIsEmpty) {
    const SAHBVH bvh;

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_EQ(bvh.leaf_size(), 32);
    EXPECT_EQ(bvh.bin_count(), 100);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
    EXPECT_TRUE(bvh.bounds().empty());
    EXPECT_TRUE(bvh.centroids().empty());
    EXPECT_TRUE(bvh.device_nodes().empty());
    EXPECT_TRUE(bvh.device_indices().empty());
    EXPECT_TRUE(bvh.device_triangles().empty());
}

TEST(SAHBVH, SettersClampToSupportedRange) {
    SAHBVH bvh;

    bvh.set_leaf_size(0);
    bvh.set_bin_count(0);

    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.bin_count(), 4);

    bvh.set_leaf_size(8);
    bvh.set_bin_count(300);

    EXPECT_EQ(bvh.leaf_size(), 8);
    EXPECT_EQ(bvh.bin_count(), 256);
}

TEST(SAHBVH, BuildPopulatesHierarchyBuffers) {
    SAHBVH bvh;
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

TEST(SAHBVH, BuildCreatesLeafForSingleTriangle) {
    SAHBVH bvh;
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

TEST(SAHBVH, BuildCreatesInternalNodeWhenSplitIsNeeded) {
    SAHBVH bvh;
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

    const auto& left_child  = bvh.nodes()[static_cast<std::size_t>(root.left)];
    const auto& right_child = bvh.nodes()[static_cast<std::size_t>(root.right)];

    EXPECT_GE(left_child.count + right_child.count, 1);
    EXPECT_TRUE(left_child.is_leaf || right_child.is_leaf);
    EXPECT_GE(static_cast<int>(bvh.nodes().size()), 3);
}

TEST(SAHBVH, BuildPreservesAllPrimitiveIndicesAfterPartition) {
    SAHBVH bvh;
    bvh.set_leaf_size(1);

    const auto triangles = make_split_triangles();

    bvh.build(triangles);

    ASSERT_EQ(bvh.indices().size(), triangles.size());

    std::vector<int> sorted_indices(bvh.indices().begin(), bvh.indices().end());
    std::sort(sorted_indices.begin(), sorted_indices.end());

    EXPECT_EQ(sorted_indices, (std::vector<int> { 0, 1, 2 }));
}

TEST(SAHBVH, ResetClearsBuiltState) {
    SAHBVH bvh;
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

TEST(SAHBVH, GeometryOperatorReferencesCurrentBuffers) {
    SAHBVH bvh;
    bvh.build(make_triangles());

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
