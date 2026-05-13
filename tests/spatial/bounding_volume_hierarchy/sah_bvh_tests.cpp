#include "../../utilities/test_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <testkit/testkit.h>

namespace {

using atlas::HostBuffer;
using atlas::TriangleContainer4;
using atlas::Vector3F;
using atlas::spatial::SurfaceAreaHeuristicBoundingVolumeHierachy;

using SahBvh = SurfaceAreaHeuristicBoundingVolumeHierachy<float>;

HostBuffer<TriangleContainer4<float>>
make_triangles() {
    TriangleContainer4<float> triangle {};
    triangle[0] = Vector3F(0, 0, 0);
    triangle[1] = Vector3F(1, 0, 0);
    triangle[2] = Vector3F(0, 1, 0);
    triangle[3] = Vector3F(0, 0, 1);
    return { triangle };
}

HostBuffer<TriangleContainer4<float>>
make_split_triangles() {
    TriangleContainer4<float> left {};
    left[0] = Vector3F(0, 0, 0);
    left[1] = Vector3F(1, 0, 0);
    left[2] = Vector3F(0, 1, 0);
    left[3] = Vector3F(0, 0, 1);

    TriangleContainer4<float> right {};
    right[0] = Vector3F(10, 0, 0);
    right[1] = Vector3F(11, 0, 0);
    right[2] = Vector3F(10, 1, 0);
    right[3] = Vector3F(0, 0, 1);

    TriangleContainer4<float> top {};
    top[0] = Vector3F(0, 10, 0);
    top[1] = Vector3F(1, 10, 0);
    top[2] = Vector3F(0, 11, 0);
    top[3] = Vector3F(0, 0, 1);

    return { left, right, top };
}

} // namespace

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, DefaultStateIsEmpty) {
    // Arrange: create a default SAH BVH.
    const SahBvh bvh;

    // Assert: default construction has no built hierarchy data.
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
    // Arrange: create a mutable SAH BVH.
    SahBvh bvh;

    // Act: set values below the supported minimum.
    bvh.set_leaf_size(0);
    bvh.set_num_of_bins(0);

    // Assert: lower-bound clamping is applied.
    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.num_of_bins(), 4);

    // Act: set supported and above-maximum values.
    bvh.set_leaf_size(8);
    bvh.set_num_of_bins(300);

    // Assert: supported values are preserved and upper-bound clamping is applied.
    EXPECT_EQ(bvh.leaf_size(), 8);
    EXPECT_EQ(bvh.num_of_bins(), 256);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildPopulatesHierarchyBuffers) {
    // Arrange: create a SAH BVH and deterministic triangle input.
    SahBvh bvh;
    const auto triangles = make_triangles();

    // Act: build the hierarchy.
    bvh.build(triangles);

    // Assert: build populates host and device hierarchy buffers.
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
    // Arrange: build a SAH BVH from one triangle.
    SahBvh bvh;
    bvh.build(make_triangles());

    // Assert: the root is a valid node index.
    ASSERT_GE(bvh.root(), 0);
    ASSERT_LT(bvh.root(), static_cast<int>(bvh.nodes().size()));

    // Assert: a single triangle produces a leaf root.
    const auto& root = bvh.nodes()[static_cast<std::size_t>(bvh.root())];
    EXPECT_TRUE(root.is_leaf);
    EXPECT_EQ(root.left, -1);
    EXPECT_EQ(root.right, -1);
    EXPECT_EQ(root.start, 0);
    EXPECT_EQ(root.count, 1);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildCreatesInternalNodeWhenSplitIsNeeded) {
    // Arrange: force splitting by setting one primitive per leaf.
    SahBvh bvh;
    bvh.set_leaf_size(1);

    const auto triangles = make_split_triangles();

    // Act: build from spatially separated triangles.
    bvh.build(triangles);

    // Assert: the root is a valid internal node.
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

    // Assert: children were created and at least one child is a leaf partition.
    const auto& left_child  = bvh.nodes()[static_cast<std::size_t>(root.left)];
    const auto& right_child = bvh.nodes()[static_cast<std::size_t>(root.right)];

    EXPECT_GE(left_child.count + right_child.count, 1);
    EXPECT_TRUE(left_child.is_leaf || right_child.is_leaf);
    EXPECT_GE(static_cast<int>(bvh.nodes().size()), 3);
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, BuildPreservesAllPrimitiveIndicesAfterPartition) {
    // Arrange: force a partitioned build.
    SahBvh bvh;
    bvh.set_leaf_size(1);

    const auto triangles = make_split_triangles();

    // Act: build and sort the emitted primitive indices.
    bvh.build(triangles);

    ASSERT_EQ(bvh.indices().size(), triangles.size());

    std::vector<int> sorted_indices(bvh.indices().begin(), bvh.indices().end());
    std::sort(sorted_indices.begin(), sorted_indices.end());

    // Assert: partitioning preserves every primitive index exactly once.
    EXPECT_EQ(sorted_indices, (std::vector<int> { 0, 1, 2 }));
}

TEST(SurfaceAreaHeuristicBoundingVolumeHierachy, ResetClearsBuiltState) {
    // Arrange: build a SAH BVH.
    SahBvh bvh;
    bvh.build(make_triangles());

    // Act: reset the hierarchy.
    bvh.reset();

    // Assert: reset clears built hierarchy state.
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
    // Arrange: build a SAH BVH with triangle data.
    SahBvh bvh;
    bvh.build(make_triangles());

    // Act: create a geometry operator from current buffers.
    const auto geometry_operator = bvh.make_geometry_operator();

    // Assert: the operator references populated hierarchy buffers.
    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
