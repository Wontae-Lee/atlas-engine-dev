#include "../../utilities/test_utils.h"

#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <testkit/testkit.h>

namespace {

using atlas::AABBF;
using atlas::Vector3F;
using atlas::spatial::BVHNode;
using atlas::test::vec_near;
using atlas::tol;

using Node = BVHNode<float>;

} // namespace

TEST(BoundingVolumeHierarchyNode, DefaultStateMatchesInternalSentinelValues) {
    // Arrange: create a default BVH node.
    const Node node;

    // Assert: default node state uses internal sentinel values.
    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_FALSE(node.is_leaf);
    EXPECT_FALSE(node.bounds.is_valid());
}

TEST(BoundingVolumeHierarchyNode, LeafNodeCanStorePrimitiveRange) {
    // Arrange: create a mutable node.
    Node node;

    // Act: configure it as a leaf over a primitive range.
    node.bounds  = AABBF(Vector3F(0, 0, 0), Vector3F(1, 2, 3));
    node.start   = 4;
    node.count   = 7;
    node.is_leaf = true;

    // Assert: leaf metadata and bounds are preserved.
    EXPECT_TRUE(node.is_leaf);
    EXPECT_EQ(node.start, 4);
    EXPECT_EQ(node.count, 7);
    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_TRUE(node.bounds.is_valid());
    EXPECT_TRUE(vec_near(node.bounds.lower_corner, Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(node.bounds.upper_corner, Vector3F(1, 2, 3), tol));
}

TEST(BoundingVolumeHierarchyNode, InternalNodeCanStoreChildLinks) {
    // Arrange: create a mutable node.
    Node node;

    // Act: configure it as an internal node with child links.
    node.bounds  = AABBF(Vector3F(-1, -2, -3), Vector3F(3, 2, 1));
    node.left    = 1;
    node.right   = 2;
    node.start   = -1;
    node.count   = 0;
    node.is_leaf = false;

    // Assert: internal metadata and bounds are preserved.
    EXPECT_FALSE(node.is_leaf);
    EXPECT_EQ(node.left, 1);
    EXPECT_EQ(node.right, 2);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_TRUE(node.bounds.is_valid());
    EXPECT_TRUE(vec_near(node.bounds.center(), Vector3F(1, 0, -1), tol));
}
