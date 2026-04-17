#include "../../utilities/tests_utils.h"

#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Node = atlas::spatial::BVHNode<T>;
using Aabb = atlas::spatial::AxisAlignedBoundingBox<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(BoundingVolumeHierarchyNode, DefaultStateMatchesInternalSentinelValues) {
    const Node node;

    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_FALSE(node.is_leaf);
    EXPECT_FALSE(node.bounds.is_valid());
}

TEST(BoundingVolumeHierarchyNode, LeafNodeCanStorePrimitiveRange) {
    Node node;

    node.bounds = Aabb(atlas::Vector3<T>(0, 0, 0), atlas::Vector3<T>(1, 2, 3));
    node.start = 4;
    node.count = 7;
    node.is_leaf = true;

    EXPECT_TRUE(node.is_leaf);
    EXPECT_EQ(node.start, 4);
    EXPECT_EQ(node.count, 7);
    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_TRUE(node.bounds.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(node.bounds.lower_corner, atlas::Vector3<T>(0, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(node.bounds.upper_corner, atlas::Vector3<T>(1, 2, 3), kEps));
}

TEST(BoundingVolumeHierarchyNode, InternalNodeCanStoreChildLinks) {
    Node node;

    node.bounds = Aabb(atlas::Vector3<T>(-1, -2, -3), atlas::Vector3<T>(3, 2, 1));
    node.left = 1;
    node.right = 2;
    node.start = -1;
    node.count = 0;
    node.is_leaf = false;

    EXPECT_FALSE(node.is_leaf);
    EXPECT_EQ(node.left, 1);
    EXPECT_EQ(node.right, 2);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_TRUE(node.bounds.is_valid());
    EXPECT_TRUE(atlas::test::vec_near(node.bounds.center(), atlas::Vector3<T>(1, 0, -1), kEps));
}
