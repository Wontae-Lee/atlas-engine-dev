#include <atlas/spatial/bounding_volume_hierarchy/node.h>

#include <gtest/gtest.h>

namespace {

using atlas::AABB;
using atlas::BVHNode;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(BVHNode, DefaultStateMatchesInternalSentinelValues) {
    const BVHNode node;

    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_FALSE(node.is_leaf);
    EXPECT_FALSE(node.bounds.is_valid());
}

TEST(BVHNode, LeafNodeCanStorePrimitiveRange) {
    BVHNode node;

    node.bounds  = AABB(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 2.0f, 3.0f));
    node.start   = 4;
    node.count   = 7;
    node.is_leaf = true;

    EXPECT_TRUE(node.is_leaf);
    EXPECT_EQ(node.start, 4);
    EXPECT_EQ(node.count, 7);
    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_TRUE(node.bounds.is_valid());
    expect_vec_near(node.bounds.lower_corner, Vector3(0.0f, 0.0f, 0.0f));
    expect_vec_near(node.bounds.upper_corner, Vector3(1.0f, 2.0f, 3.0f));
}

TEST(BVHNode, InternalNodeCanStoreChildLinks) {
    BVHNode node;

    node.bounds  = AABB(Vector3(-1.0f, -2.0f, -3.0f), Vector3(3.0f, 2.0f, 1.0f));
    node.left    = 1;
    node.right   = 2;
    node.start   = -1;
    node.count   = 0;
    node.is_leaf = false;

    EXPECT_FALSE(node.is_leaf);
    EXPECT_EQ(node.left, 1);
    EXPECT_EQ(node.right, 2);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
    EXPECT_TRUE(node.bounds.is_valid());
    expect_vec_near(node.bounds.center(), Vector3(1.0f, 0.0f, -1.0f));
}
