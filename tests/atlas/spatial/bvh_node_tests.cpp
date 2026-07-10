#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <gtest/gtest.h>

namespace {

using atlas::BVHNode;
using atlas::LBVH;
using atlas::SAHBVH;

}

TEST(BVHNode, DefaultIsAnInteriorNodeWithNullLinks) {
    const BVHNode node;

    EXPECT_FALSE(node.is_leaf);
    EXPECT_EQ(node.left, -1);
    EXPECT_EQ(node.right, -1);
    EXPECT_EQ(node.start, -1);
    EXPECT_EQ(node.count, 0);
}

TEST(BVHNode, DefaultBoundsAndMomentsAreCleared) {
    const BVHNode node;

    // The bounds member default-constructs to the canonical empty (invalid) box.
    EXPECT_FALSE(node.bounds.is_valid());

    EXPECT_FLOAT_EQ(node.solid_angle_moment.x, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_moment.y, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_moment.z, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_normal_area.x, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_normal_area.y, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_normal_area.z, 0.0f);
    EXPECT_FLOAT_EQ(node.solid_angle_area, 0.0f);
}

TEST(BVHNode, LeafConfigurationDistinguishesFromInterior) {
    BVHNode leaf;
    leaf.is_leaf = true;
    leaf.start   = 4;
    leaf.count   = 3;

    EXPECT_TRUE(leaf.is_leaf);
    EXPECT_EQ(leaf.start, 4);
    EXPECT_EQ(leaf.count, 3);

    BVHNode interior;
    interior.left  = 1;
    interior.right = 2;

    EXPECT_FALSE(interior.is_leaf);
    EXPECT_EQ(interior.left, 1);
    EXPECT_EQ(interior.right, 2);
}

TEST(LBVH, DefaultConfigurationBeforeBuild) {
    const LBVH bvh;

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_EQ(bvh.morton_bits(), 10);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
}

TEST(LBVH, SetMortonBitsClampsIntoOneToTen) {
    LBVH bvh;

    bvh.set_morton_bits(6);
    EXPECT_EQ(bvh.morton_bits(), 6);

    bvh.set_morton_bits(0);
    EXPECT_EQ(bvh.morton_bits(), 1);

    bvh.set_morton_bits(11);
    EXPECT_EQ(bvh.morton_bits(), 10);
}

TEST(SAHBVH, DefaultConfigurationBeforeBuild) {
    const SAHBVH bvh;

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_EQ(bvh.leaf_size(), 32);
    EXPECT_EQ(bvh.bin_count(), 100);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
}

TEST(SAHBVH, SetLeafSizeClampsBelowOneToOne) {
    SAHBVH bvh;

    bvh.set_leaf_size(16);
    EXPECT_EQ(bvh.leaf_size(), 16);

    bvh.set_leaf_size(0);
    EXPECT_EQ(bvh.leaf_size(), 1);
}

TEST(SAHBVH, SetBinCountClampsIntoFourToTwoFiftySix) {
    SAHBVH bvh;

    bvh.set_bin_count(64);
    EXPECT_EQ(bvh.bin_count(), 64);

    bvh.set_bin_count(2);
    EXPECT_EQ(bvh.bin_count(), 4);

    bvh.set_bin_count(1000);
    EXPECT_EQ(bvh.bin_count(), 256);
}
