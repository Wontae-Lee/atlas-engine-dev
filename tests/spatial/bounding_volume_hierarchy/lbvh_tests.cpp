#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/random/seed.h>

#include <gtest/gtest.h>

namespace {

using atlas::HostBuffer;
using atlas::LBVH;
using atlas::MORTON_EXPAND_BITS_FINAL_MASK;
using atlas::MORTON_EXPAND_BITS_FINAL_MULTIPLIER;
using atlas::MORTON_EXPAND_BITS_FIRST_MASK;
using atlas::MORTON_EXPAND_BITS_FIRST_MULTIPLIER;
using atlas::MORTON_EXPAND_BITS_SECOND_MASK;
using atlas::MORTON_EXPAND_BITS_SECOND_MULTIPLIER;
using atlas::MORTON_EXPAND_BITS_THIRD_MASK;
using atlas::MORTON_EXPAND_BITS_THIRD_MULTIPLIER;
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

}

TEST(LBVH, DefaultStateIsEmpty) {
    const LBVH bvh;

    EXPECT_EQ(bvh.root(), -1);
    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.morton_bits(), 10);
    EXPECT_TRUE(bvh.nodes().empty());
    EXPECT_TRUE(bvh.indices().empty());
    EXPECT_TRUE(bvh.bounds().empty());
    EXPECT_TRUE(bvh.centroids().empty());
    EXPECT_TRUE(bvh.device_nodes().empty());
    EXPECT_TRUE(bvh.device_indices().empty());
    EXPECT_TRUE(bvh.device_triangles().empty());
}

TEST(LBVH, SettersClampToSupportedRange) {
    LBVH bvh;

    bvh.set_leaf_size(0);
    bvh.set_morton_bits(0);

    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.morton_bits(), 1);

    bvh.set_leaf_size(4);
    bvh.set_morton_bits(20);

    EXPECT_EQ(bvh.leaf_size(), 4);
    EXPECT_EQ(bvh.morton_bits(), 10);
}

TEST(LBVH, MortonExpandBitConstantsExposeExpectedValues) {
    EXPECT_EQ(MORTON_EXPAND_BITS_FIRST_MULTIPLIER, 0x00010001u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FIRST_MASK, 0xFF0000FFu);
    EXPECT_EQ(MORTON_EXPAND_BITS_SECOND_MULTIPLIER, 0x00000101u);
    EXPECT_EQ(MORTON_EXPAND_BITS_SECOND_MASK, 0x0F00F00Fu);
    EXPECT_EQ(MORTON_EXPAND_BITS_THIRD_MULTIPLIER, 0x00000011u);
    EXPECT_EQ(MORTON_EXPAND_BITS_THIRD_MASK, 0xC30C30C3u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FINAL_MULTIPLIER, 0x00000005u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FINAL_MASK, 0x49249249u);
}

TEST(LBVH, BuildPopulatesHierarchyBuffers) {
    LBVH bvh;
    const auto triangles = make_triangles();

    bvh.build(triangles);

    EXPECT_EQ(bvh.root(), 0);
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

TEST(LBVH, ResetClearsBuiltState) {
    LBVH bvh;
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

TEST(LBVH, GeometryOperatorReferencesCurrentBuffers) {
    LBVH bvh;
    bvh.build(make_triangles());

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
