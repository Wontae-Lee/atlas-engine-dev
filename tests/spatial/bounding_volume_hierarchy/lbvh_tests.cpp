#include "../../utilities/test_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/random/seed.h>
#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>

#include <testkit/testkit.h>

namespace {

using atlas::HostBuffer;
using atlas::TriangleContainer4;
using atlas::Vector3F;
using atlas::seed::MORTON_EXPAND_BITS_FINAL_MASK;
using atlas::seed::MORTON_EXPAND_BITS_FINAL_MULTIPLIER;
using atlas::seed::MORTON_EXPAND_BITS_FIRST_MASK;
using atlas::seed::MORTON_EXPAND_BITS_FIRST_MULTIPLIER;
using atlas::seed::MORTON_EXPAND_BITS_SECOND_MASK;
using atlas::seed::MORTON_EXPAND_BITS_SECOND_MULTIPLIER;
using atlas::seed::MORTON_EXPAND_BITS_THIRD_MASK;
using atlas::seed::MORTON_EXPAND_BITS_THIRD_MULTIPLIER;
using atlas::spatial::LinearBoundingVolumeHierachy;

using Lbvh = LinearBoundingVolumeHierachy<float>;

HostBuffer<TriangleContainer4<float>>
make_triangles() {
    TriangleContainer4<float> triangle {};
    triangle[0] = Vector3F(0, 0, 0);
    triangle[1] = Vector3F(1, 0, 0);
    triangle[2] = Vector3F(0, 1, 0);
    triangle[3] = Vector3F(0, 0, 1);
    return { triangle };
}

} // namespace

TEST(LinearBoundingVolumeHierachy, DefaultStateIsEmpty) {
    // Arrange: create a default LBVH.
    const Lbvh bvh;

    // Assert: default construction has no built hierarchy data.
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

TEST(LinearBoundingVolumeHierachy, SettersClampToSupportedRange) {
    // Arrange: create a mutable LBVH.
    Lbvh bvh;

    // Act: set values below the supported minimum.
    bvh.set_leaf_size(0);
    bvh.set_morton_bits(0);

    // Assert: lower-bound clamping is applied.
    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.morton_bits(), 1);

    // Act: set supported and above-maximum values.
    bvh.set_leaf_size(4);
    bvh.set_morton_bits(20);

    // Assert: supported values are preserved and upper-bound clamping is applied.
    EXPECT_EQ(bvh.leaf_size(), 4);
    EXPECT_EQ(bvh.morton_bits(), 10);
}

TEST(LinearBoundingVolumeHierachy, MortonExpandBitConstantsExposeExpectedValues) {
    // Assert: Morton bit expansion constants match the standard 10-bit interleaving stages.
    EXPECT_EQ(MORTON_EXPAND_BITS_FIRST_MULTIPLIER, 0x00010001u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FIRST_MASK, 0xFF0000FFu);
    EXPECT_EQ(MORTON_EXPAND_BITS_SECOND_MULTIPLIER, 0x00000101u);
    EXPECT_EQ(MORTON_EXPAND_BITS_SECOND_MASK, 0x0F00F00Fu);
    EXPECT_EQ(MORTON_EXPAND_BITS_THIRD_MULTIPLIER, 0x00000011u);
    EXPECT_EQ(MORTON_EXPAND_BITS_THIRD_MASK, 0xC30C30C3u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FINAL_MULTIPLIER, 0x00000005u);
    EXPECT_EQ(MORTON_EXPAND_BITS_FINAL_MASK, 0x49249249u);
}

TEST(LinearBoundingVolumeHierachy, BuildPopulatesHierarchyBuffers) {
    // Arrange: create an LBVH and deterministic triangle input.
    Lbvh bvh;
    const auto triangles = make_triangles();

    // Act: build the hierarchy.
    bvh.build(triangles);

    // Assert: build populates host and device hierarchy buffers.
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

TEST(LinearBoundingVolumeHierachy, ResetClearsBuiltState) {
    // Arrange: build an LBVH.
    Lbvh bvh;
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

TEST(LinearBoundingVolumeHierachy, GeometryOperatorReferencesCurrentBuffers) {
    // Arrange: build an LBVH with triangle data.
    Lbvh bvh;
    bvh.build(make_triangles());

    // Act: create a geometry operator from current buffers.
    const auto geometry_operator = bvh.make_geometry_operator();

    // Assert: the operator references populated hierarchy buffers.
    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
