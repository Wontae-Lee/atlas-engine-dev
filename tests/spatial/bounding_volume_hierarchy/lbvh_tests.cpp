#include "../../utilities/tests_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Lbvh = atlas::spatial::LinearBoundingVolumeHierachy<T>;

atlas::HostBuffer<atlas::TriangleContainer4<T>>
make_triangles() {
    atlas::TriangleContainer4<T> triangle {};
    triangle[0] = Vec3(0, 0, 0);
    triangle[1] = Vec3(1, 0, 0);
    triangle[2] = Vec3(0, 1, 0);
    triangle[3] = Vec3(0, 0, 1);
    return { triangle };
}

} // namespace

TEST(LinearBoundingVolumeHierachy, DefaultStateIsEmpty) {
    const Lbvh bvh;

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
    Lbvh bvh;

    bvh.set_leaf_size(0);
    bvh.set_morton_bits(0);
    EXPECT_EQ(bvh.leaf_size(), 1);
    EXPECT_EQ(bvh.morton_bits(), 1);

    bvh.set_leaf_size(4);
    bvh.set_morton_bits(20);
    EXPECT_EQ(bvh.leaf_size(), 4);
    EXPECT_EQ(bvh.morton_bits(), 10);
}

TEST(LinearBoundingVolumeHierachy, BuildPopulatesHierarchyBuffers) {
    Lbvh bvh;
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

TEST(LinearBoundingVolumeHierachy, ResetClearsBuiltState) {
    Lbvh bvh;
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

TEST(LinearBoundingVolumeHierachy, GeometryOperatorReferencesCurrentBuffers) {
    Lbvh bvh;
    bvh.build(make_triangles());

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.bvh_root, bvh.root());
    EXPECT_NE(geometry_operator.bvh_nodes, nullptr);
    EXPECT_NE(geometry_operator.bvh_indices, nullptr);
    EXPECT_NE(geometry_operator.bvh_tris, nullptr);
}
