#include "../../utilities/test_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <testkit/testkit.h>

namespace {

using atlas::BVH;
using atlas::BVHDevicePtr;
using atlas::BVHHostPtr;
using atlas::HostBuffer;
using atlas::TriangleContainer4;
using atlas::Vector3F;
using atlas::device_shared_ptr;
using atlas::host_shared_ptr;
using atlas::spatial::BoundingVolumeHierachy;
using atlas::spatial::BvhGeometryOperator;

class DummyBvh final : public BoundingVolumeHierachy<float> {
public:
    void
    build(const HostBuffer<TriangleContainer4<float>>& triangles) override {
        built = true;
        triangle_count = triangles.size();
    }

    BvhGeometryOperator<float>
    make_geometry_operator() const override {
        return {};
    }

    bool built = false;

    std::size_t triangle_count = 0;
};

} // namespace

TEST(BoundingVolumeHierarchy, AliasTypesCompile) {
    // Assert: public aliases match the underlying BVH types.
    static_assert(std::is_same_v<BVH<float>, BoundingVolumeHierachy<float>>);
    static_assert(std::is_same_v<BVHHostPtr<float>, host_shared_ptr<BVH<float>>>);
    static_assert(std::is_same_v<BVHDevicePtr<float>, device_shared_ptr<BVH<float>>>);

    SUCCEED();
}

TEST(BoundingVolumeHierarchy, DerivedImplementationCanBuild) {
    // Arrange: create a concrete test BVH and one triangle.
    DummyBvh bvh;

    TriangleContainer4<float> triangle {};
    triangle[0] = Vector3F(0, 0, 0);
    triangle[1] = Vector3F(1, 0, 0);
    triangle[2] = Vector3F(0, 1, 0);
    triangle[3] = Vector3F(0, 0, 0);

    const HostBuffer<TriangleContainer4<float>> triangles { triangle };

    // Act: build the derived BVH implementation.
    bvh.build(triangles);

    // Assert: the derived implementation observed the input triangles.
    EXPECT_TRUE(bvh.built);
    EXPECT_EQ(bvh.triangle_count, triangles.size());
}

TEST(BoundingVolumeHierarchy, DerivedImplementationReturnsGeometryOperator) {
    // Arrange: create a default concrete test BVH.
    const DummyBvh bvh;

    // Act: request a geometry operator from the derived implementation.
    const auto geometry_operator = bvh.make_geometry_operator();

    // Assert: the default operator is empty.
    EXPECT_EQ(geometry_operator.triangle_count, 0);
    EXPECT_EQ(geometry_operator.vertices, nullptr);
    EXPECT_EQ(geometry_operator.indices, nullptr);
}
