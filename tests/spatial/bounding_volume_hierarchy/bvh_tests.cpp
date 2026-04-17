#include "../../utilities/tests_utils.h"

#include <atlas/geometry/triangle_mesh.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <gtest/gtest.h>

namespace {

using T = float;

class DummyBvh final : public atlas::spatial::BoundingVolumeHierachy<T> {
public:
    void
    build(const atlas::HostBuffer<atlas::TriangleContainer4<T>>& triangles) override {

        built = true;
        triangle_count = triangles.size();
    }

    atlas::spatial::BvhGeometryOperator<T>
    make_geometry_operator() const override {

        return {};
    }

    bool built = false;

    std::size_t triangle_count = 0;
};

} // namespace

TEST(BoundingVolumeHierarchy, AliasTypesCompile) {
    static_assert(std::is_same_v<atlas::BVH<T>, atlas::spatial::BoundingVolumeHierachy<T>>);
    static_assert(std::is_same_v<atlas::BVHHostPtr<T>, atlas::host_shared_ptr<atlas::BVH<T>>>);
    static_assert(std::is_same_v<atlas::BVHDevicePtr<T>, atlas::device_shared_ptr<atlas::BVH<T>>>);

    SUCCEED();
}

TEST(BoundingVolumeHierarchy, DerivedImplementationCanBuild) {
    DummyBvh bvh;

    atlas::TriangleContainer4<T> triangle {};
    triangle[0] = atlas::Vector3<T>(0, 0, 0);
    triangle[1] = atlas::Vector3<T>(1, 0, 0);
    triangle[2] = atlas::Vector3<T>(0, 1, 0);
    triangle[3] = atlas::Vector3<T>(0, 0, 0);

    const atlas::HostBuffer<atlas::TriangleContainer4<T>> triangles { triangle };

    bvh.build(triangles);

    EXPECT_TRUE(bvh.built);
    EXPECT_EQ(bvh.triangle_count, triangles.size());
}

TEST(BoundingVolumeHierarchy, DerivedImplementationReturnsGeometryOperator) {
    const DummyBvh bvh;

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.triangle_count, 0);
    EXPECT_EQ(geometry_operator.vertices, nullptr);
    EXPECT_EQ(geometry_operator.indices, nullptr);
}
