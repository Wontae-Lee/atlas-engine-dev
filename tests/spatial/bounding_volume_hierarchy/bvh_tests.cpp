#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <atlas/geometry/triangle_mesh.h>

#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

using atlas::BVH;
using atlas::BVH;
using atlas::BVHDevicePtr;
using atlas::BVHHostPtr;
using atlas::BvhGeometryOperator;
using atlas::HostBuffer;
using atlas::TriangleContainer4;
using atlas::Float3;
using atlas::device_shared_ptr;
using atlas::host_shared_ptr;

class DummyBvh final : public BVH {
public:
    void
    build(const HostBuffer<TriangleContainer4>& triangles) override {
        built          = true;
        triangle_count = triangles.size();
    }

    BvhGeometryOperator
    make_geometry_operator() const override {
        return {};
    }

    bool built = false;

    std::size_t triangle_count = 0;
};

}

TEST(BVH, AliasTypesCompile) {
    static_assert(std::is_same_v<BVH, BVH>);
    static_assert(std::is_same_v<BVHHostPtr, host_shared_ptr<BVH>>);
    static_assert(std::is_same_v<BVHDevicePtr, device_shared_ptr<BVH>>);

    SUCCEED();
}

TEST(BVH, DerivedImplementationCanBuild) {
    DummyBvh bvh;

    const TriangleContainer4 triangle(Float3(0.0f, 0.0f, 0.0f),
                                      Float3(1.0f, 0.0f, 0.0f),
                                      Float3(0.0f, 1.0f, 0.0f),
                                      Float3(0.0f, 0.0f, 0.0f));

    HostBuffer<TriangleContainer4> triangles;
    triangles.push_back(triangle);

    bvh.build(triangles);

    EXPECT_TRUE(bvh.built);
    EXPECT_EQ(bvh.triangle_count, triangles.size());
}

TEST(BVH, DerivedImplementationReturnsGeometryOperator) {
    const DummyBvh bvh;

    const auto geometry_operator = bvh.make_geometry_operator();

    EXPECT_EQ(geometry_operator.triangle_count, 0);
    EXPECT_EQ(geometry_operator.vertices, nullptr);
    EXPECT_EQ(geometry_operator.indices, nullptr);
}
