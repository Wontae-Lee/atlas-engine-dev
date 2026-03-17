#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SurfaceDespawnOperator, DespawnCollectsIndicesOfSurfaceParticlesFromBoxQueryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_query_operator(box);

    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 1.0),
        Vector3<double>(0.5, 0.5, 0.5),
        Vector3<double>(-1.0, -1.0, -1.0)
    };

    DeviceBuffer<int> despawn_indices;

    system::SurfaceDespawnOperator<double> {}.despawn(
        despawn_indices,
        particles,
        query,
        0.0);

    ASSERT_EQ(despawn_indices.size(), 3u);
    EXPECT_EQ(despawn_indices[0], 1);
    EXPECT_EQ(despawn_indices[1], 2);
    EXPECT_EQ(despawn_indices[2], 4);
}

TEST(SurfaceDespawnOperator, DespawnClampsNegativeToleranceToZero) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_query_operator(sphere);

    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(0.0, 0.0, 1.0),
        Vector3<double>(0.0, 0.0, 0.5),
        Vector3<double>(1.0, 0.0, 0.0)
    };

    DeviceBuffer<int> despawn_indices;

    system::SurfaceDespawnOperator<double> {}.despawn(
        despawn_indices,
        particles,
        query,
        -1.0);

    ASSERT_EQ(despawn_indices.size(), 2u);
    EXPECT_EQ(despawn_indices[0], 0);
    EXPECT_EQ(despawn_indices[1], 2);
}

TEST(SurfaceDespawnOperator, DespawnReturnsEmptyBufferForInvalidQueryOperator) {
    const geometry::QueryOperator<double> query;
    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0)
    };

    DeviceBuffer<int> despawn_indices = { 7, 8, 9 };

    system::SurfaceDespawnOperator<double> {}.despawn(
        despawn_indices,
        particles,
        query,
        0.0);

    EXPECT_TRUE(despawn_indices.empty());
}
