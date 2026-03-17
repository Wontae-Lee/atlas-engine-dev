#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(VolumeDespawnOperator, DespawnCollectsIndicesOfInteriorParticlesFromSphereQueryOperator) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_query_operator(sphere);

    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(0.25, 0.25, 0.25),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0)
    };

    DeviceBuffer<int> despawn_indices;

    system::VolumeDespawnOperator<double> {}.despawn(
        despawn_indices,
        particles,
        query,
        0.0);

    ASSERT_EQ(despawn_indices.size(), 3u);
    EXPECT_EQ(despawn_indices[0], 0);
    EXPECT_EQ(despawn_indices[1], 1);
    EXPECT_EQ(despawn_indices[2], 2);
}

TEST(VolumeDespawnOperator, DespawnClampsNegativeToleranceToZero) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_query_operator(box);

    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 1.0, 1.0),
        Vector3<double>(2.0, 0.0, 0.0)
    };

    DeviceBuffer<int> despawn_indices;

    system::VolumeDespawnOperator<double> {}.despawn(
        despawn_indices,
        particles,
        query,
        -10.0);

    ASSERT_EQ(despawn_indices.size(), 2u);
    EXPECT_EQ(despawn_indices[0], 0);
    EXPECT_EQ(despawn_indices[1], 1);
}

TEST(DespawnOperator, DespawnDispatchesByRuntimeType) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_query_operator(box);

    const DeviceBuffer<Vector3<double>> particles = {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0)
    };

    DeviceBuffer<int> surface_indices;
    DeviceBuffer<int> volume_indices;

    system::DespawnOperator<double>(system::DespawnType::Surface).despawn(
        surface_indices,
        particles,
        query,
        0.0);

    system::DespawnOperator<double>(system::DespawnType::Volume).despawn(
        volume_indices,
        particles,
        query,
        0.0);

    ASSERT_EQ(surface_indices.size(), 1u);
    EXPECT_EQ(surface_indices[0], 1);

    ASSERT_EQ(volume_indices.size(), 2u);
    EXPECT_EQ(volume_indices[0], 0);
    EXPECT_EQ(volume_indices[1], 1);
}
