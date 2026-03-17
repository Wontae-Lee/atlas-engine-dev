#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(VolumeSpawnOperator, SpawnCreatesVolumeGridPointsFromBoxQueryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_query_operator(box);

    DeviceBuffer<Vector3<double>> particles;

    system::VolumeSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    ASSERT_EQ(particles.size(), 27u);

    bool found_center = false;
    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));

        if (test::vec_near(particle, Vector3<double>(0.0, 0.0, 0.0), eps)) {
            found_center = true;
        }
    }

    EXPECT_TRUE(found_center);
}

TEST(VolumeSpawnOperator, SpawnClampsNegativeToleranceToZero) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_query_operator(sphere);

    DeviceBuffer<Vector3<double>> particles;

    system::VolumeSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        -1.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromSphereQueryOperator) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_query_operator(sphere);

    DeviceBuffer<Vector3<double>> particles;

    system::VolumeSpawnOperator<double> {}.spawn(
        particles,
        query,
        0.5,
        0.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromCylinderQueryOperator) {
    const geometry::Cylinder<double> cylinder = test::make_cylinder();
    const auto query                          = test::make_cylinder_query_operator(cylinder);

    DeviceBuffer<Vector3<double>> particles;

    system::VolumeSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnReturnsEmptyBufferForInvalidQueryOperator) {
    const geometry::QueryOperator<double> query;

    DeviceBuffer<Vector3<double>> particles(4, Vector3<double>(1.0, 2.0, 3.0));

    system::VolumeSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    EXPECT_TRUE(particles.empty());
}
