#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(VolumeSpawnOperator, SpawnCreatesVolumeGridPointsFromBoxGeometryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);
    const system::VolumeSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

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

TEST(VolumeSpawnOperator, SpawnAcceptsInteriorSamplesUsingExplicitTolerance) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);
    const system::VolumeSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return atlas::system::VolumeSpawnOperator<double>::spawn(query_op, sample, tol);
        });

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromSphereGeometryOperator) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);
    const system::VolumeSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        0.5,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromCylinderGeometryOperator) {
    const geometry::Cylinder<double> cylinder = test::make_cylinder();
    const auto query                          = test::make_cylinder_geometry_operator(cylinder);
    const system::VolumeSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_inside(particle, 0.0));
    }
}

TEST(VolumeSpawnOperator, SpawnReturnsEmptyBufferForInvalidGeometryOperator) {
    const geometry::GeometryOperator<double> query;
    const system::VolumeSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles(4, Vector3<double>(1.0, 2.0, 3.0));

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

    EXPECT_TRUE(particles.empty());
}