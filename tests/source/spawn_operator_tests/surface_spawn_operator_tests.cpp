#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SurfaceSpawnOperator, SpawnCreatesSurfaceGridPointsFromBoxGeometryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);
    const system::SurfaceSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.0,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

    ASSERT_EQ(particles.size(), 26u);

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
        EXPECT_FALSE(test::vec_near(particle, Vector3<double>(0.0, 0.0, 0.0), eps));
    }
}

TEST(SurfaceSpawnOperator, SpawnAcceptsSurfaceSamplesUsingExplicitToleranceBand) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);
    const system::SurfaceSpawnOperator<double> spawn_operator {};

    DeviceBuffer<Vector3<double>> particles;

    atlas::sampling::sample_spawn_grid(
        particles,
        query,
        1.0,
        0.5,
        [=] ATLAS_ALL_DEVICE(const auto& query_op, const auto& sample, const double tol) {
            return spawn_operator.spawn(query_op, sample, tol);
        });

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.5));
    }
}

TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromCylinderGeometryOperator) {
    const geometry::Cylinder<double> cylinder = test::make_cylinder();
    const auto query                          = test::make_cylinder_geometry_operator(cylinder);
    const system::SurfaceSpawnOperator<double> spawn_operator {};

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
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
    }
}

TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromTriangleGeometryOperator) {
    const geometry::Triangle<double> triangle = test::make_triangle();
    const auto query                          = test::make_triangle_geometry_operator(triangle);
    const system::SurfaceSpawnOperator<double> spawn_operator {};

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
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
    }
}

TEST(SurfaceSpawnOperator, SpawnReturnsEmptyBufferForInvalidGeometryOperator) {
    const geometry::GeometryOperator<double> query;
    const system::SurfaceSpawnOperator<double> spawn_operator {};

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