#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SurfaceSpawnOperator, SpawnCreatesSurfaceGridPointsFromBoxQueryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_query_operator(box);

    DeviceBuffer<Vector3<double>> particles;

    system::SurfaceSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    ASSERT_EQ(particles.size(), 26u);

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
        EXPECT_FALSE(test::vec_near(particle, Vector3<double>(0.0, 0.0, 0.0), eps));
    }
}

TEST(SurfaceSpawnOperator, SpawnUsesDefaultToleranceBandWhenNegativeToleranceIsGiven) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_query_operator(sphere);

    DeviceBuffer<Vector3<double>> particles;

    system::SurfaceSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        -1.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.5));
    }
}

TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromCylinderQueryOperator) {
    const geometry::Cylinder<double> cylinder = test::make_cylinder();
    const auto query                          = test::make_cylinder_query_operator(cylinder);

    DeviceBuffer<Vector3<double>> particles;

    system::SurfaceSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
    }
}

TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromTriangleQueryOperator) {
    const geometry::Triangle<double> triangle = test::make_triangle();
    const auto query                          = test::make_triangle_query_operator(triangle);

    DeviceBuffer<Vector3<double>> particles;

    system::SurfaceSpawnOperator<double> {}.spawn(
        particles,
        query,
        0.5,
        0.0);

    ASSERT_FALSE(particles.empty());

    for (const auto& particle : particles) {
        EXPECT_TRUE(test::is_finite_vec(particle));
        EXPECT_TRUE(query.is_on_surface(particle, 0.0));
    }
}

TEST(SurfaceSpawnOperator, SpawnReturnsEmptyBufferForInvalidQueryOperator) {
    const geometry::QueryOperator<double> query;

    DeviceBuffer<Vector3<double>> particles(4, Vector3<double>(1.0, 2.0, 3.0));

    system::SurfaceSpawnOperator<double> {}.spawn(
        particles,
        query,
        1.0,
        0.0);

    EXPECT_TRUE(particles.empty());
}
