#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SurfaceDespawnOperator, DespawnReturnsTrueForSurfaceParticleFromBoxGeometryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);

    EXPECT_TRUE(system::SurfaceDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(1.0, 0.0, 0.0),
        0.0));
}

TEST(SurfaceDespawnOperator, DespawnReturnsFalseForInteriorParticleFromSphereGeometryOperator) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);

    EXPECT_FALSE(system::SurfaceDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(0.0, 0.0, 0.0),
        0.0));
}

TEST(SurfaceDespawnOperator, DespawnUsesToleranceBandForNearSurfaceParticle) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);

    EXPECT_TRUE(system::SurfaceDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(1.1, 0.0, 0.0),
        0.15));
}

TEST(SurfaceDespawnOperator, DespawnReturnsFalseForInvalidGeometryOperator) {
    const geometry::GeometryOperator<double> query;

    EXPECT_FALSE(system::SurfaceDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(1.0, 0.0, 0.0),
        0.0));
}
