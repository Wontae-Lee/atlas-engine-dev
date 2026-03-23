#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(VolumeDespawnOperator, DespawnReturnsTrueForInteriorParticleFromSphereGeometryOperator) {
    const geometry::Sphere<double> sphere = test::make_sphere();
    const auto query                      = test::make_sphere_geometry_operator(sphere);

    EXPECT_TRUE(system::VolumeDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(0.25, 0.25, 0.25),
        0.0));
}

TEST(VolumeDespawnOperator, DespawnReturnsFalseForExteriorParticleFromBoxGeometryOperator) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);

    EXPECT_FALSE(system::VolumeDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(2.0, 0.0, 0.0),
        0.0));
}

TEST(VolumeDespawnOperator, DespawnUsesToleranceBandForNearInteriorParticle) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);

    EXPECT_TRUE(system::VolumeDespawnOperator<double> {}.despawn(
        query,
        Vector3<double>(1.1, 0.0, 0.0),
        0.15));
}

TEST(DespawnOperator, DespawnDispatchesByRuntimeTypeForSingleParticle) {
    const geometry::Box<double> box = test::make_box();
    const auto query                = test::make_box_geometry_operator(box);
    const auto particle             = Vector3<double>(1.0, 0.0, 0.0);

    EXPECT_TRUE(system::DespawnOperator<double>(system::DespawnType::Surface).despawn(query, particle, 0.0));

    EXPECT_TRUE(system::DespawnOperator<double>(system::DespawnType::Volume).despawn(query, particle, 0.0));
}