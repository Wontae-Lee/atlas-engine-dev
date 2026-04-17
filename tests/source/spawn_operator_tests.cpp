#include "../utilities/tests_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/source/spawn_operator.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::GeometryOperator<T>
make_box_operator() {
    const auto box = atlas::geometry::Box<T>::builder()
                         .with_lower_corner(Vec3(-1, -1, -1))
                         .with_upper_corner(Vec3(1, 1, 1))
                         .build();
    return box.make_geometry_operator();
}

} // namespace

TEST(SpawnOperator, SurfaceOperatorDetectsSurfacePoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::fluid::SurfaceSpawnOperator<T>::spawn(geometry_operator, Vec3(1, 0, 0), 0.0f));
    EXPECT_FALSE(atlas::fluid::SurfaceSpawnOperator<T>::spawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
}

TEST(SpawnOperator, VolumeOperatorDetectsInteriorPoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::fluid::VolumeSpawnOperator<T>::spawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_FALSE(atlas::fluid::VolumeSpawnOperator<T>::spawn(geometry_operator, Vec3(3, 0, 0), 0.0f));
}

TEST(SpawnOperator, DefaultConstructorCreatesSurfaceVariant) {
    const atlas::fluid::SpawnOperator<T> spawn_operator;

    EXPECT_EQ(spawn_operator.type, atlas::fluid::SpawnType::Surface);
}

TEST(SpawnOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    const atlas::fluid::SpawnOperator<T> volume_operator(atlas::fluid::SpawnType::Volume);

    EXPECT_EQ(volume_operator.type, atlas::fluid::SpawnType::Volume);
}

TEST(SpawnOperator, CopyConstructionAndAssignmentPreserveType) {
    const atlas::fluid::SpawnOperator<T> original(atlas::fluid::SpawnType::Volume);
    const atlas::fluid::SpawnOperator<T> copied(original);

    atlas::fluid::SpawnOperator<T> assigned;
    assigned = original;

    EXPECT_EQ(copied.type, atlas::fluid::SpawnType::Volume);
    EXPECT_EQ(assigned.type, atlas::fluid::SpawnType::Volume);
}

TEST(SpawnOperator, SpawnDispatchesToActiveVariant) {
    const auto geometry_operator = make_box_operator();
    const atlas::fluid::SpawnOperator<T> surface_operator(atlas::fluid::SpawnType::Surface);
    const atlas::fluid::SpawnOperator<T> volume_operator(atlas::fluid::SpawnType::Volume);

    EXPECT_TRUE(surface_operator.spawn(geometry_operator, Vec3(1, 0, 0), 0.0f));
    EXPECT_FALSE(surface_operator.spawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_TRUE(volume_operator.spawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_FALSE(volume_operator.spawn(geometry_operator, Vec3(2, 0, 0), 0.0f));
}
