#include "../utilities/tests_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sink/despawn_operator.h>

#include <testkit/testkit.h>

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

TEST(DespawnOperator, SurfaceOperatorDetectsSurfacePoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::fluid::SurfaceDespawnOperator<T>::despawn(geometry_operator, Vec3(1, 0, 0), 0.0f));
    EXPECT_FALSE(atlas::fluid::SurfaceDespawnOperator<T>::despawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
}

TEST(DespawnOperator, VolumeOperatorDetectsInteriorPoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::fluid::VolumeDespawnOperator<T>::despawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_FALSE(atlas::fluid::VolumeDespawnOperator<T>::despawn(geometry_operator, Vec3(3, 0, 0), 0.0f));
}

TEST(DespawnOperator, DefaultConstructorCreatesSurfaceVariant) {
    const atlas::fluid::DespawnOperator<T> despawn_operator;

    EXPECT_EQ(despawn_operator.type, atlas::fluid::DespawnType::Surface);
}

TEST(DespawnOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    const atlas::fluid::DespawnOperator<T> volume_operator(atlas::fluid::DespawnType::Volume);

    EXPECT_EQ(volume_operator.type, atlas::fluid::DespawnType::Volume);
}

TEST(DespawnOperator, CopyConstructionAndAssignmentPreserveType) {
    const atlas::fluid::DespawnOperator<T> original(atlas::fluid::DespawnType::Volume);
    const atlas::fluid::DespawnOperator<T> copied(original);

    atlas::fluid::DespawnOperator<T> assigned;
    assigned = original;

    EXPECT_EQ(copied.type, atlas::fluid::DespawnType::Volume);
    EXPECT_EQ(assigned.type, atlas::fluid::DespawnType::Volume);
}

TEST(DespawnOperator, DespawnDispatchesToActiveVariant) {
    const auto geometry_operator = make_box_operator();
    const atlas::fluid::DespawnOperator<T> surface_operator(atlas::fluid::DespawnType::Surface);
    const atlas::fluid::DespawnOperator<T> volume_operator(atlas::fluid::DespawnType::Volume);

    EXPECT_TRUE(surface_operator.despawn(geometry_operator, Vec3(1, 0, 0), 0.0f));
    EXPECT_FALSE(surface_operator.despawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_TRUE(volume_operator.despawn(geometry_operator, Vec3(0, 0, 0), 0.0f));
    EXPECT_FALSE(volume_operator.despawn(geometry_operator, Vec3(2, 0, 0), 0.0f));
}
