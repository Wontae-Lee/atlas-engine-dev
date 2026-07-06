#include <atlas/source/spawn.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

namespace {

atlas::Geometry
make_box_operator() {
    static const auto box = atlas::Box::builder()
                                .with_lower_corner(atlas::Vector3(-1.0f, -1.0f, -1.0f))
                                .with_upper_corner(atlas::Vector3(1.0f, 1.0f, 1.0f))
                                .build();
    return atlas::Geometry(box);
}

}

TEST(Spawn, SurfaceOperatorDetectsSurfacePoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::SurfaceSpawn::spawn(geometry_operator, atlas::Vector3(1.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(atlas::SurfaceSpawn::spawn(geometry_operator, atlas::Vector3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Spawn, VolumeOperatorDetectsInteriorPoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(atlas::VolumeSpawn::spawn(geometry_operator, atlas::Vector3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(atlas::VolumeSpawn::spawn(geometry_operator, atlas::Vector3(3.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Spawn, DefaultConstructorCreatesSurfaceVariant) {
    const atlas::Spawn spawn_operator;

    EXPECT_EQ(spawn_operator.type, atlas::SpawnType::surface);
}

TEST(Spawn, ExplicitTypeConstructorSelectsRequestedVariant) {
    const atlas::Spawn volume_operator(atlas::SpawnType::volume);

    EXPECT_EQ(volume_operator.type, atlas::SpawnType::volume);
}

TEST(Spawn, PayloadConstructorSelectsMatchingVariant) {
    const atlas::Spawn surface_operator(atlas::SurfaceSpawn {});
    const atlas::Spawn volume_operator(atlas::VolumeSpawn {});

    EXPECT_EQ(surface_operator.type, atlas::SpawnType::surface);
    EXPECT_EQ(volume_operator.type, atlas::SpawnType::volume);
}

TEST(Spawn, CopyConstructionAndAssignmentPreserveType) {
    const atlas::Spawn original(atlas::SpawnType::volume);

    const atlas::Spawn copied(original);

    atlas::Spawn assigned;
    assigned = original;

    EXPECT_EQ(copied.type, atlas::SpawnType::volume);
    EXPECT_EQ(assigned.type, atlas::SpawnType::volume);
}

TEST(Spawn, SpawnDispatchesToActiveVariant) {
    const auto geometry_operator = make_box_operator();
    const atlas::Spawn surface_operator(atlas::SpawnType::surface);
    const atlas::Spawn volume_operator(atlas::SpawnType::volume);

    EXPECT_TRUE(surface_operator.spawn(geometry_operator, atlas::Vector3(1.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(surface_operator.spawn(geometry_operator, atlas::Vector3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(volume_operator.spawn(geometry_operator, atlas::Vector3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(volume_operator.spawn(geometry_operator, atlas::Vector3(2.0f, 0.0f, 0.0f), 0.0f));
}
