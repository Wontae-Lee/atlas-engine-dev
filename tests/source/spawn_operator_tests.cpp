#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/source/spawn_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryOperator;
using atlas::Vector3F;
using atlas::SpawnOperator;
using atlas::SpawnType;
using atlas::SurfaceSpawnOperator;
using atlas::VolumeSpawnOperator;

GeometryOperator<float>
make_box_operator() {
    static const auto box = Box<float>::builder()
                                .with_lower_corner(Vector3F(-1, -1, -1))
                                .with_upper_corner(Vector3F(1, 1, 1))
                                .build();
    return box.make_device_geometry_view();
}

} // namespace

TEST(SpawnOperator, SurfaceOperatorDetectsSurfacePoints) {
    // Arrange: create a box geometry query operator.
    const auto geometry_operator = make_box_operator();

    // Assert: surface spawning accepts boundary points and rejects interior points.
    EXPECT_TRUE(SurfaceSpawnOperator<float>::spawn(geometry_operator, Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(SurfaceSpawnOperator<float>::spawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
}

TEST(SpawnOperator, VolumeOperatorDetectsInteriorPoints) {
    // Arrange: create a box geometry query operator.
    const auto geometry_operator = make_box_operator();

    // Assert: volume spawning accepts interior points and rejects exterior points.
    EXPECT_TRUE(VolumeSpawnOperator<float>::spawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_FALSE(VolumeSpawnOperator<float>::spawn(geometry_operator, Vector3F(3, 0, 0), 0.0f));
}

TEST(SpawnOperator, DefaultConstructorCreatesSurfaceVariant) {
    // Arrange: create a default runtime spawn operator.
    const SpawnOperator<float> spawn_operator;

    // Assert: the default variant is surface spawning.
    EXPECT_EQ(spawn_operator.type, SpawnType::Surface);
}

TEST(SpawnOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    // Arrange: create a runtime spawn operator with an explicit volume type.
    const SpawnOperator<float> volume_operator(SpawnType::Volume);

    // Assert: the requested variant is stored.
    EXPECT_EQ(volume_operator.type, SpawnType::Volume);
}

TEST(SpawnOperator, CopyConstructionAndAssignmentPreserveType) {
    // Arrange: create a runtime spawn operator with a non-default type.
    const SpawnOperator<float> original(SpawnType::Volume);

    // Act: copy-construct and copy-assign runtime spawn operators.
    const SpawnOperator<float> copied(original);

    SpawnOperator<float> assigned;
    assigned = original;

    // Assert: copied operators preserve the active type.
    EXPECT_EQ(copied.type, SpawnType::Volume);
    EXPECT_EQ(assigned.type, SpawnType::Volume);
}

TEST(SpawnOperator, SpawnDispatchesToActiveVariant) {
    // Arrange: create surface and volume runtime spawn operators.
    const auto geometry_operator = make_box_operator();
    const SpawnOperator<float> surface_operator(SpawnType::Surface);
    const SpawnOperator<float> volume_operator(SpawnType::Volume);

    // Assert: dispatch follows the active runtime variant.
    EXPECT_TRUE(surface_operator.spawn(geometry_operator, Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(surface_operator.spawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(volume_operator.spawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_FALSE(volume_operator.spawn(geometry_operator, Vector3F(2, 0, 0), 0.0f));
}
