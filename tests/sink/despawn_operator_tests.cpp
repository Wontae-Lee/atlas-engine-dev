#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sink/despawn_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryOperator;
using atlas::Vector3F;
using atlas::fluid::DespawnOperator;
using atlas::fluid::DespawnType;
using atlas::fluid::SurfaceDespawnOperator;
using atlas::fluid::VolumeDespawnOperator;

GeometryOperator<float>
make_box_operator() {
    static const auto box = Box<float>::builder()
                                .with_lower_corner(Vector3F(-1, -1, -1))
                                .with_upper_corner(Vector3F(1, 1, 1))
                                .build();
    return box.make_geometry_operator();
}

} // namespace

TEST(DespawnOperator, SurfaceOperatorDetectsSurfacePoints) {
    // Arrange: create a box geometry query operator.
    const auto geometry_operator = make_box_operator();

    // Assert: surface despawning accepts boundary points and rejects interior points.
    EXPECT_TRUE(SurfaceDespawnOperator<float>::despawn(geometry_operator, Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(SurfaceDespawnOperator<float>::despawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
}

TEST(DespawnOperator, VolumeOperatorDetectsInteriorPoints) {
    // Arrange: create a box geometry query operator.
    const auto geometry_operator = make_box_operator();

    // Assert: volume despawning accepts interior points and rejects exterior points.
    EXPECT_TRUE(VolumeDespawnOperator<float>::despawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_FALSE(VolumeDespawnOperator<float>::despawn(geometry_operator, Vector3F(3, 0, 0), 0.0f));
}

TEST(DespawnOperator, DefaultConstructorCreatesSurfaceVariant) {
    // Arrange: create a default runtime despawn operator.
    const DespawnOperator<float> despawn_operator;

    // Assert: the default variant is surface despawning.
    EXPECT_EQ(despawn_operator.type, DespawnType::Surface);
}

TEST(DespawnOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    // Arrange: create a runtime despawn operator with an explicit volume type.
    const DespawnOperator<float> volume_operator(DespawnType::Volume);

    // Assert: the requested variant is stored.
    EXPECT_EQ(volume_operator.type, DespawnType::Volume);
}

TEST(DespawnOperator, CopyConstructionAndAssignmentPreserveType) {
    // Arrange: create a runtime despawn operator with a non-default type.
    const DespawnOperator<float> original(DespawnType::Volume);

    // Act: copy-construct and copy-assign runtime despawn operators.
    const DespawnOperator<float> copied(original);

    DespawnOperator<float> assigned;
    assigned = original;

    // Assert: copied operators preserve the active type.
    EXPECT_EQ(copied.type, DespawnType::Volume);
    EXPECT_EQ(assigned.type, DespawnType::Volume);
}

TEST(DespawnOperator, DespawnDispatchesToActiveVariant) {
    // Arrange: create surface and volume runtime despawn operators.
    const auto geometry_operator = make_box_operator();
    const DespawnOperator<float> surface_operator(DespawnType::Surface);
    const DespawnOperator<float> volume_operator(DespawnType::Volume);

    // Assert: dispatch follows the active runtime variant.
    EXPECT_TRUE(surface_operator.despawn(geometry_operator, Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(surface_operator.despawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(volume_operator.despawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
    EXPECT_FALSE(volume_operator.despawn(geometry_operator, Vector3F(2, 0, 0), 0.0f));
}
