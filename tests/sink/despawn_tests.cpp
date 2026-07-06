#include <atlas/sink/despawn.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

namespace {

using atlas::Box;
using atlas::Despawn;
using atlas::DespawnType;
using atlas::Geometry;
using atlas::Float3;

Geometry
make_box_operator() {
    static const auto box = Box::builder()
                                .with_lower_corner(Float3(-1.0f, -1.0f, -1.0f))
                                .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
                                .build();
    return Geometry(box);
}

}

TEST(Despawn, DefaultConstructorCreatesSurfaceVariant) {
    const Despawn despawn_operator;

    EXPECT_EQ(despawn_operator.type, DespawnType::surface);
}

TEST(Despawn, ExplicitTypeConstructorSelectsRequestedVariant) {
    const Despawn volume_operator(DespawnType::volume);
    const Despawn tracing_operator(DespawnType::tracing);

    EXPECT_EQ(volume_operator.type, DespawnType::volume);
    EXPECT_EQ(tracing_operator.type, DespawnType::tracing);
}

TEST(Despawn, CopyConstructionAndAssignmentPreserveType) {
    const Despawn original(DespawnType::volume);

    const Despawn copied(original);

    Despawn assigned;
    assigned = original;

    EXPECT_EQ(copied.type, DespawnType::volume);
    EXPECT_EQ(assigned.type, DespawnType::volume);
}

TEST(Despawn, DespawnDispatchesToActiveVariant) {
    const auto geometry_operator = make_box_operator();
    const Despawn surface_operator(DespawnType::surface);
    const Despawn volume_operator(DespawnType::volume);
    const Despawn tracing_operator(DespawnType::tracing);

    EXPECT_TRUE(surface_operator.despawn(geometry_operator, Float3(1.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(surface_operator.despawn(geometry_operator, Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(volume_operator.despawn(geometry_operator, Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(volume_operator.despawn(geometry_operator, Float3(2.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(tracing_operator.despawn(
        geometry_operator, Float3(-2.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f), 1.0f));
    EXPECT_FALSE(tracing_operator.despawn(
        geometry_operator, Float3(-2.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f), 0.0f));
}
