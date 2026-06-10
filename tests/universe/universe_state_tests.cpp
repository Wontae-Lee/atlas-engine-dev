#include "../utilities/test_utils.h"

#include <atlas/universe/universe_state.h>

#include <testkit/testkit.h>

namespace {

using atlas::Vector;
using atlas::Vector3F;
using atlas::eps;
using atlas::test::vec_near;
using atlas::universe::UniverseBulkVelocityState;
using atlas::universe::UniverseCollisionRemainderState;
using atlas::universe::UniverseMaterialRatioState;
using atlas::universe::UniverseNumberParticleState;
using atlas::universe::UniverseTemperatureState;
using atlas::universe::UniverseThermalEnergyState;

} // namespace

TEST(UniverseState, TemperatureStateStoresScalarData) {
    // Arrange: create a scalar temperature state with fixed storage.
    UniverseTemperatureState<float> state(3);

    // Act: write representative temperature values.
    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;

    // Assert: the state keeps its configured size and scalar values.
    EXPECT_EQ(state.size(), 3u);
    EXPECT_NEAR(state.data()[0], 100.0f, eps);
    EXPECT_NEAR(state.data()[1], 200.0f, eps);
}

TEST(UniverseState, BulkVelocityStateStoresVectorData) {
    // Arrange: create a vector-valued bulk velocity state.
    UniverseBulkVelocityState<float> state(2);

    // Act: write a representative velocity vector.
    state.data()[0] = Vector3F(1, 2, 3);

    // Assert: the vector value is stored without changing state size.
    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(vec_near(state.data()[0], Vector3F(1, 2, 3), eps));
}

TEST(UniverseState, ThermalEnergyStateStoresScalarData) {
    // Arrange: create a scalar thermal energy state.
    UniverseThermalEnergyState<float> state(2);

    // Act: write one thermal energy sample.
    state.data()[1] = 4.0f;

    // Assert: the scalar value is available at the assigned index.
    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(state.data()[1], 4.0f, eps);
}

TEST(UniverseState, NumberParticleStateStoresScalarData) {
    // Arrange: create a scalar particle-count state.
    UniverseNumberParticleState<float> state(2);

    // Act: write one particle-count sample.
    state.data()[0] = 3.0f;

    // Assert: the scalar value is available at the assigned index.
    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(state.data()[0], 3.0f, eps);
}

TEST(UniverseState, CollisionRemainderStateStoresScalarData) {
    UniverseCollisionRemainderState<float> state(2);

    state.data()[0] = 0.25f;
    state.data()[1] = 0.75f;

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(state.data()[0], 0.25f, eps);
    EXPECT_NEAR(state.data()[1], 0.75f, eps);
}

TEST(UniverseState, MaterialRatioStateStoresVectorData) {
    // Arrange: create a two-material ratio state.
    UniverseMaterialRatioState<float, 2> state(2);

    // Act: write a representative material ratio vector.
    state.data()[0] = Vector<float, 2>(0.25f, 0.75f);

    // Assert: the vector ratio is stored without changing state size.
    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(vec_near(state.data()[0], Vector<float, 2>(0.25f, 0.75f), eps));
}
