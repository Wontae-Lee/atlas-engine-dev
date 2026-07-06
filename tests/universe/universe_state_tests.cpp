#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

bool
expect_vec_near(const atlas::Vector3& a, const atlas::Vector3& b) {
    return std::abs(a.x - b.x) <= atlas::tol
        && std::abs(a.y - b.y) <= atlas::tol
        && std::abs(a.z - b.z) <= atlas::tol;
}

}

TEST(UniverseState, TemperatureStateStoresScalarData) {
    atlas::UniverseTemperatureState state(3);

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;

    const float t0 = state.data()[0];
    const float t1 = state.data()[1];

    EXPECT_EQ(state.size(), 3u);
    EXPECT_NEAR(t0, 100.0f, atlas::tol);
    EXPECT_NEAR(t1, 200.0f, atlas::tol);
}

TEST(UniverseState, BulkVelocityStateStoresVectorData) {
    atlas::UniverseBulkVelocityState state(2);

    state.data()[0] = atlas::Vector3(1.0f, 2.0f, 3.0f);

    const atlas::Vector3 v = state.data()[0];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(expect_vec_near(v, atlas::Vector3(1.0f, 2.0f, 3.0f)));
}

TEST(UniverseState, ThermalEnergyStateStoresScalarData) {
    atlas::UniverseThermalEnergyState state(2);

    state.data()[1] = 4.0f;

    const float e1 = state.data()[1];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(e1, 4.0f, atlas::tol);
}

TEST(UniverseState, NumberParticleStateStoresScalarData) {
    atlas::UniverseNumberParticleState state(2);

    state.data()[0] = 3.0f;

    const float n0 = state.data()[0];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(n0, 3.0f, atlas::tol);
}

TEST(UniverseState, CollisionRemainderStateStoresScalarData) {
    atlas::UniverseCollisionRemainderState state(2);

    state.data()[0] = 0.25f;
    state.data()[1] = 0.75f;

    const float r0 = state.data()[0];
    const float r1 = state.data()[1];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(r0, 0.25f, atlas::tol);
    EXPECT_NEAR(r1, 0.75f, atlas::tol);
}

TEST(UniverseState, CollisionCountStateStoresIntegerData) {
    atlas::UniverseCollisionCountState state(2);

    state.data()[0] = 2;
    state.data()[1] = 6;

    const int c0 = state.data()[0];
    const int c1 = state.data()[1];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_EQ(c0, 2);
    EXPECT_EQ(c1, 6);
}

TEST(UniverseState, MaterialRatioStateStoresContainerData) {
    atlas::UniverseMaterialRatioState<2> state(2);

    state.data()[0] = atlas::Container<float, 2>(0.25f, 0.75f);

    const atlas::Container<float, 2> ratio = state.data()[0];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(ratio[0], 0.25f, atlas::tol);
    EXPECT_NEAR(ratio[1], 0.75f, atlas::tol);
}

TEST(UniverseState, ResetClearsStoredValues) {
    atlas::UniverseTemperatureState state(2);

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;

    state.reset();

    const float t0 = state.data()[0];
    const float t1 = state.data()[1];

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(t0, 0.0f, atlas::tol);
    EXPECT_NEAR(t1, 0.0f, atlas::tol);
}
