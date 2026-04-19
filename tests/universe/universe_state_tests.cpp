#include "../utilities/tests_utils.h"

#include <atlas/universe/universe_state.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(UniverseState, TemperatureStateStoresScalarData) {
    atlas::universe::UniverseTemperatureState<T> state(3);

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;

    EXPECT_EQ(state.size(), 3u);
    EXPECT_NEAR(state.data()[0], 100.0f, kEps);
    EXPECT_NEAR(state.data()[1], 200.0f, kEps);
}

TEST(UniverseState, BulkVelocityStateStoresVectorData) {
    atlas::universe::UniverseBulkVelocityState<T> state(2);

    state.data()[0] = Vec3(1, 2, 3);

    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(atlas::test::vec_near(state.data()[0], Vec3(1, 2, 3), kEps));
}

TEST(UniverseState, ThermalEnergyStateStoresScalarData) {
    atlas::universe::UniverseThermalEnergyState<T> state(2);

    state.data()[1] = 4.0f;

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(state.data()[1], 4.0f, kEps);
}

TEST(UniverseState, NumberParticleStateStoresScalarData) {
    atlas::universe::UniverseNumberParticleState<T> state(2);

    state.data()[0] = 3.0f;

    EXPECT_EQ(state.size(), 2u);
    EXPECT_NEAR(state.data()[0], 3.0f, kEps);
}

TEST(UniverseState, MaterialRatioStateStoresVectorData) {
    atlas::universe::UniverseMaterialRatioState<T, 2> state(2);

    state.data()[0] = atlas::Vector<T, 2>(T(0.25), T(0.75));

    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(atlas::test::vec_near(state.data()[0], atlas::Vector<T, 2>(T(0.25), T(0.75)), kEps));
}
