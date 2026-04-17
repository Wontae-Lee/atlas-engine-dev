#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid_state.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(FluidState, PositionStateStoresVectorData) {
    atlas::fluid::FluidPositionState<T> state(3);

    state.data()[0] = Vec3(1, 2, 3);
    state.data()[1] = Vec3(4, 5, 6);

    EXPECT_EQ(state.size(), 3u);
    EXPECT_TRUE(atlas::test::vec_near(state.data()[0], Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(state.data()[1], Vec3(4, 5, 6), kEps));
}

TEST(FluidState, VelocityStateCompactsKeptEntries) {
    atlas::fluid::FluidVelocityState<T> state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 0u, 2u };

    state.data()[0] = Vec3(1, 0, 0);
    state.data()[1] = Vec3(9, 9, 9);
    state.data()[2] = Vec3(0, 1, 0);
    state.data()[3] = Vec3(0, 0, 1);

    state.compact(compact_indices, 2);

    EXPECT_TRUE(atlas::test::vec_near(state.data()[0], Vec3(1, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(state.data()[1], Vec3(0, 1, 0), kEps));
}

TEST(FluidState, SpeciesStateCompactsKeptEntries) {
    atlas::fluid::FluidSpeciesState<T> state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 1u, 3u };

    state.data()[0] = 10u;
    state.data()[1] = 20u;
    state.data()[2] = 30u;
    state.data()[3] = 40u;

    state.compact(compact_indices, 2);

    EXPECT_EQ(state.data()[0], 20u);
    EXPECT_EQ(state.data()[1], 40u);
}

TEST(FluidState, ActiveStateCompactsKeptEntries) {
    atlas::fluid::FluidActiveState<T> state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 0u, 3u };

    state.data()[0] = 1;
    state.data()[1] = 0;
    state.data()[2] = 0;
    state.data()[3] = 1;

    state.compact(compact_indices, 2);

    EXPECT_EQ(state.data()[0], 1);
    EXPECT_EQ(state.data()[1], 1);
}

TEST(FluidState, TemperatureStateCompactsKeptEntries) {
    atlas::fluid::FluidTemperatureState<T> state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 2u, 3u };

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;
    state.data()[2] = 300.0f;
    state.data()[3] = 400.0f;

    state.compact(compact_indices, 2);

    EXPECT_NEAR(state.data()[0], 300.0f, kEps);
    EXPECT_NEAR(state.data()[1], 400.0f, kEps);
}

TEST(FluidState, CompactWithZeroKeptLeavesExistingDataAccessible) {
    atlas::fluid::FluidTemperatureState<T> state(2);
    atlas::DeviceBuffer<std::size_t> compact_indices;

    state.data()[0] = 123.0f;
    state.data()[1] = 456.0f;

    EXPECT_NO_THROW(state.compact(compact_indices, 0));
    EXPECT_NEAR(state.data()[0], 123.0f, kEps);
    EXPECT_NEAR(state.data()[1], 456.0f, kEps);
}
