#include <atlas/fluid/fluid_state.h>

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

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

TEST(FluidState, PositionStateStoresVectorData) {
    atlas::FluidPositionState state(3);

    state.data()[0] = atlas::Vector3(1.0f, 2.0f, 3.0f);
    state.data()[1] = atlas::Vector3(4.0f, 5.0f, 6.0f);

    const atlas::Vector3 p0 = state.data()[0];
    const atlas::Vector3 p1 = state.data()[1];

    EXPECT_EQ(state.size(), 3u);
    EXPECT_TRUE(expect_vec_near(p0, atlas::Vector3(1.0f, 2.0f, 3.0f)));
    EXPECT_TRUE(expect_vec_near(p1, atlas::Vector3(4.0f, 5.0f, 6.0f)));
}

TEST(FluidState, VelocityStateCompactsKeptEntries) {
    atlas::FluidVelocityState state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 0u, 2u };

    state.data()[0] = atlas::Vector3(1.0f, 0.0f, 0.0f);
    state.data()[1] = atlas::Vector3(9.0f, 9.0f, 9.0f);
    state.data()[2] = atlas::Vector3(0.0f, 1.0f, 0.0f);
    state.data()[3] = atlas::Vector3(0.0f, 0.0f, 1.0f);

    state.compact(compact_indices, 2);

    const atlas::Vector3 v0 = state.data()[0];
    const atlas::Vector3 v1 = state.data()[1];

    EXPECT_TRUE(expect_vec_near(v0, atlas::Vector3(1.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(expect_vec_near(v1, atlas::Vector3(0.0f, 1.0f, 0.0f)));
}

TEST(FluidState, SpeciesStateCompactsKeptEntries) {
    atlas::FluidSpeciesState state(4);
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
    atlas::FluidActiveState state(4);
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
    atlas::FluidTemperatureState state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 2u, 3u };

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;
    state.data()[2] = 300.0f;
    state.data()[3] = 400.0f;

    state.compact(compact_indices, 2);

    const float t0 = state.data()[0];
    const float t1 = state.data()[1];

    EXPECT_NEAR(t0, 300.0f, atlas::tol);
    EXPECT_NEAR(t1, 400.0f, atlas::tol);
}

TEST(FluidState, InternalEnergyStateCompactsKeptEntries) {
    atlas::FluidInternalEnergyState state(4);
    atlas::DeviceBuffer<std::size_t> compact_indices = { 1u, 3u };

    state.data()[0] = atlas::FluidInternalEnergy { 1.0f, 2.0f, 3.0f };
    state.data()[1] = atlas::FluidInternalEnergy { 4.0f, 5.0f, 6.0f };
    state.data()[2] = atlas::FluidInternalEnergy { 7.0f, 8.0f, 9.0f };
    state.data()[3] = atlas::FluidInternalEnergy { 10.0f, 11.0f, 12.0f };

    state.compact(compact_indices, 2);

    atlas::FluidInternalEnergy actual[2] {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(state.data().data()), actual, 2);

    EXPECT_NEAR(actual[0].translational, 4.0f, atlas::tol);
    EXPECT_NEAR(actual[0].rotational, 5.0f, atlas::tol);
    EXPECT_NEAR(actual[0].vibrational, 6.0f, atlas::tol);
    EXPECT_NEAR(actual[1].translational, 10.0f, atlas::tol);
    EXPECT_NEAR(actual[1].rotational, 11.0f, atlas::tol);
    EXPECT_NEAR(actual[1].vibrational, 12.0f, atlas::tol);
}

TEST(FluidState, CompactWithZeroKeptLeavesExistingDataAccessible) {
    atlas::FluidTemperatureState state(2);
    atlas::DeviceBuffer<std::size_t> compact_indices;

    state.data()[0] = 123.0f;
    state.data()[1] = 456.0f;

    EXPECT_NO_THROW(state.compact(compact_indices, 0));

    const float t0 = state.data()[0];
    const float t1 = state.data()[1];

    EXPECT_NEAR(t0, 123.0f, atlas::tol);
    EXPECT_NEAR(t1, 456.0f, atlas::tol);
}
