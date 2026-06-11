#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeviceBuffer;
using atlas::FluidInternalEnergy;
using atlas::Vector3F;
using atlas::fluid::FluidActiveState;
using atlas::fluid::FluidInternalEnergyState;
using atlas::fluid::FluidPositionState;
using atlas::fluid::FluidSpeciesState;
using atlas::fluid::FluidTemperatureState;
using atlas::fluid::FluidVelocityState;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(FluidState, PositionStateStoresVectorData) {
    // Arrange: create a position state with three slots.
    FluidPositionState<float> state(3);

    // Act: write vector values into the state buffer.
    state.data()[0] = Vector3F(1, 2, 3);
    state.data()[1] = Vector3F(4, 5, 6);

    // Assert: the state preserves size and vector data.
    EXPECT_EQ(state.size(), 3u);
    EXPECT_TRUE(vec_near(state.data()[0], Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(state.data()[1], Vector3F(4, 5, 6), tol));
}

TEST(FluidState, VelocityStateCompactsKeptEntries) {
    // Arrange: create a velocity state and compact index list.
    FluidVelocityState<float> state(4);
    DeviceBuffer<std::size_t> compact_indices = { 0u, 2u };

    state.data()[0] = Vector3F(1, 0, 0);
    state.data()[1] = Vector3F(9, 9, 9);
    state.data()[2] = Vector3F(0, 1, 0);
    state.data()[3] = Vector3F(0, 0, 1);

    // Act: compact the kept entries into the active prefix.
    state.compact(compact_indices, 2);

    // Assert: kept entries preserve their relative order.
    EXPECT_TRUE(vec_near(state.data()[0], Vector3F(1, 0, 0), tol));
    EXPECT_TRUE(vec_near(state.data()[1], Vector3F(0, 1, 0), tol));
}

TEST(FluidState, SpeciesStateCompactsKeptEntries) {
    // Arrange: create a species state and compact index list.
    FluidSpeciesState<float> state(4);
    DeviceBuffer<std::size_t> compact_indices = { 1u, 3u };

    state.data()[0] = 10u;
    state.data()[1] = 20u;
    state.data()[2] = 30u;
    state.data()[3] = 40u;

    // Act: compact the kept entries into the active prefix.
    state.compact(compact_indices, 2);

    // Assert: kept species ids preserve their relative order.
    EXPECT_EQ(state.data()[0], 20u);
    EXPECT_EQ(state.data()[1], 40u);
}

TEST(FluidState, ActiveStateCompactsKeptEntries) {
    // Arrange: create an active state and compact index list.
    FluidActiveState<float> state(4);
    DeviceBuffer<std::size_t> compact_indices = { 0u, 3u };

    state.data()[0] = 1;
    state.data()[1] = 0;
    state.data()[2] = 0;
    state.data()[3] = 1;

    // Act: compact the kept entries into the active prefix.
    state.compact(compact_indices, 2);

    // Assert: kept active flags preserve their relative order.
    EXPECT_EQ(state.data()[0], 1);
    EXPECT_EQ(state.data()[1], 1);
}

TEST(FluidState, TemperatureStateCompactsKeptEntries) {
    // Arrange: create a temperature state and compact index list.
    FluidTemperatureState<float> state(4);
    DeviceBuffer<std::size_t> compact_indices = { 2u, 3u };

    state.data()[0] = 100.0f;
    state.data()[1] = 200.0f;
    state.data()[2] = 300.0f;
    state.data()[3] = 400.0f;

    // Act: compact the kept entries into the active prefix.
    state.compact(compact_indices, 2);

    // Assert: kept temperatures preserve their relative order.
    EXPECT_NEAR(state.data()[0], 300.0f, tol);
    EXPECT_NEAR(state.data()[1], 400.0f, tol);
}

TEST(FluidState, InternalEnergyStateCompactsKeptEntries) {
    // Arrange: create an internal-energy state and compact index list.
    FluidInternalEnergyState<float> state(4);
    DeviceBuffer<std::size_t> compact_indices = { 1u, 3u };

    state.data()[0] = FluidInternalEnergy<float> { 1.0f, 2.0f, 3.0f };
    state.data()[1] = FluidInternalEnergy<float> { 4.0f, 5.0f, 6.0f };
    state.data()[2] = FluidInternalEnergy<float> { 7.0f, 8.0f, 9.0f };
    state.data()[3] = FluidInternalEnergy<float> { 10.0f, 11.0f, 12.0f };

    // Act: compact the kept entries into the active prefix.
    state.compact(compact_indices, 2);

    // Assert: kept internal-energy values preserve their relative order.
    FluidInternalEnergy<float> actual[2] {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(state.data().data()), actual, 2);
    EXPECT_NEAR(actual[0].translational, 4.0f, tol);
    EXPECT_NEAR(actual[0].rotational, 5.0f, tol);
    EXPECT_NEAR(actual[0].vibrational, 6.0f, tol);
    EXPECT_NEAR(actual[1].translational, 10.0f, tol);
    EXPECT_NEAR(actual[1].rotational, 11.0f, tol);
    EXPECT_NEAR(actual[1].vibrational, 12.0f, tol);
}

TEST(FluidState, CompactWithZeroKeptLeavesExistingDataAccessible) {
    // Arrange: create a temperature state and an empty compact index list.
    FluidTemperatureState<float> state(2);
    DeviceBuffer<std::size_t> compact_indices;

    state.data()[0] = 123.0f;
    state.data()[1] = 456.0f;

    // Act and assert: compacting zero entries is safe and leaves data accessible.
    EXPECT_NO_THROW(state.compact(compact_indices, 0));
    EXPECT_NEAR(state.data()[0], 123.0f, tol);
    EXPECT_NEAR(state.data()[1], 456.0f, tol);
}
