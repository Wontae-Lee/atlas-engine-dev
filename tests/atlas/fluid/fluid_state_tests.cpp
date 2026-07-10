#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

namespace {

using atlas::DeviceBuffer;
using atlas::Float3;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidState;
using atlas::FluidTemperatureState;
using atlas::FluidVelocityState;
using atlas::tol;

/** Compares two Float3 values component-wise; never uses operator== on floats. */
void
expect_float3_near(const Float3& value, const Float3& expected) {
    EXPECT_NEAR(value.x, expected.x, tol);
    EXPECT_NEAR(value.y, expected.y, tol);
    EXPECT_NEAR(value.z, expected.z, tol);
}

}

TEST(FluidState, DefaultConstructedPositionStateIsEmpty) {
    const FluidPositionState state {};
    EXPECT_EQ(state.size(), std::size_t { 0 });
}

TEST(FluidState, PositionStateSizeMatchesCapacity) {
    const FluidPositionState state(16);
    EXPECT_EQ(state.size(), std::size_t { 16 });
    EXPECT_EQ(state.data().size(), std::size_t { 16 });
}

TEST(FluidState, VelocityStateSizeMatchesCapacity) {
    const FluidVelocityState state(8);
    EXPECT_EQ(state.size(), std::size_t { 8 });
    EXPECT_EQ(state.data().size(), std::size_t { 8 });
}

TEST(FluidState, SpeciesStateSizeMatchesCapacity) {
    const FluidSpeciesState state(4);
    EXPECT_EQ(state.size(), std::size_t { 4 });
    EXPECT_EQ(state.data().size(), std::size_t { 4 });
}

TEST(FluidState, TemperatureStateSizeMatchesCapacity) {
    const FluidTemperatureState state(5);
    EXPECT_EQ(state.size(), std::size_t { 5 });
    EXPECT_EQ(state.data().size(), std::size_t { 5 });
}

TEST(FluidState, PositionStateAdoptsBufferAndRoundTripsFields) {
    DeviceBuffer<Float3> buffer(3);
    buffer[0] = Float3(1.0f, 2.0f, 3.0f);
    buffer[1] = Float3(-4.0f, 5.5f, 6.0f);
    buffer[2] = Float3(0.0f, 0.0f, 9.0f);

    const FluidPositionState state(std::move(buffer));
    ASSERT_EQ(state.size(), std::size_t { 3 });

    // Copy each device element into a host value before inspecting it.
    const Float3 first  = state.data()[0];
    const Float3 second = state.data()[1];
    const Float3 third  = state.data()[2];
    expect_float3_near(first, Float3(1.0f, 2.0f, 3.0f));
    expect_float3_near(second, Float3(-4.0f, 5.5f, 6.0f));
    expect_float3_near(third, Float3(0.0f, 0.0f, 9.0f));
}

TEST(FluidState, SpeciesStateAdoptsBufferAndRoundTripsIndices) {
    DeviceBuffer<std::size_t> buffer(3);
    buffer[0] = std::size_t { 7 };
    buffer[1] = std::size_t { 0 };
    buffer[2] = std::size_t { 42 };

    const FluidSpeciesState state(std::move(buffer));
    ASSERT_EQ(state.size(), std::size_t { 3 });

    const std::size_t first  = state.data()[0];
    const std::size_t second = state.data()[1];
    const std::size_t third  = state.data()[2];
    EXPECT_EQ(first, std::size_t { 7 });
    EXPECT_EQ(second, std::size_t { 0 });
    EXPECT_EQ(third, std::size_t { 42 });
}

TEST(FluidState, TemperatureStateRoundTripsScalarField) {
    DeviceBuffer<float> buffer(2);
    buffer[0] = 273.15f;
    buffer[1] = 1000.0f;

    const FluidTemperatureState state(std::move(buffer));
    ASSERT_EQ(state.size(), std::size_t { 2 });

    const float first  = state.data()[0];
    const float second = state.data()[1];
    EXPECT_NEAR(first, 273.15f, tol);
    EXPECT_NEAR(second, 1000.0f, tol);
}

TEST(FluidState, ResetZeroesEveryElement) {
    FluidPositionState state(2);
    state.data()[0] = Float3(1.0f, 2.0f, 3.0f);
    state.data()[1] = Float3(4.0f, 5.0f, 6.0f);

    state.reset();

    const Float3 first  = state.data()[0];
    const Float3 second = state.data()[1];
    expect_float3_near(first, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_near(second, Float3(0.0f, 0.0f, 0.0f));
}

TEST(FluidState, LeafIsMoveOnly) {
    // The leaf owns a DeviceBuffer, so copy is deleted and only move is allowed.
    EXPECT_FALSE(std::is_copy_constructible_v<FluidPositionState>);
    EXPECT_FALSE(std::is_copy_assignable_v<FluidPositionState>);
    EXPECT_TRUE(std::is_move_constructible_v<FluidPositionState>);
    EXPECT_TRUE(std::is_move_assignable_v<FluidPositionState>);
}

TEST(FluidState, LeafDerivesFromFluidState) {
    EXPECT_TRUE((std::is_base_of_v<FluidState, FluidPositionState>));
    EXPECT_TRUE((std::is_base_of_v<FluidState, FluidVelocityState>));
    EXPECT_TRUE((std::is_base_of_v<FluidState, FluidSpeciesState>));
}
