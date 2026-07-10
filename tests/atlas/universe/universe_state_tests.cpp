#include <atlas/universe/universe_state.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace {

using atlas::DeviceBuffer;
using atlas::Float3;
using atlas::tol;
using atlas::UniverseBulkVelocityState;
using atlas::UniverseCollisionCountState;
using atlas::UniverseTemperatureState;

// Fill a device buffer element by element. Single-element assignment through the
// container's subscript is a plain host<->device copy under either backend; it
// never dereferences a raw device element, so it stays portable.
template <typename T>
DeviceBuffer<T>
make_buffer(std::initializer_list<T> values) {
    DeviceBuffer<T> buffer(values.size());
    std::size_t index = 0;
    for (const T& value : values) {
        buffer[index++] = value;
    }
    return buffer;
}

}

TEST(UniverseState, SizedConstructorZeroInitializesFloatField) {
    const UniverseTemperatureState state(std::size_t { 3 });

    ASSERT_EQ(state.size(), std::size_t { 3 });
    for (std::size_t i = 0; i < state.size(); ++i) {
        // Copy the device element into a host local before comparing.
        const float value = state.data()[i];
        EXPECT_NEAR(value, 0.0f, tol);
    }
}

TEST(UniverseState, AdoptedFloatBufferRoundTripsThroughDeviceBuffer) {
    UniverseTemperatureState state(make_buffer<float>({ 1.5f, 2.5f, 3.5f }));

    ASSERT_EQ(state.size(), std::size_t { 3 });
    const float first = state.data()[0];
    const float second = state.data()[1];
    const float third = state.data()[2];
    EXPECT_NEAR(first, 1.5f, tol);
    EXPECT_NEAR(second, 2.5f, tol);
    EXPECT_NEAR(third, 3.5f, tol);
}

TEST(UniverseState, AdoptedFloat3BufferRoundTripsThroughDeviceBuffer) {
    UniverseBulkVelocityState state(make_buffer<Float3>({ Float3(1.0f, 2.0f, 3.0f), Float3(4.0f, 5.0f, 6.0f) }));

    ASSERT_EQ(state.size(), std::size_t { 2 });
    const Float3 first = state.data()[0];
    const Float3 second = state.data()[1];
    EXPECT_FLOAT_EQ(first.x, 1.0f);
    EXPECT_FLOAT_EQ(first.y, 2.0f);
    EXPECT_FLOAT_EQ(first.z, 3.0f);
    EXPECT_FLOAT_EQ(second.x, 4.0f);
    EXPECT_FLOAT_EQ(second.y, 5.0f);
    EXPECT_FLOAT_EQ(second.z, 6.0f);
}

TEST(UniverseState, AdoptedIntBufferRoundTripsThroughDeviceBuffer) {
    UniverseCollisionCountState state(make_buffer<int>({ 7, 0, 42 }));

    ASSERT_EQ(state.size(), std::size_t { 3 });
    const int first = state.data()[0];
    const int second = state.data()[1];
    const int third = state.data()[2];
    EXPECT_EQ(first, 7);
    EXPECT_EQ(second, 0);
    EXPECT_EQ(third, 42);
}

TEST(UniverseState, ConstDataAccessorExposesSameBuffer) {
    const UniverseTemperatureState state(make_buffer<float>({ 9.0f }));

    ASSERT_EQ(state.size(), std::size_t { 1 });
    const float value = state.data()[0];
    EXPECT_NEAR(value, 9.0f, tol);
}

TEST(UniverseState, MoveConstructionTransfersBuffer) {
    UniverseTemperatureState source(make_buffer<float>({ 5.0f, 6.0f }));
    const UniverseTemperatureState moved = std::move(source);

    ASSERT_EQ(moved.size(), std::size_t { 2 });
    const float first = moved.data()[0];
    const float second = moved.data()[1];
    EXPECT_NEAR(first, 5.0f, tol);
    EXPECT_NEAR(second, 6.0f, tol);
}
