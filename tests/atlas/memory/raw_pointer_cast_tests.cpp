#include <atlas/memory/raw_pointer_cast.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::copy_device_to_host;
using atlas::copy_host_to_device;
using atlas::raw_pointer_cast;

// On a plain host pointer raw_pointer_cast is the identity, so the result stays
// host-dereferenceable. On a DeviceBuffer's pointer the result is a bare device address
// under CUDA, so it is verified only by moving data through copy_* and checking values.

}

TEST(RawPointerCast, IsIdentityOnMutableHostPointer) {
    int value = 42;

    int* raw = raw_pointer_cast(&value);

    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(*raw, 42);

    *raw = 55;
    EXPECT_EQ(value, 55);
}

TEST(RawPointerCast, IsIdentityOnConstHostPointer) {
    const int value = 7;

    const int* raw = raw_pointer_cast(&value);

    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(*raw, 7);
}

TEST(RawPointerCast, YieldsUsablePointerIntoDeviceBuffer) {
    const std::vector<int> source { 3, 6, 9, 12 };
    DeviceBuffer<int> device(source.size());
    copy_host_to_device(source.data(), device, source.size());

    // The raw pointer must address the buffer's storage: reading through it (via a
    // device-to-host copy) must reproduce the buffer's first elements.
    const int* raw = raw_pointer_cast(device.data());
    ASSERT_NE(raw, nullptr);

    std::vector<int> readback(source.size(), -1);
    copy_device_to_host(raw, readback.data(), source.size());

    EXPECT_EQ(readback, source);
}

TEST(RawPointerCast, MutableBufferPointerWritesThroughToStorage) {
    const std::vector<int> source { 1, 1, 1 };
    DeviceBuffer<int> device(source.size());

    // Writing through the raw pointer must land in the buffer's storage.
    int* raw = raw_pointer_cast(device.data());
    ASSERT_NE(raw, nullptr);
    copy_host_to_device(source.data(), raw, source.size());

    std::vector<int> readback(source.size(), -1);
    copy_device_to_host(device, readback.data(), source.size());

    EXPECT_EQ(readback, source);
}
