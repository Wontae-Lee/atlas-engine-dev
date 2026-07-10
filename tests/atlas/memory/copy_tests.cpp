#include <atlas/memory/copy.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::copy_device_to_host;
using atlas::copy_host_to_device;
using atlas::raw_pointer_cast;

// Host staging uses std::vector so .data() is guaranteed to be a bare T*, which is what
// the raw-pointer copy overloads require. Under CUDA a DeviceBuffer's raw pointer is a
// device address, so it is only ever handed to copy_* — never dereferenced on the host.

}

TEST(Copy, RoundTripsThroughDeviceBufferOverloads) {
    const std::vector<int> source { 10, 20, 30, 40 };
    DeviceBuffer<int> device(source.size());

    copy_host_to_device(source.data(), device, source.size());

    std::vector<int> readback(source.size(), -1);
    copy_device_to_host(device, readback.data(), source.size());

    EXPECT_EQ(readback, source);
}

TEST(Copy, RoundTripsThroughRawPointerOverloads) {
    const std::vector<int> source { 1, 2, 3, 4, 5 };
    DeviceBuffer<int> device(source.size());

    copy_host_to_device(source.data(), raw_pointer_cast(device.data()), source.size());

    std::vector<int> readback(source.size(), -1);
    const int* device_src = raw_pointer_cast(device.data());
    copy_device_to_host(device_src, readback.data(), source.size());

    EXPECT_EQ(readback, source);
}

TEST(Copy, TransfersPartialPrefixOnly) {
    const std::vector<int> source { 7, 8, 9, 10 };
    DeviceBuffer<int> device(source.size());
    copy_host_to_device(source.data(), device, source.size());

    std::vector<int> readback(source.size(), -1);
    copy_device_to_host(device, readback.data(), std::size_t { 2 });

    EXPECT_EQ(readback[0], 7);
    EXPECT_EQ(readback[1], 8);
    // Elements past the requested count are left untouched.
    EXPECT_EQ(readback[2], -1);
    EXPECT_EQ(readback[3], -1);
}

TEST(Copy, ZeroCountLeavesDestinationUntouched) {
    const std::vector<int> source { 100, 200 };
    DeviceBuffer<int> device(source.size());
    copy_host_to_device(source.data(), device, source.size());

    std::vector<int> readback(source.size(), -1);
    copy_device_to_host(device, readback.data(), std::size_t { 0 });

    EXPECT_EQ(readback[0], -1);
    EXPECT_EQ(readback[1], -1);
}
