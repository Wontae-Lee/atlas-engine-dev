#include <atlas/buffer/host_buffer.h>

#include <atlas/buffer/device_buffer.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::HostBuffer;

// HostBuffer elements always live in host memory, but reads still copy into a local
// value so the same access pattern holds whether the backend alias is std::vector or
// thrust::host_vector.

}

TEST(HostBuffer, DefaultConstructionIsEmpty) {
    const HostBuffer<int> buffer {};

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), std::size_t { 0 });
}

TEST(HostBuffer, SizedConstructionValueInitializes) {
    const HostBuffer<int> buffer(3);

    EXPECT_EQ(buffer.size(), std::size_t { 3 });
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, 0);
    }
}

TEST(HostBuffer, SizedWithValueConstructionFillsEveryElement) {
    const HostBuffer<int> buffer(4, 9);

    EXPECT_EQ(buffer.size(), std::size_t { 4 });
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        const int element = buffer[i];
        EXPECT_EQ(element, 9);
    }
}

TEST(HostBuffer, InitializerListConstructionKeepsOrder) {
    const HostBuffer<int> buffer { 3, 6, 9 };

    ASSERT_EQ(buffer.size(), std::size_t { 3 });
    const int first  = buffer[0];
    const int second = buffer[1];
    const int third  = buffer[2];
    EXPECT_EQ(first, 3);
    EXPECT_EQ(second, 6);
    EXPECT_EQ(third, 9);
}

TEST(HostBuffer, ResizePreservesPrefixContents) {
    HostBuffer<int> buffer { 1, 2, 3 };

    buffer.resize(5);

    ASSERT_EQ(buffer.size(), std::size_t { 5 });
    const int first  = buffer[0];
    const int second = buffer[1];
    const int third  = buffer[2];
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 2);
    EXPECT_EQ(third, 3);
}

TEST(HostBuffer, ConstructsFromDeviceBufferRangeAsBulkCopy) {
    // The property the alias exists for: a HostBuffer built from a DeviceBuffer's
    // iterator pair is one bulk transfer, and the values must land in order.
    const std::vector<int> source { 11, 22, 33 };
    const DeviceBuffer<int> device(source.begin(), source.end());

    const HostBuffer<int> mirror(device.begin(), device.end());

    ASSERT_EQ(mirror.size(), source.size());
    for (std::size_t i = 0; i < mirror.size(); ++i) {
        const int element = mirror[i];
        EXPECT_EQ(element, source[i]);
    }
}
