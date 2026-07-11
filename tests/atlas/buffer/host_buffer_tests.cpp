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

TEST(HostBuffer, MutableElementAccessWritesThrough) {
    // operator[] yields a writable reference on the host, so an assignment must be
    // observable on read-back — the element-access half of the container contract.
    HostBuffer<int> buffer { 1, 2, 3 };

    buffer[1] = 42;

    ASSERT_EQ(buffer.size(), std::size_t { 3 });
    const int first  = buffer[0];
    const int second = buffer[1];
    const int third  = buffer[2];
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 42);
    EXPECT_EQ(third, 3);
}

TEST(HostBuffer, ResizeShrinkKeepsRemainingPrefix) {
    // Shrinking drops the tail but leaves the surviving prefix untouched, mirroring
    // the DeviceBuffer shrink guarantee for the host-side container.
    HostBuffer<int> buffer { 5, 6, 7, 8 };

    buffer.resize(2);

    ASSERT_EQ(buffer.size(), std::size_t { 2 });
    const int first  = buffer[0];
    const int second = buffer[1];
    EXPECT_EQ(first, 5);
    EXPECT_EQ(second, 6);
}

TEST(HostBuffer, CopyConstructionYieldsIndependentEqualBuffer) {
    // HostBuffer is a copyable value container: a copy shares the source's contents
    // but owns separate storage, so mutating the copy must not touch the original.
    HostBuffer<int>       original { 3, 6, 9 };
    const HostBuffer<int> copy(original);

    ASSERT_EQ(copy.size(), original.size());
    for (std::size_t i = 0; i < copy.size(); ++i) {
        const int element = copy[i];
        EXPECT_EQ(element, original[i]);
    }

    original[0] = 100;
    const int untouched = copy[0];
    EXPECT_EQ(untouched, 3);
}
