#include <atlas/parallel/parallel_fill.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::ExecutionPolicy;

std::vector<int>
to_host(const DeviceBuffer<int>& buffer) {
    std::vector<int> host(buffer.size());
    atlas::copy_device_to_host(buffer, host.data(), buffer.size());
    return host;
}

}

TEST(ParallelFill, SetsEveryElement) {
    const int count = 100;
    DeviceBuffer<int> buffer(count, 0);

    atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), 7);

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], 7);
    }
}

TEST(ParallelFill, FillsOnlyThePrefixRange) {
    const int count = 10;
    const std::ptrdiff_t filled = 4;
    DeviceBuffer<int> buffer(count, -1);

    atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.begin() + filled, 3);

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], i < filled ? 3 : -1);
    }
}

TEST(ParallelFill, EmptyRangeIsNoOp) {
    const int count = 8;
    DeviceBuffer<int> buffer(count, -1);

    atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.begin(), 9);

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], -1);
    }
}
