#include <atlas/parallel/parallel_for.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace {

using atlas::DeviceBuffer;
using atlas::ExecutionPolicy;

// Copies a whole DeviceBuffer back to host storage so the elements can be read on
// the host under either backend; a device element must never be dereferenced directly.
std::vector<int>
to_host(const DeviceBuffer<int>& buffer) {
    std::vector<int> host(buffer.size());
    atlas::copy_device_to_host(buffer, host.data(), buffer.size());
    return host;
}

}

TEST(ParallelFor, VisitsEveryIndexExactlyOnce) {
    const int count = 128;
    DeviceBuffer<int> buffer(count, 0);
    int* data = atlas::raw_pointer_cast(buffer.data());

    // Each index owns its own slot and stamps a unique, non-zero value there.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_ALL_DEVICE(const int i) { data[i] = i + 1; });

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], i + 1);
    }
}

TEST(ParallelFor, ZeroCountIsNoOp) {
    const int count = 16;
    DeviceBuffer<int> buffer(count, -1);
    int* data = atlas::raw_pointer_cast(buffer.data());

    // An empty [start, end) range must not invoke the body at all.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        0,
        [=] ATLAS_ALL_DEVICE(const int i) { data[i] = 999; });

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], -1);
    }
}

TEST(ParallelFor, EqualBoundsIsNoOp) {
    const int count = 8;
    DeviceBuffer<int> buffer(count, 5);
    int* data = atlas::raw_pointer_cast(buffer.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        4,
        4,
        [=] ATLAS_ALL_DEVICE(const int i) { data[i] = 0; });

    const std::vector<int> host = to_host(buffer);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(host[i], 5);
    }
}
