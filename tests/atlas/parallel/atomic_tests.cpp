#include <atlas/parallel/atomic.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>
#include <atlas/parallel/parallel_for.h>

#include <gtest/gtest.h>

namespace {

using atlas::DeviceBuffer;
using atlas::ExecutionPolicy;

}

TEST(Atomic, AddReturnsPriorValue) {
    // atomic_add is host+device; on the host it resolves to the builtin fetch-add.
    int counter = 10;

    const int prior = atlas::atomic_add(&counter, 5);

    EXPECT_EQ(prior, 10);
    EXPECT_EQ(counter, 15);
}

TEST(Atomic, ChainedAddsReturnEachPriorValue) {
    int counter = 0;

    EXPECT_EQ(atlas::atomic_add(&counter, 3), 0);
    EXPECT_EQ(atlas::atomic_add(&counter, 3), 3);
    EXPECT_EQ(atlas::atomic_add(&counter, 3), 6);
    EXPECT_EQ(counter, 9);
}

TEST(Atomic, ConcurrentAddsCountEveryIteration) {
    const int count = 4096;
    DeviceBuffer<int> counter(1, 0);
    int* address = atlas::raw_pointer_cast(counter.data());

    // Every iteration contributes one increment; the final total is order-independent.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_ALL_DEVICE(const int) { atlas::atomic_add(address, 1); });

    int total = 0;
    atlas::copy_device_to_host(counter, &total, 1);
    EXPECT_EQ(total, count);
}
