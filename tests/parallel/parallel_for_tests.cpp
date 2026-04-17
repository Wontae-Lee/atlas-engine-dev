#include "../utilities/tests_utils.h"

#include <atlas/parallel/parallel_for.h>

#include <gtest/gtest.h>

#include <atomic>
#include <vector>

TEST(ParallelFor, SerialIndexRangeVisitsEveryElement) {
    std::vector<int> values(6, 0);

    atlas::parallel_for<atlas::ExecutionPolicy::serial>(
        0,
        static_cast<int>(values.size()),
        [&values](int i) {
            values[static_cast<std::size_t>(i)] = i * 2;
        });

    EXPECT_EQ(values, (std::vector<int> { 0, 2, 4, 6, 8, 10 }));
}

TEST(ParallelFor, HostIndexRangeVisitsEveryElement) {
    std::vector<int> values(5, 0);

    atlas::parallel_for<atlas::ExecutionPolicy::host>(
        0,
        static_cast<int>(values.size()),
        [&values](int i) {
            values[static_cast<std::size_t>(i)] = i + 1;
        });

    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3, 4, 5 }));
}

TEST(ParallelFor, DeviceIndexRangeVisitsEveryElement) {
    std::vector<int> values(4, 0);

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        static_cast<int>(values.size()),
        [&values](int i) {
            values[static_cast<std::size_t>(i)] = 10 - i;
        });

    EXPECT_EQ(values, (std::vector<int> { 10, 9, 8, 7 }));
}

TEST(ParallelFor, IteratorRangeVisitsEveryElement) {
    std::vector<int> values { 1, 2, 3, 4 };
    std::atomic<int> sum = 0;

    atlas::parallel_for<atlas::ExecutionPolicy::serial>(
        values.begin(),
        values.end(),
        [&sum](int value) {
            sum.fetch_add(value, std::memory_order_relaxed);
        });

    EXPECT_EQ(sum.load(std::memory_order_relaxed), 10);
}

TEST(ParallelFor, EmptyIndexRangeIsNoOp) {
    std::atomic<int> count = 0;

    atlas::parallel_for<atlas::ExecutionPolicy::host>(
        3,
        3,
        [&count](int) {
            count.fetch_add(1, std::memory_order_relaxed);
        });

    EXPECT_EQ(count.load(std::memory_order_relaxed), 0);
}
