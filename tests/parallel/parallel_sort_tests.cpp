#include <atlas/parallel/parallel_sort.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(ParallelSort, SerialSortOrdersRangeAscending) {
    std::vector<int> values { 4, 1, 3, 2 };

    atlas::parallel_sort<atlas::ExecutionPolicy::serial>(values.begin(), values.end());

    EXPECT_EQ(values, (std::vector<int> { 1, 2, 3, 4 }));
}

TEST(ParallelSort, HostSortOrdersRangeAscending) {
    std::vector<int> values { 7, 5, 6, 4 };

    atlas::parallel_sort<atlas::ExecutionPolicy::host>(values.begin(), values.end());

    EXPECT_EQ(values, (std::vector<int> { 4, 5, 6, 7 }));
}

TEST(ParallelSort, DeviceSortOrdersRangeAscending) {
    const std::vector<int> input { 9, 2, 8, 1 };
    atlas::DeviceBuffer<int> values(input.size());
    atlas::copy_host_to_device(input.data(), values, input.size());

    atlas::parallel_sort<atlas::ExecutionPolicy::device>(values.begin(), values.end());

    std::vector<int> host_values(values.size(), 0);
    atlas::copy_device_to_host(values, host_values.data(), host_values.size());

    EXPECT_EQ(host_values, (std::vector<int> { 1, 2, 8, 9 }));
}

TEST(ParallelSort, SortByKeyKeepsValuePairsAligned) {
    std::vector<int> keys { 30, 10, 20 };
    std::vector<std::string> values { "c", "a", "b" };

    atlas::parallel_sort_by_key<atlas::ExecutionPolicy::host>(keys.begin(), keys.end(), values.begin());

    EXPECT_EQ(keys, (std::vector<int> { 10, 20, 30 }));
    EXPECT_EQ(values, (std::vector<std::string> { "a", "b", "c" }));
}

TEST(ParallelSort, DeviceSortByKeyKeepsValuePairsAligned) {
    const std::vector<int> input_keys { 30, 10, 20 };
    const std::vector<int> input_values { 3, 1, 2 };

    atlas::DeviceBuffer<int> keys(input_keys.size());
    atlas::DeviceBuffer<int> values(input_values.size());
    atlas::copy_host_to_device(input_keys.data(), keys, input_keys.size());
    atlas::copy_host_to_device(input_values.data(), values, input_values.size());

    atlas::parallel_sort_by_key<atlas::ExecutionPolicy::device>(keys.begin(), keys.end(), values.begin());

    std::vector<int> host_keys(keys.size(), 0);
    std::vector<int> host_values(values.size(), 0);
    atlas::copy_device_to_host(keys, host_keys.data(), host_keys.size());
    atlas::copy_device_to_host(values, host_values.data(), host_values.size());

    EXPECT_EQ(host_keys, (std::vector<int> { 10, 20, 30 }));
    EXPECT_EQ(host_values, (std::vector<int> { 1, 2, 3 }));
}

TEST(ParallelSort, EmptyRangeIsNoOp) {
    std::vector<int> values { 3, 1 };

    EXPECT_NO_THROW(atlas::parallel_sort<atlas::ExecutionPolicy::serial>(values.begin(), values.begin()));
    EXPECT_EQ(values, (std::vector<int> { 3, 1 }));
}
