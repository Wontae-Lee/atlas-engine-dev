#include "../utilities/tests_utils.h"

#include <atlas/parallel/parallel_sort.h>

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
    std::vector<int> values { 9, 2, 8, 1 };

    atlas::parallel_sort<atlas::ExecutionPolicy::device>(values.begin(), values.end());

    EXPECT_EQ(values, (std::vector<int> { 1, 2, 8, 9 }));
}

TEST(ParallelSort, SortByKeyKeepsValuePairsAligned) {
    std::vector<int> keys { 30, 10, 20 };
    std::vector<std::string> values { "c", "a", "b" };

    atlas::parallel_sort_by_key<atlas::ExecutionPolicy::host>(keys.begin(), keys.end(), values.begin());

    EXPECT_EQ(keys, (std::vector<int> { 10, 20, 30 }));
    EXPECT_EQ(values, (std::vector<std::string> { "a", "b", "c" }));
}

TEST(ParallelSort, EmptyRangeIsNoOp) {
    std::vector<int> values { 3, 1 };

    EXPECT_NO_THROW(atlas::parallel_sort<atlas::ExecutionPolicy::serial>(values.begin(), values.begin()));
    EXPECT_EQ(values, (std::vector<int> { 3, 1 }));
}
