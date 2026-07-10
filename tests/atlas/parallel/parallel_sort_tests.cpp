#include <atlas/parallel/parallel_sort.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <gtest/gtest.h>

#include <algorithm>
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

// Sorts an input on the device backend and returns the result on the host.
std::vector<int>
sort_on_device(const std::vector<int>& input) {
    DeviceBuffer<int> buffer(input.begin(), input.end());
    int* first = atlas::raw_pointer_cast(buffer.data());
    int* last  = first + input.size();

    atlas::parallel_sort<ExecutionPolicy::device>(first, last);

    return to_host(buffer);
}

}

TEST(ParallelSort, SortsReverseSortedInput) {
    const std::vector<int> input { 5, 4, 3, 2, 1, 0 };

    const std::vector<int> sorted = sort_on_device(input);

    EXPECT_EQ(sorted, (std::vector<int> { 0, 1, 2, 3, 4, 5 }));
}

TEST(ParallelSort, LeavesAlreadySortedInputSorted) {
    const std::vector<int> input { 0, 1, 2, 3, 4, 5 };

    EXPECT_EQ(sort_on_device(input), input);
}

TEST(ParallelSort, HandlesAllEqualInput) {
    const std::vector<int> input(7, 42);

    EXPECT_EQ(sort_on_device(input), input);
}

TEST(ParallelSort, HandlesSingleElement) {
    const std::vector<int> input { 99 };

    EXPECT_EQ(sort_on_device(input), input);
}

TEST(ParallelSort, ProducesSortedPermutationOfInput) {
    const std::vector<int> input { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };

    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(sort_on_device(input), expected);
}
