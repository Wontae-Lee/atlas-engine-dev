#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <atlas/shuffle/shuffle_operator.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace atlas;

TEST(ShuffleOperator, SameIndexAndSeedProduceDeterministicKey) {
    const atlas::ShuffleOperator shuffle_operator {};

    const std::uint64_t key_a = shuffle_operator(7, 123u);
    const std::uint64_t key_b = shuffle_operator(7, 123u);

    EXPECT_EQ(key_a, key_b);
}

TEST(ShuffleOperator, DifferentSeedsChangeGeneratedKey) {
    const atlas::ShuffleOperator shuffle_operator {};

    const std::uint64_t key_a = shuffle_operator(7, 123u);
    const std::uint64_t key_b = shuffle_operator(7, 124u);

    EXPECT_NE(key_a, key_b);
}

TEST(ShuffleOperator, GeneratesUniqueKeysForSmallSequentialRange) {
    const atlas::ShuffleOperator shuffle_operator {};

    std::vector<std::uint64_t> keys;
    keys.reserve(64);
    for (int i = 0; i < 64; ++i) {
        keys.push_back(shuffle_operator(i, 42u));
    }

    std::sort(keys.begin(), keys.end());
    EXPECT_TRUE(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
}

TEST(ShuffleOperator, DeviceGeneratedKeysDriveStablePermutation) {
    constexpr int count = 8;

    DeviceBuffer<std::uint64_t> keys(count, 0);
    DeviceBuffer<int> values(count, 0);

    int* values_ptr         = atlas::raw_pointer_cast(values.data());
    std::uint64_t* keys_ptr = atlas::raw_pointer_cast(keys.data());
    const atlas::ShuffleOperator shuffle_operator {};

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            values_ptr[i] = i;
            keys_ptr[i]   = shuffle_operator(i, 11u);
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        keys.begin(),
        keys.end(),
        values.begin());

    const auto host_values = test::copy_device_buffer(values);
    ASSERT_EQ(host_values.size(), static_cast<std::size_t>(count));

    std::vector<int> sorted_values = host_values;
    std::sort(sorted_values.begin(), sorted_values.end());

    std::vector<int> expected(count);
    std::iota(expected.begin(), expected.end(), 0);

    EXPECT_EQ(sorted_values, expected);
    EXPECT_NE(host_values, expected);
}