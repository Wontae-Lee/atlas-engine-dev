#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

TEST(DevicePairIndexer, MappingMatchesUpperTriangularExample) {
    constexpr atlas::DevicePairIndexer<int> idx;

    constexpr int n = 4;

    EXPECT_EQ(idx.pair_index(0, 0, n), 0);
    EXPECT_EQ(idx.pair_index(0, 1, n), 1);
    EXPECT_EQ(idx.pair_index(0, 2, n), 2);
    EXPECT_EQ(idx.pair_index(0, 3, n), 3);

    EXPECT_EQ(idx.pair_index(1, 1, n), 4);
    EXPECT_EQ(idx.pair_index(1, 2, n), 5);
    EXPECT_EQ(idx.pair_index(1, 3, n), 6);

    EXPECT_EQ(idx.pair_index(2, 2, n), 7);
    EXPECT_EQ(idx.pair_index(2, 3, n), 8);

    EXPECT_EQ(idx.pair_index(3, 3, n), 9);
}

TEST(DevicePairIndexer, SymmetryHoldsForSwappedInputs) {
    constexpr atlas::DevicePairIndexer<int> idx;

    constexpr int n = 7;

    EXPECT_EQ(idx.pair_index(2, 5, n), idx.pair_index(5, 2, n));
    EXPECT_EQ(idx.pair_index(0, 6, n), idx.pair_index(6, 0, n));
    EXPECT_EQ(idx.pair_index(3, 3, n), idx.pair_index(3, 3, n));
}

TEST(DevicePairIndexer, ProducesUniqueIndicesForAllUnorderedPairs) {
    constexpr atlas::DevicePairIndexer<int> idx;

    constexpr int n        = 8;
    constexpr int expected = n * (n + 1) / 2;

    bool seen[36] = {};

    int count = 0;
    for (int s = 0; s < n; ++s) {
        for (int r = s; r < n; ++r) {
            const int k = idx.pair_index(s, r, n);
            ASSERT_GE(k, 0);
            ASSERT_LT(k, expected);

            EXPECT_FALSE(seen[k]) << "collision at index " << k << " for pair (" << s << "," << r << ")";
            if (!seen[k]) {
                seen[k] = true;
                ++count;
            }
        }
    }

    EXPECT_EQ(count, expected);

    for (int k = 0; k < expected; ++k) {
        EXPECT_TRUE(seen[k]) << "missing index " << k;
    }
}

TEST(DevicePairIndexer, RangeMatchesPackedTriangularSize) {
    constexpr atlas::DevicePairIndexer<int> idx;

    for (int n = 1; n <= 32; ++n) {
        const int max_index = n * (n + 1) / 2 - 1;

        EXPECT_EQ(idx.pair_index(n - 1, n - 1, n), max_index);

        EXPECT_EQ(idx.pair_index(0, 0, n), 0);
    }
}

TEST(DevicePairIndexer, SupportsDifferentIntegerTypes) {
    constexpr atlas::DevicePairIndexer<std::size_t> idx;

    constexpr std::size_t n = 10;

    static_assert(std::is_same_v<decltype(idx.pair_index(0, 0, n)), std::size_t>);

    EXPECT_EQ(idx.pair_index(0, 9, n), static_cast<std::size_t>(9));
    EXPECT_EQ(idx.pair_index(3, 3, n), static_cast<std::size_t>(3 * 10 - (3 * 2) / 2));
}