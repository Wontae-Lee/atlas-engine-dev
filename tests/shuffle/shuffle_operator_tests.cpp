#include "../utilities/tests_utils.h"

#include <atlas/shuffle/shuffle_operator.h>

#include <gtest/gtest.h>

TEST(ShuffleOperator, CallOperatorMatchesShuffleKey) {
    const atlas::ShuffleOperator shuffle;

    EXPECT_EQ(shuffle(5, 123u), shuffle.shuffle_key(5, 123u));
}

TEST(ShuffleOperator, SameIndexAndSeedProduceDeterministicKey) {
    const atlas::ShuffleOperator shuffle;

    const auto key0 = shuffle.shuffle_key(7, 999u);
    const auto key1 = shuffle.shuffle_key(7, 999u);

    EXPECT_EQ(key0, key1);
}

TEST(ShuffleOperator, DifferentSeedsChangeKey) {
    const atlas::ShuffleOperator shuffle;

    EXPECT_NE(shuffle.shuffle_key(3, 11u), shuffle.shuffle_key(3, 12u));
}

TEST(ShuffleOperator, DifferentIndicesChangeKey) {
    const atlas::ShuffleOperator shuffle;

    EXPECT_NE(shuffle.shuffle_key(3, 11u), shuffle.shuffle_key(4, 11u));
}

TEST(ShuffleOperator, InternalConstantsExposeExpectedValues) {
    const atlas::ShuffleOperator shuffle;

    EXPECT_EQ(shuffle.first_shift, 30);
    EXPECT_EQ(shuffle.second_shift, 27);
    EXPECT_EQ(shuffle.final_shift, 31);
    EXPECT_EQ(shuffle.index_offset, 0x9e3779b97f4a7c15ull);
    EXPECT_EQ(shuffle.first_multiplier, 0xbf58476d1ce4e5b9ull);
    EXPECT_EQ(shuffle.second_multiplier, 0x94d049bb133111ebull);
}
