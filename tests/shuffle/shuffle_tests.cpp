#include <atlas/shuffle/shuffle.h>

#include <gtest/gtest.h>

namespace {

using atlas::Shuffle;
using atlas::SHUFFLE_HASH_FINAL_SHIFT;
using atlas::SHUFFLE_HASH_FIRST_MULTIPLIER;
using atlas::SHUFFLE_HASH_FIRST_SHIFT;
using atlas::SHUFFLE_HASH_INDEX_OFFSET;
using atlas::SHUFFLE_HASH_SECOND_MULTIPLIER;
using atlas::SHUFFLE_HASH_SECOND_SHIFT;

} // namespace

TEST(Shuffle, CallOperatorMatchesShuffleKey) {
    // Arrange: create a shuffle operator.
    const Shuffle shuffle;

    // Assert: call operator forwards to shuffle_key.
    EXPECT_EQ(shuffle(5, 123u), shuffle.shuffle_key(5, 123u));
}

TEST(Shuffle, SameIndexAndSeedProduceDeterministicKey) {
    // Arrange: create a shuffle operator.
    const Shuffle shuffle;

    // Act: compute the same key twice.
    const auto key0 = shuffle.shuffle_key(7, 999u);
    const auto key1 = shuffle.shuffle_key(7, 999u);

    // Assert: identical index and seed produce deterministic keys.
    EXPECT_EQ(key0, key1);
}

TEST(Shuffle, DifferentSeedsChangeKey) {
    // Arrange: create a shuffle operator.
    const Shuffle shuffle;

    // Assert: changing the seed changes the generated key.
    EXPECT_NE(shuffle.shuffle_key(3, 11u), shuffle.shuffle_key(3, 12u));
}

TEST(Shuffle, DifferentIndicesChangeKey) {
    // Arrange: create a shuffle operator.
    const Shuffle shuffle;

    // Assert: changing the index changes the generated key.
    EXPECT_NE(shuffle.shuffle_key(3, 11u), shuffle.shuffle_key(4, 11u));
}

TEST(Shuffle, SeedConstantsExposeExpectedValues) {
    // Assert: seed constants match the SplitMix-style shuffle parameters.
    EXPECT_EQ(SHUFFLE_HASH_FIRST_SHIFT, 30);
    EXPECT_EQ(SHUFFLE_HASH_SECOND_SHIFT, 27);
    EXPECT_EQ(SHUFFLE_HASH_FINAL_SHIFT, 31);
    EXPECT_EQ(SHUFFLE_HASH_INDEX_OFFSET, 0x9e3779b97f4a7c15ull);
    EXPECT_EQ(SHUFFLE_HASH_FIRST_MULTIPLIER, 0xbf58476d1ce4e5b9ull);
    EXPECT_EQ(SHUFFLE_HASH_SECOND_MULTIPLIER, 0x94d049bb133111ebull);
}
