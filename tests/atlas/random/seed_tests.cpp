#include <atlas/random/seed.h>
// seed.h holds only named constants; its "hashing/mixing" is realized by
// atlas::shuffle_key, the SplitMix64 finalizer that consumes those constants.
// It lives in sampling.h, so that header is pulled in to exercise the hash the
// seed constants drive.
#include <atlas/sampling/sampling.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using atlas::shuffle_key;

// A fixed reference stream seed for the hash cases.
constexpr std::uint64_t reference_stream = 0xabcdef0123456789ull;

}

TEST(Seed, DiffuseAndMixSaltsAreDistinct) {
    // The three sine-hash salts must differ or two "independent" draws sharing
    // one geometric seed would collapse onto the same value.
    EXPECT_NE(atlas::RANDOM_HASH_SALT_DIFFUSE_U1, atlas::RANDOM_HASH_SALT_DIFFUSE_U2);
    EXPECT_NE(atlas::RANDOM_HASH_SALT_DIFFUSE_U1, atlas::RANDOM_HASH_SALT_MIX);
    EXPECT_NE(atlas::RANDOM_HASH_SALT_DIFFUSE_U2, atlas::RANDOM_HASH_SALT_MIX);
}

TEST(Seed, DsmcStreamSaltsAreDistinct) {
    EXPECT_NE(atlas::DSMC_COLLISION_LHS_SALT, atlas::DSMC_COLLISION_RHS_SALT);
    EXPECT_NE(atlas::DSMC_COLLISION_LHS_SALT, atlas::DSMC_COLLISION_ACCEPT_SALT);
    EXPECT_NE(atlas::DSMC_COLLISION_RHS_SALT, atlas::DSMC_COLLISION_ACCEPT_SALT);
    EXPECT_NE(atlas::DSMC_CELL_STREAM_MULTIPLIER, atlas::DSMC_COLLISION_LHS_SALT);

    // The scatter engine's seed shares the collision's stream base with the partner and
    // acceptance draws, so its salt must separate it from all three.
    EXPECT_NE(atlas::DSMC_COLLISION_SCATTER_SALT, atlas::DSMC_COLLISION_LHS_SALT);
    EXPECT_NE(atlas::DSMC_COLLISION_SCATTER_SALT, atlas::DSMC_COLLISION_RHS_SALT);
    EXPECT_NE(atlas::DSMC_COLLISION_SCATTER_SALT, atlas::DSMC_COLLISION_ACCEPT_SALT);
    EXPECT_NE(atlas::DSMC_COLLISION_SCATTER_SALT, atlas::DSMC_CELL_STREAM_MULTIPLIER);
}

TEST(Seed, GoldenRatioConstantsAreBitIdentical) {
    // Documented to be the same bit pattern in both roles (stream stride and
    // SplitMix64 increment).
    EXPECT_EQ(atlas::DSMC_CELL_STREAM_MULTIPLIER, atlas::SHUFFLE_HASH_INDEX_OFFSET);
}

TEST(Seed, UnitIntervalScaleIsPositiveReciprocalOfTwoPow53) {
    EXPECT_EQ(atlas::RANDOM_HASH_UNIT_INTERVAL_SHIFT, 11);
    EXPECT_GT(atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE, 0.0f);
    // scale * 2^53 == 1 by construction.
    EXPECT_FLOAT_EQ(atlas::RANDOM_HASH_UNIT_INTERVAL_SCALE * 9007199254740992.0f, 1.0f);
}

TEST(Seed, DefaultUnsignedIntSeedIsZero) {
    EXPECT_EQ(atlas::DEFAULT_UNSIGNED_INT_SEED, 0u);
}

TEST(Seed, ShuffleKeyIsPureFunctionOfArguments) {
    // Same (index, seed) always yields the same key: no hidden state.
    EXPECT_EQ(shuffle_key(42, reference_stream), shuffle_key(42, reference_stream));
    EXPECT_EQ(shuffle_key(-7, reference_stream), shuffle_key(-7, reference_stream));
}

TEST(Seed, ShuffleKeyDecorrelatesAdjacentIndices) {
    // Avalanche mixing means consecutive indices map to different keys.
    for (int index = 0; index < 256; ++index) {
        EXPECT_NE(shuffle_key(index, reference_stream), shuffle_key(index + 1, reference_stream));
    }
}

TEST(Seed, ShuffleKeyDecorrelatesAdjacentSeeds) {
    for (std::uint64_t seed = 0; seed < 256; ++seed) {
        EXPECT_NE(shuffle_key(99, seed), shuffle_key(99, seed + 1));
    }
}
