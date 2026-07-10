#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using atlas::default_random_engine;

using result_type = default_random_engine::result_type;

// A fixed reference seed so every case is reproducible run to run.
constexpr result_type reference_seed = 12345u;

}

TEST(DefaultRandomEngine, SameSeedProducesSameSequence) {
    default_random_engine a(reference_seed);
    default_random_engine b(reference_seed);

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(a(), b());
    }
}

TEST(DefaultRandomEngine, DifferentSeedsDecorrelateFirstDraw) {
    // The recurrence is a bijection on [1, m-1], so distinct non-zero seeds
    // land on distinct states and therefore distinct first draws.
    default_random_engine e1(1u);
    default_random_engine e2(2u);
    default_random_engine e3(3u);

    const result_type d1 = e1();
    const result_type d2 = e2();
    const result_type d3 = e3();

    EXPECT_NE(d1, d2);
    EXPECT_NE(d1, d3);
    EXPECT_NE(d2, d3);
}

TEST(DefaultRandomEngine, ZeroSeedMapsToDefaultSeed) {
    // seed() maps a value congruent to zero onto default_seed, so a zero seed,
    // the default seed, and a default-constructed engine share one stream.
    default_random_engine zero_seeded(0u);
    default_random_engine default_seeded(default_random_engine::default_seed);
    default_random_engine constructed;

    for (int i = 0; i < 16; ++i) {
        const result_type expected = default_seeded();
        EXPECT_EQ(zero_seeded(), expected);
        EXPECT_EQ(constructed(), expected);
    }
}

TEST(DefaultRandomEngine, DrawsStayWithinMinMax) {
    default_random_engine engine(reference_seed);

    for (int i = 0; i < 4096; ++i) {
        const result_type value = engine();
        EXPECT_GE(value, default_random_engine::min());
        EXPECT_LE(value, default_random_engine::max());
        // The state must never reach zero or the stream would collapse.
        EXPECT_NE(value, 0u);
    }
}

TEST(DefaultRandomEngine, DiscardMatchesRepeatedDraws) {
    default_random_engine skipped(reference_seed);
    default_random_engine stepped(reference_seed);

    skipped.discard(10);
    for (int i = 0; i < 10; ++i) {
        static_cast<void>(stepped());
    }

    EXPECT_EQ(skipped(), stepped());
}

TEST(DefaultRandomEngine, SeedRestartsTheStream) {
    default_random_engine engine(reference_seed);

    const result_type first = engine();
    static_cast<void>(engine());
    static_cast<void>(engine());

    engine.seed(reference_seed);
    EXPECT_EQ(engine(), first);
}

TEST(DefaultRandomEngine, MinMaxBracketTheModulus) {
    EXPECT_EQ(default_random_engine::min(), 1u);
    EXPECT_EQ(default_random_engine::max(), default_random_engine::modulus - 1u);
}
