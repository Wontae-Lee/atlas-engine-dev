#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <thrust/random.h>
#include <type_traits>

TEST(DefaultRandomEngine, AliasIsDefaultConstructibleAndUsable) {
    atlas::default_random_engine engine;

    const auto first  = engine();
    const auto second = engine();

    EXPECT_NE(first, second);
}

TEST(DefaultRandomEngine, AliasMatchesThrustEngine) {
    EXPECT_TRUE((std::is_same_v<atlas::default_random_engine, thrust::default_random_engine>));
}

TEST(DefaultRandomEngine, SameSeedProducesSameSequence) {
    atlas::default_random_engine first(42u);
    atlas::default_random_engine second(42u);

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(first(), second());
    }
}
