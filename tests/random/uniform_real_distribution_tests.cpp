#include <atlas/random/uniform_real_distribution.h>

#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <thrust/random/uniform_real_distribution.h>
#include <type_traits>

TEST(UniformRealDistribution, ProducesValuesInsideRequestedRange) {
    atlas::default_random_engine engine(7u);
    atlas::uniform_real_distribution<float> distribution(-2.0f, 3.0f);

    for (int i = 0; i < 64; ++i) {
        const float value = distribution(engine);
        EXPECT_GE(value, -2.0f);
        EXPECT_LE(value, 3.0f);
    }
}

TEST(UniformRealDistribution, AliasMatchesThrustDistributionAndDefaultsToFloat) {
    EXPECT_TRUE((std::is_same_v<atlas::uniform_real_distribution<float>,
                                thrust::uniform_real_distribution<float>>));
    EXPECT_TRUE((std::is_same_v<atlas::uniform_real_distribution<>,
                                atlas::uniform_real_distribution<float>>));
}
