#include "../utilities/tests_utils.h"

#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using T = float;

} // namespace

TEST(UniformRealDistribution, ProducesValuesInsideRequestedRange) {
    atlas::default_random_engine<T> engine(7u);
    atlas::uniform_real_distribution<T> distribution(-2.0f, 3.0f);

    for (int i = 0; i < 64; ++i) {
        const auto value = distribution(engine);
        EXPECT_GE(value, -2.0f);
        EXPECT_LE(value, 3.0f);
    }
}

TEST(UniformRealDistribution, AliasMatchesStandardDistributionInTbbBuild) {
#ifndef ATLAS_TASKING_CUDA
    EXPECT_TRUE((std::is_same_v<atlas::uniform_real_distribution<T>, std::uniform_real_distribution<T>>));
#else
    SUCCEED();
#endif
}
