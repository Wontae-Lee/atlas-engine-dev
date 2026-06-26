#include "../utilities/test_utils.h"

#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using atlas::default_random_engine;
using atlas::uniform_real_distribution;

} // namespace

TEST(UniformRealDistribution, ProducesValuesInsideRequestedRange) {
    // Arrange: create a deterministic engine and bounded distribution.
    default_random_engine<float> engine(7u);
    uniform_real_distribution<float> distribution(-2.0f, 3.0f);

    // Act and assert: sampled values stay inside the requested range.
    for (int i = 0; i < 64; ++i) {
        const auto value = distribution(engine);
        EXPECT_GE(value, -2.0f);
        EXPECT_LE(value, 3.0f);
    }
}

TEST(UniformRealDistribution, AliasMatchesStandardDistributionInTbbBuild) {
#ifndef ATLAS_TASKING_CUDA
    // Assert: the TBB backend uses the standard uniform real distribution.
    EXPECT_TRUE((std::is_same_v<uniform_real_distribution<float>, std::uniform_real_distribution<float>>));
#else
    // Assert: CUDA builds use a backend-specific distribution type.
    SUCCEED();
#endif
}
