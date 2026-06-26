#include "../utilities/test_utils.h"

#include <atlas/random/default_random_engine.h>

#include <testkit/testkit.h>

#include <type_traits>

namespace {

using atlas::default_random_engine;

} // namespace

TEST(DefaultRandomEngine, AliasIsDefaultConstructibleAndUsable) {
    // Arrange: create a default random engine through the Atlas alias.
    default_random_engine<float> engine;

    // Act: generate two values from the same engine sequence.
    const auto first = engine();
    const auto second = engine();

    // Assert: consecutive samples advance the engine state.
    EXPECT_NE(first, second);
}

TEST(DefaultRandomEngine, AliasMatchesStandardEngineInTbbBuild) {
#ifndef ATLAS_TASKING_CUDA
    // Assert: the TBB backend uses the standard default random engine.
    EXPECT_TRUE((std::is_same_v<default_random_engine<float>, std::default_random_engine>));
#else
    // Assert: CUDA builds use a backend-specific engine type.
    SUCCEED();
#endif
}
