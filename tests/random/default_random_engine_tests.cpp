#include "../utilities/tests_utils.h"

#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using T = float;

} // namespace

TEST(DefaultRandomEngine, AliasIsDefaultConstructibleAndUsable) {
    atlas::default_random_engine<T> engine;

    const auto first = engine();
    const auto second = engine();

    EXPECT_NE(first, second);
}

TEST(DefaultRandomEngine, AliasMatchesStandardEngineInTbbBuild) {
#ifndef ATLAS_TASKING_CUDA
    EXPECT_TRUE((std::is_same_v<atlas::default_random_engine<T>, std::default_random_engine>));
#else
    SUCCEED();
#endif
}
