#include "../utilities/test_utils.h"

#include <atlas/tuple/tuple.h>

#include <testkit/testkit.h>

#include <string>
#include <tuple>
#include <type_traits>

TEST(Tuple, MakeTupleAndGetWork) {
    const auto tuple = atlas::make_tuple(1, 2.5, std::string("x"));

    EXPECT_EQ(atlas::get<0>(tuple), 1);
    EXPECT_DOUBLE_EQ(atlas::get<1>(tuple), 2.5);
    EXPECT_EQ(atlas::get<2>(tuple), "x");
}

TEST(Tuple, AliasMatchesStdTupleInTbbBuild) {
#ifndef ATLAS_TASKING_CUDA
    EXPECT_TRUE((std::is_same_v<atlas::tuple<int, float>, std::tuple<int, float>>));
#else
    SUCCEED();
#endif
}
