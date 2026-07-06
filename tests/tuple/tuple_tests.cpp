#include <atlas/tuple/tuple.h>

#include <gtest/gtest.h>

#include <thrust/tuple.h>
#include <type_traits>

TEST(Tuple, MakeTupleAndGetWork) {
    const auto tuple = atlas::make_tuple(1, 2.5, 'x');

    EXPECT_EQ(atlas::get<0>(tuple), 1);
    EXPECT_DOUBLE_EQ(atlas::get<1>(tuple), 2.5);
    EXPECT_EQ(atlas::get<2>(tuple), 'x');
}

TEST(Tuple, AliasMatchesThrustTuple) {
    EXPECT_TRUE((std::is_same_v<atlas::tuple<int, float>, thrust::tuple<int, float>>));
}
