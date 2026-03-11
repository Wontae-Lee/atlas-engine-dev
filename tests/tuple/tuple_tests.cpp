#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <string>
#include <type_traits>

TEST(Tuple, MakeTupleAndGet_BasicThreeValues) {
    const auto t = atlas::make_tuple(42, 3.5, std::string("hello"));

    EXPECT_EQ(atlas::get<0>(t), 42);
    EXPECT_DOUBLE_EQ(atlas::get<1>(t), 3.5);
    EXPECT_EQ(atlas::get<2>(t), "hello");
}

TEST(Tuple, TupleIsOneObjectHoldingMultipleDifferentTypes) {
    using T   = atlas::tuple<int, double, const char*>;
    const T t = atlas::make_tuple(7, 1.25, "world");

    EXPECT_EQ(atlas::get<0>(t), 7);
    EXPECT_DOUBLE_EQ(atlas::get<1>(t), 1.25);
    EXPECT_STREQ(atlas::get<2>(t), "world");
}

TEST(Tuple, GetUsesCompileTimeIndex) {
    const auto t = atlas::make_tuple(10, 20, 30);

    EXPECT_EQ(atlas::get<0>(t), 10);
    EXPECT_EQ(atlas::get<1>(t), 20);
    EXPECT_EQ(atlas::get<2>(t), 30);
}

TEST(Tuple, CopyKeepsAllElements) {
    const auto t0  = atlas::make_tuple(1, 2.0, std::string("a"));
    const auto& t1 = t0;

    EXPECT_EQ(atlas::get<0>(t1), 1);
    EXPECT_DOUBLE_EQ(atlas::get<1>(t1), 2.0);
    EXPECT_EQ(atlas::get<2>(t1), "a");
}

TEST(Tuple, AssignmentOverwritesAllElements) {
    auto t = atlas::make_tuple(1, std::string("old"));
    t      = atlas::make_tuple(99, std::string("new"));

    EXPECT_EQ(atlas::get<0>(t), 99);
    EXPECT_EQ(atlas::get<1>(t), "new");
}

TEST(Tuple, CanStoreVectorsOrUserTypesLikeAnyValue) {
    const atlas::math::Vector<double, 3> p(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> n(0.0, 0.0, 1.0);

    const auto t = atlas::make_tuple(p, n);

    EXPECT_TRUE(atlas::test::vec_near(atlas::get<0>(t), p, static_cast<double>(atlas::eps)));
    EXPECT_TRUE(atlas::test::vec_near(atlas::get<1>(t), n, static_cast<double>(atlas::eps)));
}

TEST(Tuple, TypeCheck_GetReturnsExpectedElementType) {
    ATLAS_MAYBE_UNUSED const auto t = atlas::make_tuple(1, 2.0f, 3.0);

    using T0 = decltype(atlas::get<0>(t));
    using T1 = decltype(atlas::get<1>(t));
    using T2 = decltype(atlas::get<2>(t));

    EXPECT_TRUE((std::is_same_v<T0, int>));
    EXPECT_TRUE((std::is_same_v<T1, float>));
    EXPECT_TRUE((std::is_same_v<T2, double>));
}

TEST(Tuple, PracticalExample_PackRelatedValuesTogether) {
    const atlas::math::Vector<double, 3> position(0.25, 0.50, 1.00);
    const atlas::math::Vector<double, 3> direction(0.0, 0.0, -1.0);

    const auto ray_like = atlas::make_tuple(position, direction);

    const auto o = atlas::get<0>(ray_like);
    const auto d = atlas::get<1>(ray_like);

    EXPECT_TRUE(atlas::test::vec_near(o, position, static_cast<double>(atlas::eps)));
    EXPECT_TRUE(atlas::test::vec_near(d, direction, static_cast<double>(atlas::eps)));
}