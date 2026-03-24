#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#ifdef ATLAS_TASKING_CUDA
#include <thrust/tuple.h>
#endif

namespace {

#ifdef ATLAS_TASKING_CUDA
template <typename... Ts>
auto make_test_zip_tuple(Ts&&... args) {
    return thrust::make_tuple(std::forward<Ts>(args)...);
}

template <std::size_t I, typename Tuple>
decltype(auto) zip_get(Tuple&& t) {
    return thrust::get<I>(std::forward<Tuple>(t));
}
#else
template <typename... Ts>
auto make_test_zip_tuple(Ts&&... args) {
    return std::make_tuple(std::forward<Ts>(args)...);
}

template <std::size_t I, typename Tuple>
decltype(auto) zip_get(Tuple&& t) {
    return std::get<I>(std::forward<Tuple>(t));
}
#endif

}

TEST(ZipIterator, DereferenceReturnsTupleOfReferences) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    const auto it = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));

    auto t = *it;

    EXPECT_TRUE(atlas::test::near(zip_get<0>(t), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(zip_get<1>(t), 10.0, eps));

    zip_get<0>(t) = -7.0;
    zip_get<1>(t) = -9.0;

    EXPECT_TRUE(atlas::test::near(a[0], -7.0, eps));
    EXPECT_TRUE(atlas::test::near(b[0], -9.0, eps));
}

TEST(ZipIterator, PreIncrementAndPostIncrementAdvanceInLockstep) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    auto it = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));

    ++it;
    {
        const auto t = *it;
        EXPECT_TRUE(atlas::test::near(zip_get<0>(t), 2.0, eps));
        EXPECT_TRUE(atlas::test::near(zip_get<1>(t), 20.0, eps));
    }

    const auto old = it++;
    {
        const auto to = *old;
        EXPECT_TRUE(atlas::test::near(zip_get<0>(to), 2.0, eps));
        EXPECT_TRUE(atlas::test::near(zip_get<1>(to), 20.0, eps));
    }
    {
        const auto tn = *it;
        EXPECT_TRUE(atlas::test::near(zip_get<0>(tn), 3.0, eps));
        EXPECT_TRUE(atlas::test::near(zip_get<1>(tn), 30.0, eps));
    }
}

TEST(ZipIterator, EqualityAndInequalityCompareUnderlyingIterators) {
    std::vector<int> a { 1, 2, 3, 4 };
    std::vector<int> b { 10, 20, 30, 40 };

    const auto it0 = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));
    const auto it1 = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));
    const auto it2 = atlas::make_zip_iterator(make_test_zip_tuple(a.begin() + 1, b.begin() + 1));

    EXPECT_TRUE(it0 == it1);
    EXPECT_FALSE(it0 != it1);

    EXPECT_TRUE(it0 != it2);
    EXPECT_FALSE(it0 == it2);
}

TEST(ZipIterator, PlusNAdvancesAndMinusComputesDistanceFromFirstIterator) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    std::vector<double> a { 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> b { 10.0, 20.0, 30.0, 40.0, 50.0 };

    const auto begin = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));
    const auto end   = atlas::make_zip_iterator(make_test_zip_tuple(a.end(), b.end()));

    const auto it = begin + 3;
    {
        const auto t = *it;
        EXPECT_TRUE(atlas::test::near(zip_get<0>(t), 4.0, eps));
        EXPECT_TRUE(atlas::test::near(zip_get<1>(t), 40.0, eps));
    }

    EXPECT_EQ(end - begin, static_cast<std::ptrdiff_t>(a.size()));
    EXPECT_EQ(it - begin, static_cast<std::ptrdiff_t>(3));
    EXPECT_EQ(begin - it, static_cast<std::ptrdiff_t>(-3));
}

TEST(ZipIterator, CanIterateLikeAForwardIterator) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    auto it         = atlas::make_zip_iterator(make_test_zip_tuple(a.begin(), b.begin()));
    const auto it_e = atlas::make_zip_iterator(make_test_zip_tuple(a.end(), b.end()));

    std::size_t count = 0;
    for (; it != it_e; ++it) {
        auto t           = *it;
        const double sum = zip_get<0>(t) + zip_get<1>(t);

        if (count == 0) EXPECT_TRUE(atlas::test::near(sum, 11.0, eps));
        if (count == 1) EXPECT_TRUE(atlas::test::near(sum, 22.0, eps));
        if (count == 2) EXPECT_TRUE(atlas::test::near(sum, 33.0, eps));

        ++count;
    }

    EXPECT_EQ(count, static_cast<std::size_t>(3));
}
