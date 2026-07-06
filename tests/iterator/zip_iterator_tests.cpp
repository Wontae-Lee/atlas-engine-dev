#include <atlas/iterator/zip_iterator.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

TEST(ZipIterator, DereferenceReturnsTupleOfReferences) {
    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    const auto it = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));

    auto t = *it;

    EXPECT_DOUBLE_EQ(atlas::zip_get<0>(t), 1.0);
    EXPECT_DOUBLE_EQ(atlas::zip_get<1>(t), 10.0);

    atlas::zip_get<0>(t) = -7.0;
    atlas::zip_get<1>(t) = -9.0;

    EXPECT_DOUBLE_EQ(a[0], -7.0);
    EXPECT_DOUBLE_EQ(b[0], -9.0);
}

TEST(ZipIterator, PreIncrementAndPostIncrementAdvanceInLockstep) {
    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    auto it = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));

    ++it;
    {
        const auto t = *it;
        EXPECT_DOUBLE_EQ(atlas::zip_get<0>(t), 2.0);
        EXPECT_DOUBLE_EQ(atlas::zip_get<1>(t), 20.0);
    }

    const auto old = it++;
    {
        const auto to = *old;
        EXPECT_DOUBLE_EQ(atlas::zip_get<0>(to), 2.0);
        EXPECT_DOUBLE_EQ(atlas::zip_get<1>(to), 20.0);
    }
    {
        const auto tn = *it;
        EXPECT_DOUBLE_EQ(atlas::zip_get<0>(tn), 3.0);
        EXPECT_DOUBLE_EQ(atlas::zip_get<1>(tn), 30.0);
    }
}

TEST(ZipIterator, EqualityAndInequalityCompareUnderlyingIterators) {
    std::vector<int> a { 1, 2, 3, 4 };
    std::vector<int> b { 10, 20, 30, 40 };

    const auto it0 = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));
    const auto it1 = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));
    const auto it2 = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin() + 1, b.begin() + 1));

    EXPECT_TRUE(it0 == it1);
    EXPECT_FALSE(it0 != it1);

    EXPECT_TRUE(it0 != it2);
    EXPECT_FALSE(it0 == it2);
}

TEST(ZipIterator, PlusNAdvancesAndMinusComputesDistance) {
    std::vector<double> a { 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> b { 10.0, 20.0, 30.0, 40.0, 50.0 };

    const auto begin = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));
    const auto end   = atlas::make_zip_iterator(atlas::make_zip_tuple(a.end(), b.end()));

    const auto it = begin + 3;
    {
        const auto t = *it;
        EXPECT_DOUBLE_EQ(atlas::zip_get<0>(t), 4.0);
        EXPECT_DOUBLE_EQ(atlas::zip_get<1>(t), 40.0);
    }

    EXPECT_EQ(end - begin, static_cast<std::ptrdiff_t>(a.size()));
    EXPECT_EQ(it - begin, static_cast<std::ptrdiff_t>(3));
    EXPECT_EQ(begin - it, static_cast<std::ptrdiff_t>(-3));
}

TEST(ZipIterator, CanIterateLikeAForwardIterator) {
    std::vector<double> a { 1.0, 2.0, 3.0 };
    std::vector<double> b { 10.0, 20.0, 30.0 };

    auto it         = atlas::make_zip_iterator(atlas::make_zip_tuple(a.begin(), b.begin()));
    const auto it_e = atlas::make_zip_iterator(atlas::make_zip_tuple(a.end(), b.end()));

    std::size_t count = 0;
    for (; it != it_e; ++it) {
        auto t           = *it;
        const double sum = atlas::zip_get<0>(t) + atlas::zip_get<1>(t);

        if (count == 0) EXPECT_DOUBLE_EQ(sum, 11.0);
        if (count == 1) EXPECT_DOUBLE_EQ(sum, 22.0);
        if (count == 2) EXPECT_DOUBLE_EQ(sum, 33.0);

        ++count;
    }

    EXPECT_EQ(count, static_cast<std::size_t>(3));
}
