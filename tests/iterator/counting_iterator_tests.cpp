#include <atlas/iterator/counting_iterator.h>

#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

TEST(CountingIterator, DefaultConstructStartsAtZero) {
    atlas::counting_iterator<int> it;

    EXPECT_EQ(*it, 0);
    EXPECT_EQ(it.base(), 0);

    ++it;
    EXPECT_EQ(*it, 1);
    --it;
    EXPECT_EQ(*it, 0);
}

TEST(CountingIterator, ConstructFromStartValue) {
    atlas::counting_iterator<int> it(7);

    EXPECT_EQ(*it, 7);
    EXPECT_EQ(it.base(), 7);

    auto it2 = it;
    it2 += 5;
    EXPECT_EQ(*it2, 12);
    EXPECT_EQ(it2.base(), 12);
}

TEST(CountingIterator, DerefAndIndexingReturnExpectedValues) {
    atlas::counting_iterator<int> it(10);

    EXPECT_EQ(*it, 10);
    EXPECT_EQ(it[0], 10);
    EXPECT_EQ(it[1], 11);
    EXPECT_EQ(it[5], 15);
    EXPECT_EQ(it[-3], 7);

    EXPECT_EQ(*it, 10);
}

TEST(CountingIterator, IteratorArithmeticPlusMinusAndDistance) {
    atlas::counting_iterator<int> a(3);
    atlas::counting_iterator<int> b(11);

    EXPECT_EQ(b - a, 8);
    EXPECT_EQ(a - b, -8);

    const auto c = a + 4;
    EXPECT_EQ(*c, 7);

    const auto e = b - 6;
    EXPECT_EQ(*e, 5);

    a += 10;
    EXPECT_EQ(*a, 13);
    a -= 2;
    EXPECT_EQ(*a, 11);
}

TEST(CountingIterator, ComparisonsBehaveLikeValues) {
    atlas::counting_iterator<int> a(5);
    atlas::counting_iterator<int> b(7);
    atlas::counting_iterator<int> c(5);

    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a != c);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);

    EXPECT_TRUE(a <= c);
    EXPECT_TRUE(a <= b);

    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(c >= a);
}

TEST(CountingIterator, WorksWithUnsignedValueTypeDistanceIsSigned) {
    using U = std::size_t;

    atlas::counting_iterator<U> a(static_cast<U>(2));
    atlas::counting_iterator<U> b(static_cast<U>(9));

    EXPECT_TRUE((std::is_same_v<typename atlas::counting_iterator<U>::difference_type, std::ptrdiff_t>));

    const auto d1 = b - a;
    const auto d2 = a - b;

    EXPECT_EQ(d1, static_cast<std::ptrdiff_t>(7));
    EXPECT_EQ(d2, static_cast<std::ptrdiff_t>(-7));
}

TEST(CountingIterator, PostIncrementAndPostDecrementReturnOldValue) {
    atlas::counting_iterator<int> it(4);

    const auto a = it++;
    EXPECT_EQ(*a, 4);
    EXPECT_EQ(*it, 5);

    const auto b = it--;
    EXPECT_EQ(*b, 5);
    EXPECT_EQ(*it, 4);
}
