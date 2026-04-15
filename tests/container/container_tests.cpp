#include "../utilities/tests_utils.h"

#include <atlas/container/container.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

using namespace atlas;

TEST(Container4, DefaultConstructorValueInitializes) {

    Container4<double> c;

    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
    EXPECT_DOUBLE_EQ(c[2], 0.0);
    EXPECT_DOUBLE_EQ(c[3], 0.0);

    EXPECT_EQ(Container4<double>::size(), static_cast<std::size_t>(4));

    EXPECT_FALSE(Container4<double>::empty());
}

TEST(Container4, VariadicConstructorFillsInOrder) {

    constexpr Container4<double> c(1.0, 2.0, 3.0, 4.0);

    EXPECT_DOUBLE_EQ(c[0], 1.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);
    EXPECT_DOUBLE_EQ(c[2], 3.0);
    EXPECT_DOUBLE_EQ(c[3], 4.0);
}

TEST(Container4, BracketAndAtAccess) {

    Container4<int> c(1, 2, 3, 4);

    EXPECT_EQ(c[2], 3);

    c[2] = 99;

    EXPECT_EQ(c[2], 99);

    EXPECT_EQ(c.at(0), 1);
    EXPECT_EQ(c.at(3), 4);
}

TEST(Container4, NamedAccessorsReferenceStorage) {

    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    EXPECT_DOUBLE_EQ(c.a(), 1.0);
    EXPECT_DOUBLE_EQ(c.b(), 2.0);
    EXPECT_DOUBLE_EQ(c.c(), 3.0);
    EXPECT_DOUBLE_EQ(c.d(), 4.0);

    c.a() = -1.0;
    c.b() = -2.0;
    c.c() = -3.0;
    c.d() = -4.0;

    EXPECT_DOUBLE_EQ(c[0], -1.0);
    EXPECT_DOUBLE_EQ(c[1], -2.0);
    EXPECT_DOUBLE_EQ(c[2], -3.0);
    EXPECT_DOUBLE_EQ(c[3], -4.0);
}

TEST(Container4, DataBeginEndAreConsistent) {

    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    double* p = c.data();
    ASSERT_NE(p, nullptr);

    EXPECT_TRUE(c.begin() == p);

    EXPECT_TRUE(c.end() == p + 4);

    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    p[2] = 9.0;

    EXPECT_DOUBLE_EQ(c[2], 9.0);
}

TEST(Container4, FillOverwritesAllElements) {

    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    c.fill(7.5);

    EXPECT_DOUBLE_EQ(c[0], 7.5);
    EXPECT_DOUBLE_EQ(c[1], 7.5);
    EXPECT_DOUBLE_EQ(c[2], 7.5);
    EXPECT_DOUBLE_EQ(c[3], 7.5);
}

TEST(Container4, SwapExchangesContents) {

    Container4<int> a(1, 2, 3, 4);
    Container4<int> b(9, 8, 7, 6);

    a.swap(b);

    EXPECT_EQ(a[0], 9);
    EXPECT_EQ(a[1], 8);
    EXPECT_EQ(a[2], 7);
    EXPECT_EQ(a[3], 6);

    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
    EXPECT_EQ(b[3], 4);
}

TEST(Container, AliasesHaveExpectedTypes) {

    static_assert(std::is_same_v<Container2<double>, Container<double, 2>>);
    static_assert(std::is_same_v<Container3<double>, Container<double, 3>>);
    static_assert(std::is_same_v<Container4<double>, Container<double, 4>>);
}

TEST(TriangleContainer4, HoldsFourVector3) {

    const math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const math::Vector<double, 3> c(0.0, 1.0, 0.0);
    const math::Vector<double, 3> n(0.0, 0.0, 1.0);

    TriangleContainer4<double> t(a, b, c, n);

    EXPECT_TRUE(test::vec_near(t.a(), a, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.b(), b, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.c(), c, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.d(), n, static_cast<double>(eps)));
}

TEST(Container4, NamedAccessorsAreReferences) {

    using C = Container4<double>;

    static_assert(std::is_same_v<decltype(std::declval<C&>().a()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().a()), const double&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().b()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().b()), const double&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().c()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().c()), const double&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().d()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().d()), const double&>);
}