#include <atlas/container/container.h>

#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {

using atlas::Container;
using atlas::Container2;
using atlas::Container3;
using atlas::Container4;
using atlas::TriangleContainer4;
using atlas::Vector3;
using atlas::eps;

}

TEST(Container4, DefaultConstructorValueInitializes) {
    Container4<float> c;

    EXPECT_FLOAT_EQ(c[0], 0.0f);
    EXPECT_FLOAT_EQ(c[1], 0.0f);
    EXPECT_FLOAT_EQ(c[2], 0.0f);
    EXPECT_FLOAT_EQ(c[3], 0.0f);

    EXPECT_EQ(Container4<float>::size(), static_cast<std::size_t>(4));

    EXPECT_FALSE(Container4<float>::empty());
}

TEST(Container4, VariadicConstructorFillsInOrder) {
    constexpr Container4<float> c(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_FLOAT_EQ(c[0], 1.0f);
    EXPECT_FLOAT_EQ(c[1], 2.0f);
    EXPECT_FLOAT_EQ(c[2], 3.0f);
    EXPECT_FLOAT_EQ(c[3], 4.0f);
}

TEST(Container4, BracketAndAtAccess) {
    Container4<int> c(1, 2, 3, 4);

    EXPECT_EQ(c[2], 3);

    c[2] = 99;

    EXPECT_EQ(c[2], 99);

    EXPECT_EQ(c.at(0), 1);
    EXPECT_EQ(c.at(3), 4);
}

TEST(Container4, AtThrowsOutOfRange) {
    Container4<int> c(1, 2, 3, 4);

    EXPECT_THROW(static_cast<void>(c.at(4)), std::out_of_range);
}

TEST(Container4, NamedAccessorsReferenceStorage) {
    Container4<float> c(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_FLOAT_EQ(c.a(), 1.0f);
    EXPECT_FLOAT_EQ(c.b(), 2.0f);
    EXPECT_FLOAT_EQ(c.c(), 3.0f);
    EXPECT_FLOAT_EQ(c.d(), 4.0f);

    c.a() = -1.0f;
    c.b() = -2.0f;
    c.c() = -3.0f;
    c.d() = -4.0f;

    EXPECT_FLOAT_EQ(c[0], -1.0f);
    EXPECT_FLOAT_EQ(c[1], -2.0f);
    EXPECT_FLOAT_EQ(c[2], -3.0f);
    EXPECT_FLOAT_EQ(c[3], -4.0f);
}

TEST(Container4, DataBeginEndAreConsistent) {
    Container4<float> c(1.0f, 2.0f, 3.0f, 4.0f);

    float* p = c.data();
    ASSERT_NE(p, nullptr);

    EXPECT_TRUE(c.begin() == p);

    EXPECT_TRUE(c.end() == p + 4);

    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);

    p[2] = 9.0f;

    EXPECT_FLOAT_EQ(c[2], 9.0f);
}

TEST(Container4, FillOverwritesAllElements) {
    Container4<float> c(1.0f, 2.0f, 3.0f, 4.0f);

    c.fill(7.5f);

    EXPECT_FLOAT_EQ(c[0], 7.5f);
    EXPECT_FLOAT_EQ(c[1], 7.5f);
    EXPECT_FLOAT_EQ(c[2], 7.5f);
    EXPECT_FLOAT_EQ(c[3], 7.5f);
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
    static_assert(std::is_same_v<Container2<float>, Container<float, 2>>);
    static_assert(std::is_same_v<Container3<float>, Container<float, 3>>);
    static_assert(std::is_same_v<Container4<float>, Container<float, 4>>);
    static_assert(std::is_same_v<TriangleContainer4, Container<Vector3, 4>>);
}

TEST(TriangleContainer4, HoldsFourVector3) {
    const Vector3 a(0.0f, 0.0f, 0.0f);
    const Vector3 b(1.0f, 0.0f, 0.0f);
    const Vector3 c(0.0f, 1.0f, 0.0f);
    const Vector3 n(0.0f, 0.0f, 1.0f);

    TriangleContainer4 t(a, b, c, n);

    EXPECT_NEAR((t.a() - a).length(), 0.0f, eps);
    EXPECT_NEAR((t.b() - b).length(), 0.0f, eps);
    EXPECT_NEAR((t.c() - c).length(), 0.0f, eps);
    EXPECT_NEAR((t.d() - n).length(), 0.0f, eps);
}

TEST(Container4, NamedAccessorsAreReferences) {
    using C = Container4<float>;

    static_assert(std::is_same_v<decltype(std::declval<C&>().a()), float&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().a()), const float&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().b()), float&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().b()), const float&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().c()), float&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().c()), const float&>);

    static_assert(std::is_same_v<decltype(std::declval<C&>().d()), float&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().d()), const float&>);
}
