#include <atlas/container/container.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using atlas::Container;
using atlas::Container2;
using atlas::Container3;
using atlas::Container4;

}

TEST(Container, DefaultConstructionValueInitializes) {
    constexpr Container<int, 3> values {};

    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 0);
    EXPECT_EQ(values[2], 0);
}

TEST(Container, VariadicConstructionFillsEverySlotInOrder) {
    const Container<int, 3> values(10, 20, 30);

    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[1], 20);
    EXPECT_EQ(values[2], 30);
}

TEST(Container, SubscriptWritesThroughMutableReference) {
    Container<int, 2> values(1, 2);

    values[0] = 7;
    values[1] = 8;

    EXPECT_EQ(values[0], 7);
    EXPECT_EQ(values[1], 8);
}

TEST(Container, AtReturnsElementInRange) {
    Container<int, 3> values(4, 5, 6);

    EXPECT_EQ(values.at(0), 4);
    EXPECT_EQ(values.at(2), 6);

    values.at(1) = 55;
    EXPECT_EQ(values[1], 55);
}

TEST(Container, AtThrowsOnOutOfRangeIndex) {
    const Container<int, 3> values(1, 2, 3);

    EXPECT_THROW(static_cast<void>(values.at(3)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(values.at(100)), std::out_of_range);
}

TEST(Container, NamedAccessorsAliasLeadingElements) {
    Container<int, 4> values(1, 2, 3, 4);

    EXPECT_EQ(values.a(), 1);
    EXPECT_EQ(values.b(), 2);
    EXPECT_EQ(values.c(), 3);
    EXPECT_EQ(values.d(), 4);

    values.a() = 11;
    EXPECT_EQ(values[0], 11);
}

TEST(Container, SizeAndEmptyReflectTemplateCount) {
    EXPECT_EQ((Container<int, 3>::size()), std::size_t { 3 });
    EXPECT_FALSE((Container<int, 3>::empty()));
}

TEST(Container, ZeroLengthReportsEmptyWithEqualBeginEnd) {
    Container<int, 0> values {};

    EXPECT_EQ(values.size(), std::size_t { 0 });
    EXPECT_TRUE(values.empty());
    EXPECT_EQ(values.begin(), values.end());
}

TEST(Container, IterationVisitsElementsInStorageOrder) {
    const Container<int, 4> values(2, 4, 6, 8);

    int expected = 2;
    for (const int element : values) {
        EXPECT_EQ(element, expected);
        expected += 2;
    }
    EXPECT_EQ(expected, 10);

    EXPECT_EQ(values.end() - values.begin(), 4);
}

TEST(Container, DataPointsAtContiguousStorage) {
    Container<int, 3> values(1, 2, 3);

    EXPECT_EQ(values.data(), &values[0]);
    EXPECT_EQ(values.data()[2], 3);
}

TEST(Container, FillOverwritesEveryElement) {
    Container<int, 4> values(1, 2, 3, 4);

    values.fill(9);

    EXPECT_EQ(values[0], 9);
    EXPECT_EQ(values[1], 9);
    EXPECT_EQ(values[2], 9);
    EXPECT_EQ(values[3], 9);
}

TEST(Container, SwapExchangesContentsElementWise) {
    Container<int, 3> lhs(1, 2, 3);
    Container<int, 3> rhs(4, 5, 6);

    lhs.swap(rhs);

    EXPECT_EQ(lhs[0], 4);
    EXPECT_EQ(lhs[2], 6);
    EXPECT_EQ(rhs[0], 1);
    EXPECT_EQ(rhs[2], 3);
}

TEST(Container, IsTriviallyCopyableWhenElementIs) {
    // The whole point of the inline storage: a Container of a trivially copyable
    // element stays trivially copyable and device-capturable.
    static_assert(std::is_trivially_copyable_v<Container<int, 4>>);
    EXPECT_TRUE((std::is_trivially_copyable_v<Container<float, 3>>));
}

TEST(Container, ConvenienceAliasesMatchExplicitCounts) {
    EXPECT_EQ((Container2<int>::size()), std::size_t { 2 });
    EXPECT_EQ((Container3<int>::size()), std::size_t { 3 });
    EXPECT_EQ((Container4<int>::size()), std::size_t { 4 });

    EXPECT_TRUE((std::is_same_v<Container2<int>, Container<int, 2>>));
    EXPECT_TRUE((std::is_same_v<Container4<float>, Container<float, 4>>));
}
