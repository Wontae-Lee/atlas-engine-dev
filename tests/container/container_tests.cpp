#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

using namespace atlas;

TEST(Container4, DefaultConstructorValueInitializes) {

    // Default construction should value-initialize every element.
    //
    // For arithmetic scalar types such as double, value-initialization
    // produces zero. This verifies the container does not leave its fixed
    // storage uninitialized on default construction.
    Container4<double> c;

    // Check each fixed slot explicitly so the test documents the exact
    // element layout and expected initial state.
    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
    EXPECT_DOUBLE_EQ(c[2], 0.0);
    EXPECT_DOUBLE_EQ(c[3], 0.0);

    // Container4 is a fixed-size container alias, so size() must always
    // report 4 independently of runtime state.
    EXPECT_EQ(Container4<double>::size(), static_cast<std::size_t>(4));

    // A fixed-size container with four elements can never be empty.
    EXPECT_FALSE(Container4<double>::empty());
}

TEST(Container4, VariadicConstructorFillsInOrder) {

    // The variadic/value constructor should place arguments into storage
    // in index order:
    //   (x0, x1, x2, x3) -> [0], [1], [2], [3]
    //
    // constexpr is used here so the test also exercises compile-time
    // friendliness of the constructor.
    constexpr Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // Verify that constructor argument order is preserved exactly.
    EXPECT_DOUBLE_EQ(c[0], 1.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);
    EXPECT_DOUBLE_EQ(c[2], 3.0);
    EXPECT_DOUBLE_EQ(c[3], 4.0);
}

TEST(Container4, BracketAndAtAccess) {

    // Initialize with known values so both read and write semantics can
    // be checked deterministically.
    Container4<int> c(1, 2, 3, 4);

    // operator[] should provide direct indexed access for valid indices.
    // This confirms the third slot initially contains the third argument.
    EXPECT_EQ(c[2], 3);

    // For non-const containers, operator[] must return a mutable reference.
    // Mutating through the subscript operator must update the underlying
    // stored value rather than a temporary copy.
    c[2] = 99;

    // The mutation should be visible immediately through the same view.
    EXPECT_EQ(c[2], 99);

    // at() should provide equivalent access for valid indices.
    // This test intentionally checks only valid indices and does not make
    // assumptions about whether out-of-range handling is assert-based,
    // exception-based, or unavailable in certain build modes.
    EXPECT_EQ(c.at(0), 1);
    EXPECT_EQ(c.at(3), 4);
}

TEST(Container4, NamedAccessorsReferenceStorage) {

    // Named accessors a()/b()/c()/d() should map to the same underlying
    // fixed storage as indices 0/1/2/3 respectively.
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // First verify read access through the named API.
    EXPECT_DOUBLE_EQ(c.a(), 1.0);
    EXPECT_DOUBLE_EQ(c.b(), 2.0);
    EXPECT_DOUBLE_EQ(c.c(), 3.0);
    EXPECT_DOUBLE_EQ(c.d(), 4.0);

    // Then mutate through the named accessors.
    // If these accessors return references to storage, the underlying
    // container contents must change accordingly.
    c.a() = -1.0;
    c.b() = -2.0;
    c.c() = -3.0;
    c.d() = -4.0;

    // Verify the mutations are visible through operator[], proving that
    // both APIs refer to the same storage rather than separate values.
    EXPECT_DOUBLE_EQ(c[0], -1.0);
    EXPECT_DOUBLE_EQ(c[1], -2.0);
    EXPECT_DOUBLE_EQ(c[2], -3.0);
    EXPECT_DOUBLE_EQ(c[3], -4.0);
}

TEST(Container4, DataBeginEndAreConsistent) {

    // A small fixed-size container should behave like contiguous storage,
    // similar to std::array. This test verifies consistency between:
    //   - data()
    //   - begin()
    //   - end()
    //   - indexed element access
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // data() must point at the first element of contiguous storage.
    double* p = c.data();
    ASSERT_NE(p, nullptr);

    // begin() should be identical to data().
    EXPECT_TRUE(c.begin() == p);

    // end() should point one-past-the-last element.
    EXPECT_TRUE(c.end() == p + 4);

    // Raw pointer indexing must observe the same values as container indexing.
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    // Mutating through the raw pointer must update the container contents,
    // which further confirms the storage is contiguous and shared.
    p[2] = 9.0;

    EXPECT_DOUBLE_EQ(c[2], 9.0);
}

TEST(Container4, FillOverwritesAllElements) {

    // Start from distinct values so it is obvious whether fill() touches
    // every slot or only a subset.
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // fill(value) should overwrite the entire fixed storage.
    c.fill(7.5);

    // Every element must now hold the same fill value.
    EXPECT_DOUBLE_EQ(c[0], 7.5);
    EXPECT_DOUBLE_EQ(c[1], 7.5);
    EXPECT_DOUBLE_EQ(c[2], 7.5);
    EXPECT_DOUBLE_EQ(c[3], 7.5);
}

TEST(Container4, SwapExchangesContents) {

    // Prepare two containers with clearly different contents so the swap
    // result is easy to verify element-by-element.
    Container4<int> a(1, 2, 3, 4);
    Container4<int> b(9, 8, 7, 6);

    // swap() should exchange the full contents of the two fixed containers.
    a.swap(b);

    // `a` must now contain what `b` originally held.
    EXPECT_EQ(a[0], 9);
    EXPECT_EQ(a[1], 8);
    EXPECT_EQ(a[2], 7);
    EXPECT_EQ(a[3], 6);

    // `b` must now contain what `a` originally held.
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
    EXPECT_EQ(b[3], 4);
}

TEST(Container, AliasesHaveExpectedTypes) {

    // These are compile-time API-contract checks.
    //
    // The project exposes convenience aliases such as Container2/3/4.
    // They must remain exact aliases of the generic Container<T, N>.
    // If an alias changes unexpectedly, this test should fail at compile time.
    static_assert(std::is_same_v<Container2<double>, Container<double, 2>>);
    static_assert(std::is_same_v<Container3<double>, Container<double, 3>>);
    static_assert(std::is_same_v<Container4<double>, Container<double, 4>>);
}

TEST(TriangleContainer4, HoldsFourVector3) {

    // Define a simple right triangle in the XY plane and an associated
    // +Z unit normal. This matches a common triangle representation:
    //   a, b, c = triangle vertices
    //   d       = associated normal or auxiliary vector
    const math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const math::Vector<double, 3> c(0.0, 1.0, 0.0);
    const math::Vector<double, 3> n(0.0, 0.0, 1.0);

    // Construct the 4-slot triangle container in the expected order.
    TriangleContainer4<double> t(a, b, c, n);

    // Verify that each named accessor returns the vector stored in its slot.
    // vec_near is used instead of exact equality to follow the project's
    // vector comparison style and remain robust for floating-point code.
    EXPECT_TRUE(test::vec_near(t.a(), a, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.b(), b, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.c(), c, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.d(), n, static_cast<double>(eps)));
}

TEST(Container4, NamedAccessorsAreReferences) {

    // This test verifies the reference category of the named accessors.
    //
    // Required API contract:
    //  - non-const container -> mutable reference (T&)
    //  - const container     -> const reference (const T&)
    //
    // These are compile-time checks because return type correctness is
    // part of the interface contract, not just runtime behavior.
    using C = Container4<double>;

    // a()
    static_assert(std::is_same_v<decltype(std::declval<C&>().a()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().a()), const double&>);

    // b()
    static_assert(std::is_same_v<decltype(std::declval<C&>().b()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().b()), const double&>);

    // c()
    static_assert(std::is_same_v<decltype(std::declval<C&>().c()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().c()), const double&>);

    // d()
    static_assert(std::is_same_v<decltype(std::declval<C&>().d()), double&>);
    static_assert(std::is_same_v<decltype(std::declval<const C&>().d()), const double&>);
}