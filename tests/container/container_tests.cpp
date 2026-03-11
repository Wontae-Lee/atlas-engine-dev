#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

using namespace atlas;

TEST(Container4, DefaultConstructorValueInitializes) {

    // Default construction should value-initialize all elements.
    // For arithmetic types (e.g., double), value-initialization yields zero.
    Container4<double> c;

    // Verify each element is zero-initialized.
    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
    EXPECT_DOUBLE_EQ(c[2], 0.0);
    EXPECT_DOUBLE_EQ(c[3], 0.0);

    // Log the default-initialized values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Default-initialized Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Container4 is a fixed-size container, so size() must be a constant 4.
    EXPECT_EQ(Container4<double>::size(), static_cast<std::size_t>(4));

    // A fixed-size container with N=4 must never be empty().
    EXPECT_FALSE(Container4<double>::empty());
}

TEST(Container4, VariadicConstructorFillsInOrder) {

    // Variadic constructor should populate the underlying storage in index order:
    // (a,b,c,d) -> [0,1,2,3]
    constexpr Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // Log the constructed values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Constructed Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Verify ordering is preserved.
    EXPECT_DOUBLE_EQ(c[0], 1.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);
    EXPECT_DOUBLE_EQ(c[2], 3.0);
    EXPECT_DOUBLE_EQ(c[3], 4.0);
}

TEST(Container4, BracketAndAtAccess) {

    // Construct with known integer values to test both read/write access.
    Container4<int> c(1, 2, 3, 4);

    // Log the initial values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Initial Container4<int> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // operator[] should provide direct access without bounds checking.
    EXPECT_EQ(c[2], 3);

    // operator[] should return a mutable reference for non-const containers.
    c[2] = 99;

    // Log the mutated value for debugging purposes (optional).
    logger::info() << "\n"
                   << "After mutation, Container4<int> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Verify the mutation took effect.
    EXPECT_EQ(c[2], 99);

    // at() should access elements by index (often bounds-checked).
    // We only verify correctness for valid indices here.
    EXPECT_EQ(c.at(0), 1);
    EXPECT_EQ(c.at(3), 4);
}

TEST(Container4, NamedAccessorsReferenceStorage) {

    // Named accessors (a,b,c,d) should reference the same underlying storage
    // as operator[] (indices 0..3).
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // Log the initial values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Initial Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Verify read access through named getters.
    EXPECT_DOUBLE_EQ(c.a(), 1.0);
    EXPECT_DOUBLE_EQ(c.b(), 2.0);
    EXPECT_DOUBLE_EQ(c.c(), 3.0);
    EXPECT_DOUBLE_EQ(c.d(), 4.0);

    // Mutate through named accessors.
    // If these return references, the underlying storage must change.
    c.a() = -1.0;
    c.b() = -2.0;
    c.c() = -3.0;
    c.d() = -4.0;

    // Log the mutated values for debugging purposes (optional).
    logger::info() << "\n"
                   << "After mutation through named accessors, Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Verify mutations are visible through operator[].
    EXPECT_DOUBLE_EQ(c[0], -1.0);
    EXPECT_DOUBLE_EQ(c[1], -2.0);
    EXPECT_DOUBLE_EQ(c[2], -3.0);
    EXPECT_DOUBLE_EQ(c[3], -4.0);
}

TEST(Container4, DataBeginEndAreConsistent) {

    // data(), begin(), and end() should form a consistent contiguous range.
    // This test ensures Container4 behaves like a small std::array-style container.
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // Log the initial values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Initial Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // data() must return a pointer to contiguous storage.
    double* p = c.data();
    ASSERT_NE(p, nullptr);

    // begin() should equal data(); end() should be data() + size().
    EXPECT_TRUE(c.begin() == p);
    EXPECT_TRUE(c.end() == p + 4);

    // Log the raw pointer values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Raw pointer values from data():\n"
                   << "p[0] = " << p[0] << "\n"
                   << "p[1] = " << p[1] << "\n"
                   << "p[2] = " << p[2] << "\n"
                   << "p[3] = " << p[3] << "\n";

    // Verify raw pointer indexing matches container indexing.
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    // Mutate through the raw pointer and ensure the container view updates.
    p[2] = 9.0;

    // Log the mutated values for debugging purposes (optional).
    logger::info() << "\n"
                   << "After mutation through raw pointer, Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    EXPECT_DOUBLE_EQ(c[2], 9.0);
}

TEST(Container4, FillOverwritesAllElements) {

    // fill(value) should overwrite every element in the container.
    Container4<double> c(1.0, 2.0, 3.0, 4.0);

    // Log the initial values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Initial Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Call fill() with a new value.
    c.fill(7.5);

    // Log the filled values for debugging purposes (optional).
    logger::info() << "\n"
                   << "After fill(7.5), Container4<double> values:\n"
                   << "c[0] = " << c[0] << "\n"
                   << "c[1] = " << c[1] << "\n"
                   << "c[2] = " << c[2] << "\n"
                   << "c[3] = " << c[3] << "\n";

    // Verify every slot is overwritten.
    EXPECT_DOUBLE_EQ(c[0], 7.5);
    EXPECT_DOUBLE_EQ(c[1], 7.5);
    EXPECT_DOUBLE_EQ(c[2], 7.5);
    EXPECT_DOUBLE_EQ(c[3], 7.5);
}

TEST(Container4, SwapExchangesContents) {

    // swap() should exchange the contents of two containers element-wise.
    Container4<int> a(1, 2, 3, 4);
    Container4<int> b(9, 8, 7, 6);

    // Log the initial values for debugging purposes (optional).
    logger::info() << "\n"
                   << "Initial Container4<int> values:\n"
                   << "a[0] = " << a[0] << "\n"
                   << "a[1] = " << a[1] << "\n"
                   << "a[2] = " << a[2] << "\n"
                   << "a[3] = " << a[3] << "\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n"
                   << "b[3] = " << b[3] << "\n";

    // Perform the swap.
    a.swap(b);

    // Log the swapped values for debugging purposes (optional).
    logger::info() << "\n"
                   << "After swap, Container4<int> values:\n"
                   << "a[0] = " << a[0] << "\n"
                   << "a[1] = " << a[1] << "\n"
                   << "a[2] = " << a[2] << "\n"
                   << "a[3] = " << a[3] << "\n"
                   << "b[0] = " << b[0] << "\n"
                   << "b[1] = " << b[1] << "\n"
                   << "b[2] = " << b[2] << "\n"
                   << "b[3] = " << b[3] << "\n";

    // After swap, `a` should contain the old contents of `b`.
    EXPECT_EQ(a[0], 9);
    EXPECT_EQ(a[1], 8);
    EXPECT_EQ(a[2], 7);
    EXPECT_EQ(a[3], 6);

    // And `b` should contain the old contents of `a`.
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
    EXPECT_EQ(b[2], 3);
    EXPECT_EQ(b[3], 4);
}

TEST(Container, AliasesHaveExpectedTypes) {

    // The ContainerN aliases must map to the generic Container<T, N>.
    // These are compile-time checks: if the alias changes, this test fails to compile.
    static_assert(std::is_same_v<Container2<double>, Container<double, 2>>);
    static_assert(std::is_same_v<Container3<double>, Container<double, 3>>);
    static_assert(std::is_same_v<Container4<double>, Container<double, 4>>);
}

TEST(TriangleContainer4, HoldsFourVector3) {

    // Define a right triangle in the XY-plane with a +Z normal.
    const math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const math::Vector<double, 3> c(0.0, 1.0, 0.0);
    const math::Vector<double, 3> n(0.0, 0.0, 1.0);

    // Construct the container with (a,b,c,n).
    TriangleContainer4<double> t(a, b, c, n);

    // Verify each named accessor returns the expected vector.
    EXPECT_TRUE(test::vec_near(t.a(), a, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.b(), b, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.c(), c, static_cast<double>(eps)));
    EXPECT_TRUE(test::vec_near(t.d(), n, static_cast<double>(eps)));
}

TEST(Container4, NamedAccessorsAreReferences) {

    // This test verifies the API contract of named accessors:
    // - for non-const containers, a()/b()/c()/d() return mutable references (T&)
    // - for const containers, they return const references (const T&)
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
