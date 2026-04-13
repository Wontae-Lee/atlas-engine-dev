#include "../../utilities/tests_utils.h"

#include <cstddef>
#include <gtest/gtest.h>

TEST(MatrixRC, DefaultConstructorIsZero) {
    using M = atlas::math::Matrix<double, 2, 3>;
    const M m;

    for (std::size_t i = 0; i < 2 * 3; ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(MatrixRC, ScalarFillConstructorFillsAllEntries) {
    using M = atlas::math::Matrix<double, 2, 3>;
    const M m(3.5);

    for (std::size_t i = 0; i < 2 * 3; ++i) {
        EXPECT_DOUBLE_EQ(m[i], 3.5);
    }
}

TEST(MatrixRC, InitializerListPadsWithZeroAndIsRowMajor) {
    using M = atlas::math::Matrix<double, 2, 3>;
    const M m { 1.0, 2.0, 3.0, 4.0 };

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[1], 2.0);
    EXPECT_DOUBLE_EQ(m[2], 3.0);
    EXPECT_DOUBLE_EQ(m[3], 4.0);
    EXPECT_DOUBLE_EQ(m[4], 0.0);
    EXPECT_DOUBLE_EQ(m[5], 0.0);

    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 0.0);

    EXPECT_DOUBLE_EQ(m.at(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 4.0);
}

TEST(MatrixRC, OperatorAndAtAreMutable) {
    using M = atlas::math::Matrix<double, 2, 3>;
    M m;

    m(1, 2) = 7.0;
    EXPECT_DOUBLE_EQ(m(1, 2), 7.0);

    m.at(0, 1) = -2.0;
    EXPECT_DOUBLE_EQ(m(0, 1), -2.0);

    m[5] = 9.0;
    EXPECT_DOUBLE_EQ(m(1, 2), 9.0);
}

TEST(MatrixRC, SetFillAndSetZero) {
    using M = atlas::math::Matrix<double, 2, 3>;
    M m;

    m.set(4.0);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(m[i], 4.0);
    }

    m.set_zero();
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(MatrixRC, SetValuesWritesRowMajorOrder) {
    using M = atlas::math::Matrix<double, 2, 3>;
    M m;
    m.set_values(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0);

    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 6.0);

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[1], 2.0);
    EXPECT_DOUBLE_EQ(m[2], 3.0);
    EXPECT_DOUBLE_EQ(m[3], 4.0);
    EXPECT_DOUBLE_EQ(m[4], 5.0);
    EXPECT_DOUBLE_EQ(m[5], 6.0);
}

TEST(MatrixRC, ScalarOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);
    using M            = atlas::math::Matrix<double, 2, 3>;

    M m;
    m.set_values(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0);

    m += 2.0;
    EXPECT_TRUE(atlas::test::near(m[0], 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m[5], 8.0, eps));

    m -= 1.0;
    EXPECT_TRUE(atlas::test::near(m[0], 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m[5], 7.0, eps));

    m *= 2.0;
    EXPECT_TRUE(atlas::test::near(m[0], 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m[5], 14.0, eps));

    m /= 4.0;
    EXPECT_TRUE(atlas::test::near(m[0], 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m[5], 3.5, eps));
}

TEST(MatrixRC, MatrixOpsAreElementWiseInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);
    using M            = atlas::math::Matrix<double, 2, 3>;

    M a;
    a.set_values(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0);

    M b;
    b.set_values(
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0);

    a += b;
    EXPECT_TRUE(atlas::test::near(a(0, 0), 11.0, eps));
    EXPECT_TRUE(atlas::test::near(a(1, 2), 66.0, eps));

    a -= b;
    EXPECT_TRUE(atlas::test::near(a(0, 0), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a(1, 2), 6.0, eps));

    a *= b;
    EXPECT_TRUE(atlas::test::near(a(0, 0), 1.0 * 10.0, eps));
    EXPECT_TRUE(atlas::test::near(a(1, 2), 6.0 * 60.0, eps));

    a /= b;
    EXPECT_TRUE(atlas::test::near(a(0, 0), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a(1, 2), 6.0, eps));
}

TEST(MatrixRC, ExactEquality) {
    using M = atlas::math::Matrix<double, 2, 3>;

    const M a { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const M b { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const M c { 1.0, 2.0, 3.0, 4.0, 5.0, 7.0 };

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(MatrixRC, Matmul_MatrixMatrix) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    using A = atlas::math::Matrix<double, 2, 3>;
    using B = atlas::math::Matrix<double, 3, 4>;
    using C = atlas::math::Matrix<double, 2, 4>;

    const A a {
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0
    };

    const B b {
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0
    };

    const C out = atlas::math::matmul<double, 2, 3, 4>(a, b);

    EXPECT_TRUE(atlas::test::near(out(0, 0), 1.0 * 7.0 + 2.0 * 11.0 + 3.0 * 15.0, eps));
    EXPECT_TRUE(atlas::test::near(out(0, 1), 1.0 * 8.0 + 2.0 * 12.0 + 3.0 * 16.0, eps));
    EXPECT_TRUE(atlas::test::near(out(0, 2), 1.0 * 9.0 + 2.0 * 13.0 + 3.0 * 17.0, eps));
    EXPECT_TRUE(atlas::test::near(out(0, 3), 1.0 * 10.0 + 2.0 * 14.0 + 3.0 * 18.0, eps));

    EXPECT_TRUE(atlas::test::near(out(1, 0), 4.0 * 7.0 + 5.0 * 11.0 + 6.0 * 15.0, eps));
    EXPECT_TRUE(atlas::test::near(out(1, 1), 4.0 * 8.0 + 5.0 * 12.0 + 6.0 * 16.0, eps));
    EXPECT_TRUE(atlas::test::near(out(1, 2), 4.0 * 9.0 + 5.0 * 13.0 + 6.0 * 17.0, eps));
    EXPECT_TRUE(atlas::test::near(out(1, 3), 4.0 * 10.0 + 5.0 * 14.0 + 6.0 * 18.0, eps));
}

TEST(MatrixRC, Matmul_MatrixVector) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    using A = atlas::math::Matrix<double, 2, 3>;
    using X = atlas::math::Vector<double, 3>;
    using Y = atlas::math::Vector<double, 2>;

    const A a {
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0
    };

    const X x(7.0, 8.0, 9.0);

    const Y y = atlas::math::matmul<double, 2, 3>(a, x);

    EXPECT_TRUE(atlas::test::near(y[0], 1.0 * 7.0 + 2.0 * 8.0 + 3.0 * 9.0, eps));
    EXPECT_TRUE(atlas::test::near(y[1], 4.0 * 7.0 + 5.0 * 8.0 + 6.0 * 9.0, eps));
}