#include "../../utilities/tests_utils.h"
#include <atlas/math/math.h>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

TEST(Matrix3x3, DefaultConstructorIsZero) {
    const atlas::math::Matrix<double, 3, 3> m;

    for (std::size_t i = 0; i < atlas::math::Matrix<double, 3, 3>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix3x3, DiagonalConstructorIsScaledIdentity) {
    const atlas::math::Matrix<double, 3, 3> m(3.5);

    EXPECT_DOUBLE_EQ(m.m00, 3.5);
    EXPECT_DOUBLE_EQ(m.m11, 3.5);
    EXPECT_DOUBLE_EQ(m.m22, 3.5);

    EXPECT_DOUBLE_EQ(m.m01, 0.0);
    EXPECT_DOUBLE_EQ(m.m02, 0.0);

    EXPECT_DOUBLE_EQ(m.m10, 0.0);
    EXPECT_DOUBLE_EQ(m.m12, 0.0);

    EXPECT_DOUBLE_EQ(m.m20, 0.0);
    EXPECT_DOUBLE_EQ(m.m21, 0.0);
}

TEST(Matrix3x3, ElementConstructorRowMajor) {
    const atlas::math::Matrix<double, 3, 3> m(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    EXPECT_DOUBLE_EQ(m.m00, 1.0);
    EXPECT_DOUBLE_EQ(m.m01, 2.0);
    EXPECT_DOUBLE_EQ(m.m02, 3.0);

    EXPECT_DOUBLE_EQ(m.m10, 4.0);
    EXPECT_DOUBLE_EQ(m.m11, 5.0);
    EXPECT_DOUBLE_EQ(m.m12, 6.0);

    EXPECT_DOUBLE_EQ(m.m20, 7.0);
    EXPECT_DOUBLE_EQ(m.m21, 8.0);
    EXPECT_DOUBLE_EQ(m.m22, 9.0);
}

TEST(Matrix3x3, InitializerListPadsWithZero) {
    const atlas::math::Matrix<double, 3, 3> m { 1.0, 2.0 };

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[1], 2.0);

    for (std::size_t i = 2; i < atlas::math::Matrix<double, 3, 3>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix3x3, RowsColsSize) {
    EXPECT_EQ(atlas::Matrix3x3<double>::rows(), static_cast<std::size_t>(3));
    EXPECT_EQ(atlas::Matrix3x3<double>::cols(), static_cast<std::size_t>(3));
    EXPECT_EQ(atlas::Matrix3x3<double>::size(), static_cast<std::size_t>(9));
}

TEST(Matrix3x3, DataPointerIsContiguousRowMajor) {
    atlas::math::Matrix<double, 3, 3> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    const double* p = m.data();
    ASSERT_NE(p, nullptr);

    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);

    EXPECT_DOUBLE_EQ(p[3], 4.0);
    EXPECT_DOUBLE_EQ(p[4], 5.0);
    EXPECT_DOUBLE_EQ(p[5], 6.0);

    EXPECT_DOUBLE_EQ(p[6], 7.0);
    EXPECT_DOUBLE_EQ(p[7], 8.0);
    EXPECT_DOUBLE_EQ(p[8], 9.0);

    double* pm = m.data();
    pm[4]      = -7.0;
    EXPECT_DOUBLE_EQ(m.m11, -7.0);
}

TEST(Matrix3x3, IndexAndAtAndCallOperator) {
    atlas::math::Matrix<double, 3, 3> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[4], 5.0);
    EXPECT_DOUBLE_EQ(m[8], 9.0);

    EXPECT_DOUBLE_EQ(m.at(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(m(2, 1), 8.0);

    m(0, 2) = -9.0;
    EXPECT_DOUBLE_EQ(m.m02, -9.0);
}

TEST(Matrix3x3, SetZeroAndSetIdentity) {
    atlas::math::Matrix<double, 3, 3> z(3.0);
    z.set_zero();
    for (std::size_t i = 0; i < atlas::math::Matrix<double, 3, 3>::size(); ++i) {
        EXPECT_DOUBLE_EQ(z[i], 0.0);
    }

    atlas::math::Matrix<double, 3, 3> i;
    i.set_identity();

    EXPECT_DOUBLE_EQ(i.m00, 1.0);
    EXPECT_DOUBLE_EQ(i.m11, 1.0);
    EXPECT_DOUBLE_EQ(i.m22, 1.0);

    EXPECT_DOUBLE_EQ(i.m01, 0.0);
    EXPECT_DOUBLE_EQ(i.m02, 0.0);

    EXPECT_DOUBLE_EQ(i.m10, 0.0);
    EXPECT_DOUBLE_EQ(i.m12, 0.0);

    EXPECT_DOUBLE_EQ(i.m20, 0.0);
    EXPECT_DOUBLE_EQ(i.m21, 0.0);
}

TEST(Matrix3x3, ScalarOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 3, 3> m;
    m.set_identity();

    m += 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 3.0, eps));

    m -= 1.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 2.0, eps));

    m *= 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 4.0, eps));

    m /= 4.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 1.0, eps));
}

TEST(Matrix3x3, MatrixOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 3, 3> a;
    a.set_identity();

    atlas::math::Matrix<double, 3, 3> b;
    b.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    a += b;
    EXPECT_TRUE(atlas::test::near(a.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m01, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m02, 3.0, eps));

    a -= b;
    EXPECT_TRUE(atlas::test::near(a.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m22, 1.0, eps));
}

TEST(Matrix3x3, TraceAndDeterminantIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 3, 3> m;
    m.set_identity();

    EXPECT_TRUE(atlas::test::near(m.trace(), 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.determinant(), 1.0, eps));
}

TEST(Matrix3x3, TransposeAndTransposed) {
    atlas::math::Matrix<double, 3, 3> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    const atlas::math::Matrix<double, 3, 3> t = m.transposed();

    EXPECT_DOUBLE_EQ(t.m00, 1.0);
    EXPECT_DOUBLE_EQ(t.m01, 4.0);
    EXPECT_DOUBLE_EQ(t.m02, 7.0);

    EXPECT_DOUBLE_EQ(t.m10, 2.0);
    EXPECT_DOUBLE_EQ(t.m11, 5.0);
    EXPECT_DOUBLE_EQ(t.m12, 8.0);

    EXPECT_DOUBLE_EQ(t.m20, 3.0);
    EXPECT_DOUBLE_EQ(t.m21, 6.0);
    EXPECT_DOUBLE_EQ(t.m22, 9.0);

    m.transpose();
    EXPECT_DOUBLE_EQ(m.m01, 4.0);
    EXPECT_DOUBLE_EQ(m.m10, 2.0);
    EXPECT_DOUBLE_EQ(m.m02, 7.0);
    EXPECT_DOUBLE_EQ(m.m20, 3.0);
    EXPECT_DOUBLE_EQ(m.m12, 8.0);
    EXPECT_DOUBLE_EQ(m.m21, 6.0);
}

TEST(Matrix3x3, InverseAndInversed_MultiplicativeIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 3, 3> a;
    a.set(
        4.0,
        1.0,
        2.0,
        0.0,
        3.0,
        -1.0,
        0.0,
        2.0,
        5.0);

    const atlas::math::Matrix<double, 3, 3> inv  = a.inversed();
    const atlas::math::Matrix<double, 3, 3> prod = a.mul(inv);

    EXPECT_TRUE(atlas::test::near(prod.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m22, 1.0, eps));
}

TEST(Matrix3x3, TryInverseRejectsSingular) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 3, 3> a(
        1.0,
        2.0,
        3.0,
        2.0,
        4.0,
        6.0,
        3.0,
        6.0,
        9.0);

    atlas::math::Matrix<double, 3, 3> out;
    const bool ok = a.try_inverse(out, eps);

    EXPECT_FALSE(ok);
}

TEST(Matrix3x3, IsInvertibleMatchesDeterminantMagnitude) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 3, 3> i;
    i.set_identity();
    EXPECT_TRUE(i.is_invertible(eps));

    const atlas::math::Matrix<double, 3, 3> z;
    EXPECT_FALSE(z.is_invertible(eps));
}

TEST(Matrix3x3, MatrixMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 3, 3> a(
        1.0,
        2.0,
        3.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        1.0);

    const atlas::math::Matrix<double, 3, 3> b(
        2.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        4.0);

    const atlas::math::Matrix<double, 3, 3> c = a * b;

    EXPECT_TRUE(atlas::test::near(c.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m01, 6.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m02, 12.0, eps));
}

TEST(Matrix3x3, MatrixVectorMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 3, 3> a(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0);

    const atlas::math::Vector<double, 3> v(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> y = a * v;

    EXPECT_TRUE(atlas::test::near(y[0], 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0, eps));
    EXPECT_TRUE(atlas::test::near(y[1], 4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0, eps));
    EXPECT_TRUE(atlas::test::near(y[2], 7.0 * 1.0 + 8.0 * 2.0 + 9.0 * 3.0, eps));
}

TEST(Matrix3x3, SolveAndSolved) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 3, 3> A(
        4.0,
        1.0,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        0.0,
        2.0);

    const atlas::math::Vector<double, 3> b(1.0, 2.0, 3.0);

    atlas::math::Vector<double, 3> x;
    const bool ok = A.solve(b, x, eps);
    EXPECT_TRUE(ok);

    const atlas::math::Vector<double, 3> y = A * x;
    EXPECT_TRUE(atlas::test::vec_near(y, b, eps));

    const atlas::math::Vector<double, 3> xs = A.solved(b);
    const atlas::math::Vector<double, 3> ys = A * xs;
    EXPECT_TRUE(atlas::test::vec_near(ys, b, eps));
}

TEST(Matrix3x3, FreeFunctions) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 3, 3> i = atlas::math::identity3x3<double>();
    EXPECT_TRUE(atlas::test::near(i.trace(), 3.0, eps));

    const atlas::math::Matrix<double, 3, 3> z = atlas::math::zero3x3<double>();
    EXPECT_TRUE(atlas::test::near(z.trace(), 0.0, eps));

    const atlas::math::Matrix<double, 3, 3> it = atlas::math::transpose(i);
    EXPECT_TRUE(it == i);

    const double det = atlas::math::determinant(i);
    EXPECT_TRUE(atlas::test::near(det, 1.0, eps));

    const atlas::math::Matrix<double, 3, 3> inv = atlas::math::inverse(i);
    EXPECT_TRUE(inv == i);
}