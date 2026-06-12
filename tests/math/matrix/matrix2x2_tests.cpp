#include "../../utilities/test_utils.h"
#include <cmath>
#include <cstddef>
#include <testkit/testkit.h>

TEST(Matrix2x2, DefaultConstructorIsZero) {
    const atlas::Matrix<double, 2, 2> m;

    for (std::size_t i = 0; i < atlas::Matrix<double, 2, 2>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix2x2, DiagonalConstructorIsScaledIdentity) {
    const atlas::Matrix<double, 2, 2> m(3.5);

    EXPECT_DOUBLE_EQ(m.m00, 3.5);
    EXPECT_DOUBLE_EQ(m.m11, 3.5);

    EXPECT_DOUBLE_EQ(m.m01, 0.0);
    EXPECT_DOUBLE_EQ(m.m10, 0.0);
}

TEST(Matrix2x2, ElementConstructorRowMajor) {
    const atlas::Matrix<double, 2, 2> m(
        1.0,
        2.0,
        3.0,
        4.0);

    EXPECT_DOUBLE_EQ(m.m00, 1.0);
    EXPECT_DOUBLE_EQ(m.m01, 2.0);
    EXPECT_DOUBLE_EQ(m.m10, 3.0);
    EXPECT_DOUBLE_EQ(m.m11, 4.0);
}

TEST(Matrix2x2, InitializerListPadsWithZero) {
    const atlas::Matrix<double, 2, 2> m { 1.0, 2.0 };

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[1], 2.0);

    for (std::size_t i = 2; i < atlas::Matrix<double, 2, 2>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix2x2, RowsColsSize) {
    EXPECT_EQ(atlas::Matrix2x2<double>::rows(), static_cast<std::size_t>(2));
    EXPECT_EQ(atlas::Matrix2x2<double>::cols(), static_cast<std::size_t>(2));
    EXPECT_EQ(atlas::Matrix2x2<double>::size(), static_cast<std::size_t>(4));
}

TEST(Matrix2x2, DataPointerIsContiguousRowMajor) {
    atlas::Matrix<double, 2, 2> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0);

    const double* p = m.data();
    ASSERT_NE(p, nullptr);

    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    double* pm = m.data();
    pm[3]      = -7.0;
    EXPECT_DOUBLE_EQ(m.m11, -7.0);
}

TEST(Matrix2x2, IndexAndAtAndCallOperator) {
    atlas::Matrix<double, 2, 2> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0);

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[3], 4.0);

    EXPECT_DOUBLE_EQ(m.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);

    m(1, 0) = -9.0;
    EXPECT_DOUBLE_EQ(m.m10, -9.0);
}

TEST(Matrix2x2, SetZeroAndSetIdentity) {
    atlas::Matrix<double, 2, 2> z(3.0);
    z.set_zero();
    for (std::size_t i = 0; i < atlas::Matrix<double, 2, 2>::size(); ++i) {
        EXPECT_DOUBLE_EQ(z[i], 0.0);
    }

    atlas::Matrix<double, 2, 2> i;
    i.set_identity();

    EXPECT_DOUBLE_EQ(i.m00, 1.0);
    EXPECT_DOUBLE_EQ(i.m11, 1.0);

    EXPECT_DOUBLE_EQ(i.m01, 0.0);
    EXPECT_DOUBLE_EQ(i.m10, 0.0);
}

TEST(Matrix2x2, ScalarOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Matrix<double, 2, 2> m;
    m.set_identity();

    m += 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m01, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m10, 2.0, eps));

    m -= 1.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m01, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m10, 1.0, eps));

    m *= 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m01, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m10, 2.0, eps));

    m /= 4.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m01, 0.5, eps));
    EXPECT_TRUE(atlas::test::near(m.m10, 0.5, eps));
}

TEST(Matrix2x2, MatrixOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Matrix<double, 2, 2> a;
    a.set_identity();

    atlas::Matrix<double, 2, 2> b;
    b.set(
        1.0,
        2.0,
        3.0,
        4.0);

    a += b;
    EXPECT_TRUE(atlas::test::near(a.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m01, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m10, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m11, 5.0, eps));

    a -= b;
    EXPECT_TRUE(atlas::test::near(a.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m01, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m10, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m11, 1.0, eps));
}

TEST(Matrix2x2, TraceAndDeterminantIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Matrix<double, 2, 2> m;
    m.set_identity();

    EXPECT_TRUE(atlas::test::near(m.trace(), 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.determinant(), 1.0, eps));
}

TEST(Matrix2x2, TransposeAndTransposed) {
    atlas::Matrix<double, 2, 2> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0);

    const atlas::Matrix<double, 2, 2> t = m.transposed();

    EXPECT_DOUBLE_EQ(t.m00, 1.0);
    EXPECT_DOUBLE_EQ(t.m01, 3.0);
    EXPECT_DOUBLE_EQ(t.m10, 2.0);
    EXPECT_DOUBLE_EQ(t.m11, 4.0);

    m.transpose();
    EXPECT_DOUBLE_EQ(m.m00, 1.0);
    EXPECT_DOUBLE_EQ(m.m01, 3.0);
    EXPECT_DOUBLE_EQ(m.m10, 2.0);
    EXPECT_DOUBLE_EQ(m.m11, 4.0);
}

TEST(Matrix2x2, InverseAndInversed_MultiplicativeIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Matrix<double, 2, 2> a;
    a.set(
        4.0,
        1.0,
        2.0,
        3.0);

    const atlas::Matrix<double, 2, 2> inv  = a.inversed();
    const atlas::Matrix<double, 2, 2> prod = a.mul(inv);

    EXPECT_TRUE(atlas::test::near(prod.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m11, 1.0, eps));
}

TEST(Matrix2x2, TryInverseRejectsSingular) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Matrix<double, 2, 2> a(
        1.0,
        2.0,
        2.0,
        4.0);

    atlas::Matrix<double, 2, 2> out;
    const bool ok = a.try_inverse(out, eps);

    EXPECT_FALSE(ok);
}

TEST(Matrix2x2, IsInvertibleMatchesDeterminantMagnitude) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Matrix<double, 2, 2> i;
    i.set_identity();
    EXPECT_TRUE(i.is_invertible(eps));

    const atlas::Matrix<double, 2, 2> z;
    EXPECT_FALSE(z.is_invertible(eps));
}

TEST(Matrix2x2, MatrixMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Matrix<double, 2, 2> a(
        1.0,
        2.0,
        3.0,
        4.0);

    const atlas::Matrix<double, 2, 2> b(
        2.0,
        0.0,
        0.0,
        3.0);

    const atlas::Matrix<double, 2, 2> c = a * b;

    EXPECT_TRUE(atlas::test::near(c.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m01, 6.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m10, 6.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m11, 12.0, eps));
}

TEST(Matrix2x2, MatrixVectorMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Matrix<double, 2, 2> a(
        1.0,
        2.0,
        3.0,
        4.0);

    const atlas::Vector<double, 2> v(1.0, 2.0);
    const atlas::Vector<double, 2> y = a * v;

    EXPECT_TRUE(atlas::test::near(y[0], 1.0 * 1.0 + 2.0 * 2.0, eps));
    EXPECT_TRUE(atlas::test::near(y[1], 3.0 * 1.0 + 4.0 * 2.0, eps));
}

TEST(Matrix2x2, SolveAndSolved) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Matrix<double, 2, 2> A(
        4.0,
        1.0,
        2.0,
        3.0);

    const atlas::Vector<double, 2> b(1.0, 2.0);

    atlas::Vector<double, 2> x;
    const bool ok = A.solve(b, x, eps);
    EXPECT_TRUE(ok);

    const atlas::Vector<double, 2> y = A * x;
    EXPECT_TRUE(atlas::test::vec_near(y, b, eps));

    const atlas::Vector<double, 2> xs = A.solved(b);
    const atlas::Vector<double, 2> ys = A * xs;
    EXPECT_TRUE(atlas::test::vec_near(ys, b, eps));
}

TEST(Matrix2x2, FreeFunctions) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Matrix<double, 2, 2> i = atlas::identity2x2<double>();
    EXPECT_TRUE(atlas::test::near(i.trace(), 2.0, eps));

    const atlas::Matrix<double, 2, 2> z = atlas::zero2x2<double>();
    EXPECT_TRUE(atlas::test::near(z.trace(), 0.0, eps));

    const atlas::Matrix<double, 2, 2> it = atlas::transpose(i);
    EXPECT_TRUE(it == i);

    const double det = atlas::determinant(i);
    EXPECT_TRUE(atlas::test::near(det, 1.0, eps));

    const atlas::Matrix<double, 2, 2> inv = atlas::inverse(i);
    EXPECT_TRUE(inv == i);
}