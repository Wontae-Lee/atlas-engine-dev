#include "../../utilities/tests_utils.h"
#include <atlas/math/math.h>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

TEST(Matrix4x4, DefaultConstructorIsZero) {
    const atlas::math::Matrix<double, 4, 4> m;

    for (std::size_t i = 0; i < atlas::math::Matrix<double, 4, 4>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix4x4, DiagonalConstructorIsScaledIdentity) {
    const atlas::math::Matrix<double, 4, 4> m(3.5);

    EXPECT_DOUBLE_EQ(m.m00, 3.5);
    EXPECT_DOUBLE_EQ(m.m11, 3.5);
    EXPECT_DOUBLE_EQ(m.m22, 3.5);
    EXPECT_DOUBLE_EQ(m.m33, 3.5);

    EXPECT_DOUBLE_EQ(m.m01, 0.0);
    EXPECT_DOUBLE_EQ(m.m02, 0.0);
    EXPECT_DOUBLE_EQ(m.m03, 0.0);

    EXPECT_DOUBLE_EQ(m.m10, 0.0);
    EXPECT_DOUBLE_EQ(m.m12, 0.0);
    EXPECT_DOUBLE_EQ(m.m13, 0.0);

    EXPECT_DOUBLE_EQ(m.m20, 0.0);
    EXPECT_DOUBLE_EQ(m.m21, 0.0);
    EXPECT_DOUBLE_EQ(m.m23, 0.0);

    EXPECT_DOUBLE_EQ(m.m30, 0.0);
    EXPECT_DOUBLE_EQ(m.m31, 0.0);
    EXPECT_DOUBLE_EQ(m.m32, 0.0);
}

TEST(Matrix4x4, ElementConstructorRowMajor) {
    const atlas::math::Matrix<double, 4, 4> m(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

    EXPECT_DOUBLE_EQ(m.m00, 1.0);
    EXPECT_DOUBLE_EQ(m.m01, 2.0);
    EXPECT_DOUBLE_EQ(m.m02, 3.0);
    EXPECT_DOUBLE_EQ(m.m03, 4.0);

    EXPECT_DOUBLE_EQ(m.m10, 5.0);
    EXPECT_DOUBLE_EQ(m.m11, 6.0);
    EXPECT_DOUBLE_EQ(m.m12, 7.0);
    EXPECT_DOUBLE_EQ(m.m13, 8.0);

    EXPECT_DOUBLE_EQ(m.m20, 9.0);
    EXPECT_DOUBLE_EQ(m.m21, 10.0);
    EXPECT_DOUBLE_EQ(m.m22, 11.0);
    EXPECT_DOUBLE_EQ(m.m23, 12.0);

    EXPECT_DOUBLE_EQ(m.m30, 13.0);
    EXPECT_DOUBLE_EQ(m.m31, 14.0);
    EXPECT_DOUBLE_EQ(m.m32, 15.0);
    EXPECT_DOUBLE_EQ(m.m33, 16.0);
}

TEST(Matrix4x4, InitializerListPadsWithZero) {
    const atlas::math::Matrix<double, 4, 4> m { 1.0, 2.0, 3.0 };

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[1], 2.0);
    EXPECT_DOUBLE_EQ(m[2], 3.0);

    for (std::size_t i = 3; i < atlas::math::Matrix<double, 4, 4>::size(); ++i) {
        EXPECT_DOUBLE_EQ(m[i], 0.0);
    }
}

TEST(Matrix4x4, RowsColsSize) {
    EXPECT_EQ(atlas::Matrix4x4<double>::rows(), static_cast<std::size_t>(4));
    EXPECT_EQ(atlas::Matrix4x4<double>::cols(), static_cast<std::size_t>(4));
    EXPECT_EQ(atlas::Matrix4x4<double>::size(), static_cast<std::size_t>(16));
}

TEST(Matrix4x4, DataPointerIsContiguousRowMajor) {
    atlas::math::Matrix<double, 4, 4> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

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
    EXPECT_DOUBLE_EQ(p[9], 10.0);
    EXPECT_DOUBLE_EQ(p[10], 11.0);
    EXPECT_DOUBLE_EQ(p[11], 12.0);

    EXPECT_DOUBLE_EQ(p[12], 13.0);
    EXPECT_DOUBLE_EQ(p[13], 14.0);
    EXPECT_DOUBLE_EQ(p[14], 15.0);
    EXPECT_DOUBLE_EQ(p[15], 16.0);

    double* pm = m.data();
    pm[10]     = -7.0;
    EXPECT_DOUBLE_EQ(m.m22, -7.0);
}

TEST(Matrix4x4, IndexAndAtAndCallOperator) {
    atlas::math::Matrix<double, 4, 4> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

    EXPECT_DOUBLE_EQ(m[0], 1.0);
    EXPECT_DOUBLE_EQ(m[5], 6.0);
    EXPECT_DOUBLE_EQ(m[15], 16.0);

    EXPECT_DOUBLE_EQ(m.at(2, 3), 12.0);
    EXPECT_DOUBLE_EQ(m(3, 2), 15.0);

    m(1, 2) = -9.0;
    EXPECT_DOUBLE_EQ(m.m12, -9.0);
}

TEST(Matrix4x4, SetZeroAndSetIdentity) {
    atlas::math::Matrix<double, 4, 4> z(3.0);
    z.set_zero();
    for (std::size_t i = 0; i < atlas::math::Matrix<double, 4, 4>::size(); ++i) {
        EXPECT_DOUBLE_EQ(z[i], 0.0);
    }

    atlas::math::Matrix<double, 4, 4> i;
    i.set_identity();

    EXPECT_DOUBLE_EQ(i.m00, 1.0);
    EXPECT_DOUBLE_EQ(i.m11, 1.0);
    EXPECT_DOUBLE_EQ(i.m22, 1.0);
    EXPECT_DOUBLE_EQ(i.m33, 1.0);

    EXPECT_DOUBLE_EQ(i.m01, 0.0);
    EXPECT_DOUBLE_EQ(i.m02, 0.0);
    EXPECT_DOUBLE_EQ(i.m03, 0.0);

    EXPECT_DOUBLE_EQ(i.m10, 0.0);
    EXPECT_DOUBLE_EQ(i.m12, 0.0);
    EXPECT_DOUBLE_EQ(i.m13, 0.0);

    EXPECT_DOUBLE_EQ(i.m20, 0.0);
    EXPECT_DOUBLE_EQ(i.m21, 0.0);
    EXPECT_DOUBLE_EQ(i.m23, 0.0);

    EXPECT_DOUBLE_EQ(i.m30, 0.0);
    EXPECT_DOUBLE_EQ(i.m31, 0.0);
    EXPECT_DOUBLE_EQ(i.m32, 0.0);
}

TEST(Matrix4x4, ScalarOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 4, 4> m;
    m.set_identity();

    m += 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m33, 3.0, eps));

    m -= 1.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m33, 2.0, eps));

    m *= 2.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m33, 4.0, eps));

    m /= 4.0;
    EXPECT_TRUE(atlas::test::near(m.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m22, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(m.m33, 1.0, eps));
}

TEST(Matrix4x4, MatrixOpsInPlace) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 4, 4> a;
    a.set_identity();

    atlas::math::Matrix<double, 4, 4> b;
    b.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

    a += b;
    EXPECT_TRUE(atlas::test::near(a.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m01, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m02, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m03, 4.0, eps));

    a -= b;
    EXPECT_TRUE(atlas::test::near(a.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m22, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(a.m33, 1.0, eps));
}

TEST(Matrix4x4, TraceAndDeterminantIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 4, 4> m;
    m.set_identity();

    EXPECT_TRUE(atlas::test::near(m.trace(), 4.0, eps));
    EXPECT_TRUE(atlas::test::near(m.determinant(), 1.0, eps));
}

TEST(Matrix4x4, TransposeAndTransposed) {
    atlas::math::Matrix<double, 4, 4> m;
    m.set(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

    atlas::math::Matrix<double, 4, 4> t = m.transposed();

    EXPECT_DOUBLE_EQ(t.m00, 1.0);
    EXPECT_DOUBLE_EQ(t.m01, 5.0);
    EXPECT_DOUBLE_EQ(t.m02, 9.0);
    EXPECT_DOUBLE_EQ(t.m03, 13.0);

    EXPECT_DOUBLE_EQ(t.m10, 2.0);
    EXPECT_DOUBLE_EQ(t.m11, 6.0);
    EXPECT_DOUBLE_EQ(t.m12, 10.0);
    EXPECT_DOUBLE_EQ(t.m13, 14.0);

    EXPECT_DOUBLE_EQ(t.m20, 3.0);
    EXPECT_DOUBLE_EQ(t.m21, 7.0);
    EXPECT_DOUBLE_EQ(t.m22, 11.0);
    EXPECT_DOUBLE_EQ(t.m23, 15.0);

    EXPECT_DOUBLE_EQ(t.m30, 4.0);
    EXPECT_DOUBLE_EQ(t.m31, 8.0);
    EXPECT_DOUBLE_EQ(t.m32, 12.0);
    EXPECT_DOUBLE_EQ(t.m33, 16.0);

    m.transpose();
    EXPECT_DOUBLE_EQ(m.m01, 5.0);
    EXPECT_DOUBLE_EQ(m.m10, 2.0);
    EXPECT_DOUBLE_EQ(m.m23, 15.0);
    EXPECT_DOUBLE_EQ(m.m32, 12.0);
}

TEST(Matrix4x4, InverseAndInversed_MultiplicativeIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 4, 4> a;
    a.set(
        4.0,
        1.0,
        2.0,
        0.0,
        0.0,
        3.0,
        -1.0,
        0.0,
        0.0,
        2.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0);

    const atlas::math::Matrix<double, 4, 4> inv = a.inversed();

    const atlas::math::Matrix<double, 4, 4> prod = a.mul(inv);

    EXPECT_TRUE(atlas::test::near(prod.m00, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m11, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m22, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(prod.m33, 1.0, eps));
}

TEST(Matrix4x4, TryInverseRejectsSingular) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 4, 4> a(
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        4.0,
        6.0,
        8.0,
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0);

    atlas::math::Matrix<double, 4, 4> out;
    const bool ok = a.try_inverse(out, eps);

    EXPECT_FALSE(ok);
}

TEST(Matrix4x4, IsInvertibleMatchesDeterminantMagnitude) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Matrix<double, 4, 4> i;
    i.set_identity();
    EXPECT_TRUE(i.is_invertible(eps));

    const atlas::math::Matrix<double, 4, 4> z;
    EXPECT_FALSE(z.is_invertible(eps));
}

TEST(Matrix4x4, MatrixMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 4, 4> a(
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        0.0,
        1.0);

    const atlas::math::Matrix<double, 4, 4> b(
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.0);

    const atlas::math::Matrix<double, 4, 4> c = a * b;

    EXPECT_TRUE(atlas::test::near(c.m00, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m01, 6.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m02, 12.0, eps));
    EXPECT_TRUE(atlas::test::near(c.m03, 20.0, eps));
}

TEST(Matrix4x4, MatrixVectorMultiply) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 4, 4> a(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0);

    const atlas::math::Vector<double, 4> v(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Vector<double, 4> y = a * v;

    EXPECT_TRUE(atlas::test::near(y[0], 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0, eps));
    EXPECT_TRUE(atlas::test::near(y[1], 5.0 * 1.0 + 6.0 * 2.0 + 7.0 * 3.0 + 8.0 * 4.0, eps));
    EXPECT_TRUE(atlas::test::near(y[2], 9.0 * 1.0 + 10.0 * 2.0 + 11.0 * 3.0 + 12.0 * 4.0, eps));
    EXPECT_TRUE(atlas::test::near(y[3], 13.0 * 1.0 + 14.0 * 2.0 + 15.0 * 3.0 + 16.0 * 4.0, eps));
}

TEST(Matrix4x4, SolveAndSolved) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 4, 4> A(
        4.0,
        1.0,
        0.0,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.0);

    const atlas::math::Vector<double, 4> b(1.0, 2.0, 3.0, 4.0);

    atlas::math::Vector<double, 4> x;
    const bool ok = A.solve(b, x, eps);
    EXPECT_TRUE(ok);

    const atlas::math::Vector<double, 4> y = A * x;
    EXPECT_TRUE(atlas::test::vec_near(y, b, eps));

    const atlas::math::Vector<double, 4> xs = A.solved(b);
    const atlas::math::Vector<double, 4> ys = A * xs;
    EXPECT_TRUE(atlas::test::vec_near(ys, b, eps));
}

TEST(Matrix4x4, FreeFunctions) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Matrix<double, 4, 4> i = atlas::math::identity4x4<double>();
    EXPECT_TRUE(atlas::test::near(i.trace(), 4.0, eps));

    const atlas::math::Matrix<double, 4, 4> z = atlas::math::zero4x4<double>();
    EXPECT_TRUE(atlas::test::near(z.trace(), 0.0, eps));

    const atlas::math::Matrix<double, 4, 4> it = atlas::math::transpose(i);
    EXPECT_TRUE(it == i);

    const double det = atlas::math::determinant(i);
    EXPECT_TRUE(atlas::test::near(det, 1.0, eps));

    const atlas::math::Matrix<double, 4, 4> inv = atlas::math::inverse(i);
    EXPECT_TRUE(inv == i);
}