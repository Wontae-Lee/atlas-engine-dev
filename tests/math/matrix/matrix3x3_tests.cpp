#include <atlas/math/matrix/matrix3x3.h>

#include <cstddef>
#include <gtest/gtest.h>

namespace {

constexpr float k_eps = 1e-5f;

void
expect_matrix_near(const atlas::Matrix3x3& a, const atlas::Matrix3x3& b, const float tolerance) {
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(a[i], b[i], tolerance) << "element " << i;
    }
}

}

TEST(Matrix3x3, DefaultConstructibleToZero) {
    const atlas::Matrix3x3 m {};
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(m[i], 0.0f);
    }
}

TEST(Matrix3x3, DiagonalScalarConstructor) {
    const atlas::Matrix3x3 m(2.0f);
    EXPECT_FLOAT_EQ(m.m00, 2.0f);
    EXPECT_FLOAT_EQ(m.m11, 2.0f);
    EXPECT_FLOAT_EQ(m.m22, 2.0f);
    EXPECT_FLOAT_EQ(m.m01, 0.0f);
    EXPECT_FLOAT_EQ(m.m10, 0.0f);
}

TEST(Matrix3x3, ComponentConstructorAndAccess) {
    const atlas::Matrix3x3 m(1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f,
                             7.0f, 8.0f, 9.0f);

    EXPECT_FLOAT_EQ(m.m00, 1.0f);
    EXPECT_FLOAT_EQ(m.m12, 6.0f);
    EXPECT_FLOAT_EQ(m.m21, 8.0f);

    EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(m.at(2, 1), 8.0f);
    EXPECT_FLOAT_EQ(m[4], 5.0f);
}

TEST(Matrix3x3, SetIdentityAndZero) {
    atlas::Matrix3x3 m(1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f);

    m.set_identity();
    expect_matrix_near(m, atlas::identity3x3(), 0.0f);

    m.set_zero();
    expect_matrix_near(m, atlas::zero3x3(), 0.0f);
}

TEST(Matrix3x3, ScalarOps) {
    atlas::Matrix3x3 m(1.0f);

    m.add(1.0f);
    EXPECT_FLOAT_EQ(m.m00, 2.0f);
    EXPECT_FLOAT_EQ(m.m01, 1.0f);

    m.mul(2.0f);
    EXPECT_FLOAT_EQ(m.m00, 4.0f);

    m.div(4.0f);
    EXPECT_FLOAT_EQ(m.m00, 1.0f);

    m.sub(0.5f);
    EXPECT_FLOAT_EQ(m.m00, 0.5f);
}

TEST(Matrix3x3, MatrixAddSub) {
    const atlas::Matrix3x3 a(1.0f);
    const atlas::Matrix3x3 b(2.0f);

    const atlas::Matrix3x3 s = a + b;
    EXPECT_FLOAT_EQ(s.m00, 3.0f);
    EXPECT_FLOAT_EQ(s.m01, 0.0f);

    const atlas::Matrix3x3 d = b - a;
    EXPECT_FLOAT_EQ(d.m11, 1.0f);

    atlas::Matrix3x3 c = a;
    c += b;
    c -= a;
    expect_matrix_near(c, b, 0.0f);
}

TEST(Matrix3x3, TraceAndDeterminant) {
    const atlas::Matrix3x3 m(2.0f, 0.0f, 0.0f,
                             0.0f, 3.0f, 0.0f,
                             0.0f, 0.0f, 4.0f);

    EXPECT_FLOAT_EQ(m.trace(), 9.0f);
    EXPECT_FLOAT_EQ(m.determinant(), 24.0f);
    EXPECT_FLOAT_EQ(atlas::determinant(m), 24.0f);
}

TEST(Matrix3x3, Transpose) {
    const atlas::Matrix3x3 m(1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f,
                             7.0f, 8.0f, 9.0f);

    const atlas::Matrix3x3 t = m.transposed();
    EXPECT_FLOAT_EQ(t.m01, 4.0f);
    EXPECT_FLOAT_EQ(t.m20, 3.0f);

    atlas::Matrix3x3 s = m;
    s.transpose();
    expect_matrix_near(s, t, 0.0f);
    expect_matrix_near(atlas::transpose(m), t, 0.0f);
}

TEST(Matrix3x3, InverseRoundTrip) {
    const atlas::Matrix3x3 m(2.0f, 0.0f, 1.0f,
                             0.0f, 3.0f, 0.0f,
                             1.0f, 0.0f, 1.0f);

    ASSERT_TRUE(m.is_invertible());

    const atlas::Matrix3x3 inv = m.inversed();
    expect_matrix_near(m.mul(inv), atlas::identity3x3(), k_eps);
    expect_matrix_near(atlas::inverse(m).mul(m), atlas::identity3x3(), k_eps);
}

TEST(Matrix3x3, TryInverseRejectsSingular) {
    const atlas::Matrix3x3 singular(1.0f, 2.0f, 3.0f,
                                    2.0f, 4.0f, 6.0f,
                                    0.0f, 0.0f, 1.0f);

    atlas::Matrix3x3 out;
    EXPECT_FALSE(singular.try_inverse(out));
    EXPECT_FALSE(singular.is_invertible());
}

TEST(Matrix3x3, MatrixMultiply) {
    const atlas::Matrix3x3 a(1.0f, 2.0f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f);
    const atlas::Matrix3x3 b(1.0f, 0.0f, 0.0f,
                             3.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f);

    const atlas::Matrix3x3 ab = a * b;
    EXPECT_FLOAT_EQ(ab.m00, 7.0f);
    EXPECT_FLOAT_EQ(ab.m01, 2.0f);

    expect_matrix_near(a * atlas::identity3x3(), a, 0.0f);
}

TEST(Matrix3x3, MatrixVectorMultiply) {
    const atlas::Matrix3x3 m(0.0f, -1.0f, 0.0f,
                             1.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 1.0f);
    const atlas::Vector3 v(1.0f, 0.0f, 0.5f);

    const atlas::Vector3 r = m * v;
    EXPECT_NEAR(r.x, 0.0f, k_eps);
    EXPECT_NEAR(r.y, 1.0f, k_eps);
    EXPECT_NEAR(r.z, 0.5f, k_eps);

    const atlas::Vector3 r2 = m.mul(v);
    EXPECT_NEAR(r2.y, 1.0f, k_eps);
}

TEST(Matrix3x3, RotateFreeFunctions) {
    const atlas::Matrix3x3 rot(0.0f, -1.0f, 0.0f,
                               1.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f);
    const atlas::Vector3 input(1.0f, 0.0f, 0.0f);
    const atlas::Vector3 offset(10.0f, 10.0f, 10.0f);

    atlas::Vector3 out;
    atlas::rotate(rot, input, out);
    EXPECT_NEAR(out.y, 1.0f, k_eps);

    atlas::rotate_translate(rot, input, offset, out);
    EXPECT_NEAR(out.x, 10.0f, k_eps);
    EXPECT_NEAR(out.y, 11.0f, k_eps);

    atlas::rotate_subtract(rot, input + offset, offset, out);
    EXPECT_NEAR(out.y, 1.0f, k_eps);
}

TEST(Matrix3x3, SolveLinearSystem) {
    const atlas::Matrix3x3 A(2.0f, 0.0f, 0.0f,
                             0.0f, 4.0f, 0.0f,
                             0.0f, 0.0f, 8.0f);
    const atlas::Vector3 b(2.0f, 4.0f, 8.0f);

    atlas::Vector3 x;
    ASSERT_TRUE(A.solve(b, x));
    EXPECT_NEAR(x.x, 1.0f, k_eps);
    EXPECT_NEAR(x.y, 1.0f, k_eps);
    EXPECT_NEAR(x.z, 1.0f, k_eps);

    const atlas::Vector3 x2 = atlas::solve(A, b);
    EXPECT_NEAR(x2.z, 1.0f, k_eps);

    const atlas::Matrix3x3 singular {};
    EXPECT_FALSE(atlas::solve(singular, b, x));
}
