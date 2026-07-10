#include <atlas/math/matrix/float3x3.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Float3;
using atlas::Float3x3;

// A determinant-1 non-symmetric matrix with an exact integer inverse, handy for
// round-trip checks that keep floating-point error small.
Float3x3
invertible() {
    return Float3x3(1.0f, 2.0f, 3.0f,
                    0.0f, 1.0f, 4.0f,
                    5.0f, 6.0f, 0.0f);
}

// Assert two matrices agree within an absolute tolerance on every element.
void
expect_matrix_near(const Float3x3& a, const Float3x3& b, const float tol) {
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "element " << i;
    }
}

}

TEST(Float3x3, DefaultConstructsToZero) {
    const Float3x3 m;
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(m[i], 0.0f);
    }
    EXPECT_TRUE(m == atlas::zero3x3());
}

TEST(Float3x3, ScalarConstructorBuildsAScaledIdentity) {
    const Float3x3 m(2.0f);
    EXPECT_FLOAT_EQ(m.m00, 2.0f);
    EXPECT_FLOAT_EQ(m.m11, 2.0f);
    EXPECT_FLOAT_EQ(m.m22, 2.0f);
    EXPECT_FLOAT_EQ(m.m01, 0.0f);
    EXPECT_TRUE(atlas::identity3x3() == Float3x3(1.0f));
}

TEST(Float3x3, ElementAccessorsAliasTheSameStorage) {
    const Float3x3 m = invertible();
    EXPECT_FLOAT_EQ(m.at(1, 2), m.m12);
    EXPECT_FLOAT_EQ(m(2, 0), m.m20);
    EXPECT_FLOAT_EQ(m[1 * 3 + 2], m.m12);
    EXPECT_EQ(m.data(), &m.m00);
}

TEST(Float3x3, EqualityIsExactElementWise) {
    EXPECT_TRUE(invertible() == invertible());
    Float3x3 other = invertible();
    other.m11 += 1.0f;
    EXPECT_TRUE(invertible() != other);
}

TEST(Float3x3, MultiplyingByIdentityIsANoOp) {
    const Float3x3 a = invertible();
    EXPECT_TRUE(a * atlas::identity3x3() == a);
    EXPECT_TRUE(atlas::identity3x3() * a == a);
}

TEST(Float3x3, MatrixVectorProductWithIdentityReturnsTheVector) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(atlas::identity3x3() * v == v);
    EXPECT_TRUE(atlas::identity3x3().mul(v) == v);
}

TEST(Float3x3, MatrixVectorProductAppliesEachRowAsADot) {
    const Float3x3 m(1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f);
    const Float3 r = m * Float3(1.0f, 0.0f, -1.0f);
    EXPECT_FLOAT_EQ(r.x, -2.0f);
    EXPECT_FLOAT_EQ(r.y, -2.0f);
    EXPECT_FLOAT_EQ(r.z, -2.0f);
}

TEST(Float3x3, TraceSumsTheDiagonal) {
    EXPECT_FLOAT_EQ(invertible().trace(), 2.0f);
    EXPECT_FLOAT_EQ(atlas::identity3x3().trace(), 3.0f);
}

TEST(Float3x3, TransposeIsItsOwnInverse) {
    const Float3x3 a = invertible();
    EXPECT_TRUE(a.transposed().transposed() == a);
    // (r, c) of the transpose equals (c, r) of the original.
    EXPECT_FLOAT_EQ(a.transposed().at(0, 2), a.at(2, 0));
    EXPECT_TRUE(atlas::transpose(a) == a.transposed());
}

TEST(Float3x3, TransposeInPlaceMatchesTheCopyingForm) {
    Float3x3 a = invertible();
    const Float3x3 expected = a.transposed();
    a.transpose();
    EXPECT_TRUE(a == expected);
}

TEST(Float3x3, DeterminantOfIdentityIsOne) {
    EXPECT_FLOAT_EQ(atlas::identity3x3().determinant(), 1.0f);
}

TEST(Float3x3, DeterminantMatchesTheCofactorExpansion) {
    EXPECT_FLOAT_EQ(invertible().determinant(), 1.0f);
    EXPECT_FLOAT_EQ(atlas::determinant(invertible()), 1.0f);
}

TEST(Float3x3, DeterminantIsZeroForARankDeficientMatrix) {
    const Float3x3 singular(1.0f, 2.0f, 3.0f,
                            2.0f, 4.0f, 6.0f,
                            0.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(singular.determinant(), 0.0f);
    EXPECT_FALSE(singular.is_invertible());
}

TEST(Float3x3, InverseTimesSelfIsTheIdentity) {
    const Float3x3 a = invertible();
    expect_matrix_near(a * a.inversed(), atlas::identity3x3(), 1.0e-4f);
    expect_matrix_near(a.inversed() * a, atlas::identity3x3(), 1.0e-4f);
}

TEST(Float3x3, FreeInverseMatchesTheMemberForm) {
    const Float3x3 a = invertible();
    expect_matrix_near(atlas::inverse(a), a.inversed(), 1.0e-5f);
}

TEST(Float3x3, TryInverseFailsOnASingularMatrix) {
    const Float3x3 singular(1.0f, 2.0f, 3.0f,
                            2.0f, 4.0f, 6.0f,
                            0.0f, 0.0f, 0.0f);
    Float3x3 out = invertible();
    const Float3x3 untouched = out;
    EXPECT_FALSE(singular.try_inverse(out));
    // A failed inversion must leave the output untouched.
    EXPECT_TRUE(out == untouched);
}

TEST(Float3x3, TryInverseSucceedsOnANonSingularMatrix) {
    const Float3x3 a = invertible();
    Float3x3 out;
    EXPECT_TRUE(a.try_inverse(out));
    expect_matrix_near(a * out, atlas::identity3x3(), 1.0e-4f);
}

TEST(Float3x3, SolveRecoversTheRightHandSide) {
    const Float3x3 a = invertible();
    const Float3 b(1.0f, 2.0f, 3.0f);
    Float3 x;
    EXPECT_TRUE(a.solve(b, x));
    const Float3 recovered = a * x;
    EXPECT_NEAR(recovered.x, b.x, 1.0e-4f);
    EXPECT_NEAR(recovered.y, b.y, 1.0e-4f);
    EXPECT_NEAR(recovered.z, b.z, 1.0e-4f);
}

TEST(Float3x3, SolveFailsForASingularSystem) {
    const Float3x3 singular(1.0f, 2.0f, 3.0f,
                            2.0f, 4.0f, 6.0f,
                            0.0f, 0.0f, 0.0f);
    Float3 x(9.0f, 9.0f, 9.0f);
    EXPECT_FALSE(atlas::solve(singular, Float3(1.0f, 2.0f, 3.0f), x));
    // The unchanged output confirms the singular branch wrote nothing.
    EXPECT_TRUE(x == Float3(9.0f, 9.0f, 9.0f));
}

TEST(Float3x3, ScalarAndElementWiseOperatorsCombine) {
    const Float3x3 a = atlas::identity3x3();
    const Float3x3 sum = a + a;
    EXPECT_FLOAT_EQ(sum.m00, 2.0f);
    EXPECT_TRUE((2.0f * a) == (a * 2.0f));
    EXPECT_TRUE(((a * 4.0f) / 4.0f) == a);
    EXPECT_TRUE((sum - a) == a);
}

TEST(Float3x3, RotateFreeFunctionIsSafeWhenInputAliasesOutput) {
    const Float3x3 m(1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f);
    const Float3 v(1.0f, 0.0f, -1.0f);
    Float3 out;
    atlas::rotate(m, v, out);

    Float3 aliased = v;
    atlas::rotate(m, aliased, aliased);
    EXPECT_FLOAT_EQ(aliased.x, out.x);
    EXPECT_FLOAT_EQ(aliased.y, out.y);
    EXPECT_FLOAT_EQ(aliased.z, out.z);
}

TEST(Float3x3, RotateTranslateAddsTheOffsetAfterRotation) {
    const Float3x3 id = atlas::identity3x3();
    const Float3 input(1.0f, 2.0f, 3.0f);
    const Float3 offset(10.0f, 20.0f, 30.0f);
    Float3 out;
    atlas::rotate_translate(id, input, offset, out);
    EXPECT_TRUE(out == input + offset);
}

TEST(Float3x3, RotateSubtractRemovesTheOffsetBeforeRotation) {
    const Float3x3 id = atlas::identity3x3();
    const Float3 input(10.0f, 20.0f, 30.0f);
    const Float3 offset(1.0f, 2.0f, 3.0f);
    Float3 out;
    atlas::rotate_subtract(id, input, offset, out);
    EXPECT_TRUE(out == input - offset);
}
