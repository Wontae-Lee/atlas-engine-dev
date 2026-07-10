#include <atlas/math/vector/float3.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Bool3;
using atlas::Float3;

}

TEST(Float3, DefaultConstructsToZero) {
    const Float3 v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST(Float3, ScalarConstructorFillsEveryComponent) {
    const Float3 v(2.5f);
    EXPECT_FLOAT_EQ(v.x, 2.5f);
    EXPECT_FLOAT_EQ(v.y, 2.5f);
    EXPECT_FLOAT_EQ(v.z, 2.5f);
}

TEST(Float3, IndexingMatchesNamedComponents) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
    EXPECT_EQ(v.data(), &v.x);
}

TEST(Float3, EqualityIsExactComponentWise) {
    EXPECT_TRUE(Float3(1.0f, 2.0f, 3.0f) == Float3(1.0f, 2.0f, 3.0f));
    EXPECT_TRUE(Float3(1.0f, 2.0f, 3.0f) != Float3(1.0f, 2.0f, 3.5f));
}

TEST(Float3, AdditionAndSubtractionAreInverse) {
    const Float3 a(1.0f, -2.0f, 3.0f);
    const Float3 b(4.0f, 5.0f, -6.0f);
    EXPECT_TRUE((a + b) - b == a);
}

TEST(Float3, UnaryNegationFlipsEverySign) {
    const Float3 v = -Float3(1.0f, -2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v.x, -1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, -3.0f);
    EXPECT_TRUE(+v == v);
}

TEST(Float3, ScalarMultiplicationCommutes) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(2.0f * v == v * 2.0f);
    EXPECT_TRUE(v * 2.0f == Float3(2.0f, 4.0f, 6.0f));
}

TEST(Float3, DivisionUndoesMultiplication) {
    const Float3 v(2.0f, 4.0f, 8.0f);
    EXPECT_TRUE((v * 4.0f) / 4.0f == v);
}

TEST(Float3, HadamardProductIsComponentWise) {
    const Float3 r = Float3(1.0f, 2.0f, 3.0f) * Float3(4.0f, 5.0f, 6.0f);
    EXPECT_TRUE(r == Float3(4.0f, 10.0f, 18.0f));
}

TEST(Float3, CompoundAssignmentMatchesBinaryForm) {
    Float3 v(1.0f, 2.0f, 3.0f);
    v += Float3(1.0f, 1.0f, 1.0f);
    v *= 2.0f;
    EXPECT_TRUE(v == Float3(4.0f, 6.0f, 8.0f));
}

TEST(Float3, DotEqualsLengthSquaredWithItself) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v.dot(v), v.length_squared());
    EXPECT_FLOAT_EQ(atlas::dot(v, v), 14.0f);
}

TEST(Float3, DotIsSymmetric) {
    const Float3 a(1.0f, 2.0f, 3.0f);
    const Float3 b(-4.0f, 5.0f, 6.0f);
    EXPECT_FLOAT_EQ(a.dot(b), b.dot(a));
    EXPECT_FLOAT_EQ(atlas::dot(a, b), a.dot(b));
}

TEST(Float3, CrossOfBasisVectorsFollowsRightHandRule) {
    const Float3 x(1.0f, 0.0f, 0.0f);
    const Float3 y(0.0f, 1.0f, 0.0f);
    EXPECT_TRUE(x.cross(y) == Float3(0.0f, 0.0f, 1.0f));
    EXPECT_TRUE(atlas::cross(x, y) == Float3(0.0f, 0.0f, 1.0f));
}

TEST(Float3, CrossIsAntiCommutativeAndOrthogonal) {
    const Float3 a(1.0f, 2.0f, 3.0f);
    const Float3 b(4.0f, 5.0f, 6.0f);
    const Float3 c = a.cross(b);
    EXPECT_TRUE(b.cross(a) == -c);
    EXPECT_NEAR(c.dot(a), 0.0f, 1.0e-5f);
    EXPECT_NEAR(c.dot(b), 0.0f, 1.0e-5f);
}

TEST(Float3, LengthIsThePythagoreanNorm) {
    const Float3 v(3.0f, 4.0f, 0.0f);
    EXPECT_FLOAT_EQ(v.length_squared(), 25.0f);
    EXPECT_FLOAT_EQ(v.length(), 5.0f);
}

TEST(Float3, NormalizedHasUnitLength) {
    const Float3 u = Float3(3.0f, 4.0f, 0.0f).normalized();
    EXPECT_NEAR(u.length(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(u.x, 0.6f, 1.0e-6f);
    EXPECT_NEAR(u.y, 0.8f, 1.0e-6f);
}

TEST(Float3, NormalizeLeavesTheZeroVectorUnchanged) {
    Float3 v;
    v.normalize();
    EXPECT_TRUE(v == Float3(0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(Float3().normalized() == Float3(0.0f, 0.0f, 0.0f));
}

TEST(Float3, ReflectedAboutAPlaneFlipsTheNormalComponent) {
    const Float3 v(1.0f, -1.0f, 0.0f);
    const Float3 n(0.0f, 1.0f, 0.0f);
    EXPECT_TRUE(v.reflected(n) == Float3(1.0f, 1.0f, 0.0f));
    EXPECT_TRUE(atlas::reflected(v, n) == Float3(1.0f, 1.0f, 0.0f));
}

TEST(Float3, ProjectedRemovesTheNormalComponent) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 n(0.0f, 0.0f, 1.0f);
    EXPECT_TRUE(v.projected(n) == Float3(1.0f, 2.0f, 0.0f));
    EXPECT_TRUE(atlas::projected(v, n) == Float3(1.0f, 2.0f, 0.0f));
}

TEST(Float3, RejectMatchesProjectionOntoThePlane) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 n(0.0f, 0.0f, 1.0f);
    const Float3 r = atlas::reject(v, n);
    EXPECT_NEAR(r.x, 1.0f, 1.0e-6f);
    EXPECT_NEAR(r.y, 2.0f, 1.0e-6f);
    EXPECT_NEAR(r.z, 0.0f, 1.0e-6f);
}

TEST(Float3, MinAndMaxSelectTheExtremeComponent) {
    const Float3 v(-2.0f, 5.0f, 1.0f);
    EXPECT_FLOAT_EQ(v.min(), -2.0f);
    EXPECT_FLOAT_EQ(v.max(), 5.0f);
}

TEST(Float3, MajorAndMinorAxisReportSignedExtremes) {
    const Float3 v(-2.0f, 5.0f, 1.0f);
    EXPECT_EQ(v.major_axis(), 1u);
    EXPECT_EQ(v.minor_axis(), 0u);
}

TEST(Float3, ComponentWiseMinMaxTakePerAxisExtremes) {
    const Float3 a(1.0f, 5.0f, 3.0f);
    const Float3 b(4.0f, 2.0f, 6.0f);
    EXPECT_TRUE(atlas::min(a, b) == Float3(1.0f, 2.0f, 3.0f));
    EXPECT_TRUE(atlas::max(a, b) == Float3(4.0f, 5.0f, 6.0f));
    EXPECT_TRUE(atlas::cmin(a, b) == atlas::min(a, b));
    EXPECT_TRUE(atlas::cmax(a, b) == atlas::max(a, b));
}

TEST(Float3, ClampConstrainsEachAxisToTheBox) {
    const Float3 v(-1.0f, 5.0f, 2.0f);
    const Float3 low(0.0f, 0.0f, 0.0f);
    const Float3 high(3.0f, 3.0f, 3.0f);
    EXPECT_TRUE(atlas::clamp(v, low, high) == Float3(0.0f, 3.0f, 2.0f));
}

TEST(Float3, CeilAndFloorRoundEachAxis) {
    const Float3 v(1.2f, -1.2f, 2.9f);
    EXPECT_TRUE(atlas::ceil(v) == Float3(2.0f, -1.0f, 3.0f));
    EXPECT_TRUE(atlas::floor(v) == Float3(1.0f, -2.0f, 2.0f));
}

TEST(Float3, AbsTakesTheMagnitudeOfEachAxis) {
    EXPECT_TRUE(atlas::abs(Float3(-1.0f, 2.0f, -3.0f)) == Float3(1.0f, 2.0f, 3.0f));
}

TEST(Float3, RelationalOperatorsReturnPerAxisBool3) {
    const Float3 a(1.0f, 5.0f, 3.0f);
    const Float3 b(2.0f, 2.0f, 3.0f);
    const Bool3 lt = a < b;
    EXPECT_TRUE(lt.x);
    EXPECT_FALSE(lt.y);
    EXPECT_FALSE(lt.z);
    EXPECT_TRUE(atlas::all(a <= a));
    EXPECT_TRUE(atlas::any(a > b));
}

TEST(Float3, IsFiniteRejectsANonFiniteComponent) {
    EXPECT_TRUE(atlas::isfinite(Float3(1.0f, 2.0f, 3.0f)));
    EXPECT_FALSE(atlas::isfinite(Float3(1.0f, atlas::inf, 3.0f)));
}

TEST(Float3, XyHelpersIgnoreTheZComponent) {
    const Float3 v(3.0f, 4.0f, 99.0f);
    EXPECT_FLOAT_EQ(atlas::xy_length_squared(v), 25.0f);
    EXPECT_FLOAT_EQ(atlas::xy_length(v), 5.0f);
}

TEST(Float3, NormalizedOrReturnsFallbackForTheZeroVector) {
    const Float3 fallback(1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(atlas::normalized_or(Float3(), fallback) == fallback);
    EXPECT_NEAR(atlas::normalized_or(Float3(0.0f, 3.0f, 0.0f), fallback).y, 1.0f, 1.0e-6f);
}
