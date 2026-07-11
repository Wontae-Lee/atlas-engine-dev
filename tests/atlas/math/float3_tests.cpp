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

TEST(Float3, XyDotIgnoresTheZComponent) {
    // xy_dot sums only the x and y products; the z terms must not contribute.
    EXPECT_FLOAT_EQ(atlas::xy_dot(Float3(1.0f, 2.0f, 99.0f), Float3(3.0f, 4.0f, -99.0f)), 11.0f);
}

TEST(Float3, XyNormalizedOrProjectsAndNormalizesInThePlane) {
    // The xy part (3, 4) has length 5, so the unit result is (0.6, 0.8, 0).
    const Float3 fallback(1.0f, 0.0f, 0.0f);
    const Float3 r = atlas::xy_normalized_or(Float3(3.0f, 4.0f, 9.0f), fallback);
    EXPECT_NEAR(r.x, 0.6f, 1.0e-6f);
    EXPECT_NEAR(r.y, 0.8f, 1.0e-6f);
    EXPECT_FLOAT_EQ(r.z, 0.0f);
}

TEST(Float3, XyNormalizedOrReturnsFallbackWhenTheXyPartIsTooShort) {
    // A purely vertical vector has a zero-length xy projection, so the fallback stands in.
    const Float3 fallback(1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(atlas::xy_normalized_or(Float3(0.0f, 0.0f, 5.0f), fallback) == fallback);
}

TEST(Float3, TangentialReturnsAnOrthonormalPairSpanningThePlane) {
    const Float3 n = Float3(1.0f, 2.0f, 3.0f).normalized();
    const auto tangents = n.tangential();
    const Float3 t1 = std::get<0>(tangents);
    const Float3 t2 = std::get<1>(tangents);

    EXPECT_NEAR(t1.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(t2.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(t1.dot(n), 0.0f, 1.0e-5f);
    EXPECT_NEAR(t2.dot(n), 0.0f, 1.0e-5f);
    EXPECT_NEAR(t1.dot(t2), 0.0f, 1.0e-5f);
}

TEST(Float3, FreeTangentialMatchesTheMember) {
    const Float3 n(0.0f, 0.0f, 1.0f);
    const auto member = n.tangential();
    const auto free = atlas::tangential(n);
    EXPECT_TRUE(std::get<0>(free) == std::get<0>(member));
    EXPECT_TRUE(std::get<1>(free) == std::get<1>(member));
}

TEST(Float3, OrthonormalBasisBuildsARightHandedFrame) {
    const Float3 normal(1.0f, 2.0f, 3.0f);
    Float3 unit_normal;
    Float3 tangent;
    Float3 bitangent;
    EXPECT_TRUE(atlas::orthonormal_basis(normal, unit_normal, tangent, bitangent));

    EXPECT_NEAR(unit_normal.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(tangent.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(bitangent.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(unit_normal.dot(tangent), 0.0f, 1.0e-5f);
    EXPECT_NEAR(unit_normal.dot(bitangent), 0.0f, 1.0e-5f);
    EXPECT_NEAR(tangent.dot(bitangent), 0.0f, 1.0e-5f);
    // Right-handed: unit_normal x tangent == bitangent.
    const Float3 cross = unit_normal.cross(tangent);
    EXPECT_NEAR(cross.x, bitangent.x, 1.0e-5f);
    EXPECT_NEAR(cross.y, bitangent.y, 1.0e-5f);
    EXPECT_NEAR(cross.z, bitangent.z, 1.0e-5f);
}

TEST(Float3, OrthonormalBasisFailsForADegenerateNormal) {
    Float3 unit_normal;
    Float3 tangent;
    Float3 bitangent;
    // A zero-length normal cannot be normalized, so the builder reports failure.
    EXPECT_FALSE(atlas::orthonormal_basis(Float3(0.0f, 0.0f, 0.0f), unit_normal, tangent, bitangent));
}

TEST(Float3, OrthonormalBasisTwoOutputOverloadAlsoSucceeds) {
    Float3 tangent;
    Float3 bitangent;
    EXPECT_TRUE(atlas::orthonormal_basis(Float3(0.0f, 0.0f, 5.0f), tangent, bitangent));
    EXPECT_NEAR(tangent.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(bitangent.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(tangent.dot(bitangent), 0.0f, 1.0e-5f);
}

TEST(Float3, OrthogonalUnitVectorIsUnitAndPerpendicular) {
    const Float3 normal(0.0f, 0.0f, 1.0f);
    const Float3 seed(1.0f, 0.0f, 0.0f);
    const Float3 t = atlas::orthogonal_unit_vector(normal, seed);
    EXPECT_NEAR(t.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(t.dot(normal), 0.0f, 1.0e-5f);
}

TEST(Float3, OrthogonalUnitVectorFallsBackWhenSeedIsParallelToNormal) {
    // Seed parallel to normal makes the cross product degenerate, so the arbitrary
    // orthonormal-basis tangent is used; it must still be unit and perpendicular.
    const Float3 normal(0.0f, 0.0f, 1.0f);
    const Float3 t = atlas::orthogonal_unit_vector(normal, normal);
    EXPECT_NEAR(t.length(), 1.0f, 1.0e-5f);
    EXPECT_NEAR(t.dot(normal), 0.0f, 1.0e-5f);
}

TEST(Float3, OrthogonalUnitVectorReturnsWorldXForAFullyDegenerateInput) {
    // A zero normal and zero seed exhaust every construction, leaving the (1,0,0) fallback.
    EXPECT_TRUE(atlas::orthogonal_unit_vector(Float3(), Float3()) == Float3(1.0f, 0.0f, 0.0f));
}

TEST(Float3, SphericalDirectionAboutAnAxisRealizesTheGivenPolarCosine) {
    const Float3 axis(0.0f, 0.0f, 1.0f);
    for (const float cos_theta : { 0.3f, -0.5f, 1.0f }) {
        const Float3 d = atlas::spherical_direction(axis, cos_theta, 0.7f);
        EXPECT_NEAR(d.length(), 1.0f, 1.0e-4f);
        EXPECT_NEAR(d.dot(axis), cos_theta, 1.0e-4f);
    }
}

TEST(Float3, SphericalDirectionCanonicalFrameUsesZAsThePole) {
    // With cos_theta = 0 and phi = 0 the direction lies on +x in the z-pole frame.
    const Float3 d = atlas::spherical_direction(0.0f, 0.0f);
    EXPECT_NEAR(d.x, 1.0f, 1.0e-6f);
    EXPECT_NEAR(d.y, 0.0f, 1.0e-6f);
    EXPECT_NEAR(d.z, 0.0f, 1.0e-6f);
    // The z component always equals the supplied cosine.
    EXPECT_NEAR(atlas::spherical_direction(0.42f, 1.3f).z, 0.42f, 1.0e-6f);
}
