#include <atlas/math/quaternion.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Float3;
using atlas::pi;
using atlas::Quaternion;

// A convenience axis-angle rotation used across the rotation tests.
Quaternion
rotation_z_90() {
    return Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), pi * 0.5f);
}

}

TEST(Quaternion, DefaultConstructsToTheIdentityRotation) {
    const Quaternion q;
    EXPECT_FLOAT_EQ(q.w, 1.0f);
    EXPECT_FLOAT_EQ(q.x, 0.0f);
    EXPECT_FLOAT_EQ(q.y, 0.0f);
    EXPECT_FLOAT_EQ(q.z, 0.0f);
    EXPECT_TRUE(q.is_identity());
}

TEST(Quaternion, DataAddressesTheFourComponentsInOrder) {
    const Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(q.data(), &q.w);
    EXPECT_FLOAT_EQ(q.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(q.data()[3], 4.0f);
}

TEST(Quaternion, MultiplyingByIdentityIsANoOp) {
    const Quaternion q(0.5f, 0.5f, 0.5f, 0.5f);
    EXPECT_TRUE(q * Quaternion() == q);
    EXPECT_TRUE(Quaternion() * q == q);
}

TEST(Quaternion, MultiplicationIsNotCommutative) {
    const Quaternion a = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), 0.7f);
    const Quaternion b = Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), 1.1f);
    EXPECT_TRUE(a * b != b * a);
}

TEST(Quaternion, ConjugateNegatesTheVectorPart) {
    const Quaternion c = Quaternion(1.0f, 2.0f, 3.0f, 4.0f).conjugate();
    EXPECT_TRUE(c == Quaternion(1.0f, -2.0f, -3.0f, -4.0f));
}

TEST(Quaternion, ProductWithConjugateIsRealForAUnitQuaternion) {
    const Quaternion q = rotation_z_90();
    // q * conjugate(q) has zero vector part and w = |q|^2 = 1 for a unit q.
    EXPECT_TRUE((q * q.conjugate()).is_identity());
}

TEST(Quaternion, LengthAndLengthSquaredAgree) {
    const Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(q.length_squared(), 30.0f);
    EXPECT_FLOAT_EQ(q.length(), std::sqrt(30.0f));
}

TEST(Quaternion, NormalizedHasUnitLength) {
    const Quaternion q = Quaternion(1.0f, 2.0f, 3.0f, 4.0f).normalized();
    EXPECT_NEAR(q.length(), 1.0f, 1.0e-6f);
}

TEST(Quaternion, NormalizeLeavesANearZeroQuaternionUnchanged) {
    Quaternion q(0.0f, 0.0f, 0.0f, 0.0f);
    q.normalize();
    EXPECT_FLOAT_EQ(q.w, 0.0f);
    EXPECT_FLOAT_EQ(q.x, 0.0f);
    EXPECT_FLOAT_EQ(q.y, 0.0f);
    EXPECT_FLOAT_EQ(q.z, 0.0f);
}

TEST(Quaternion, InverseTimesSelfIsTheIdentityEvenForNonUnit) {
    const Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_TRUE((q * q.inverse()).is_identity());
    EXPECT_TRUE((q.inverse() * q).is_identity());
}

TEST(Quaternion, InverseOfANonInvertibleQuaternionIsTheIdentity) {
    EXPECT_TRUE(Quaternion(0.0f, 0.0f, 0.0f, 0.0f).inverse().is_identity());
}

TEST(Quaternion, RotatingByIdentityLeavesTheVectorUnchanged) {
    const Float3 v(1.0f, -2.0f, 3.0f);
    const Float3 r = Quaternion().rotate(v);
    EXPECT_NEAR(r.x, v.x, 1.0e-6f);
    EXPECT_NEAR(r.y, v.y, 1.0e-6f);
    EXPECT_NEAR(r.z, v.z, 1.0e-6f);
}

TEST(Quaternion, NinetyDegreeZRotationMapsXAxisToYAxis) {
    const Float3 r = rotation_z_90().rotate(Float3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(r.x, 0.0f, 1.0e-6f);
    EXPECT_NEAR(r.y, 1.0f, 1.0e-6f);
    EXPECT_NEAR(r.z, 0.0f, 1.0e-6f);
}

TEST(Quaternion, RotationPreservesLength) {
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 r = Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), 0.9f).rotate(v);
    EXPECT_NEAR(r.length(), v.length(), 1.0e-5f);
}

TEST(Quaternion, ConjugateOfAUnitRotationUndoesTheRotation) {
    const Quaternion q = rotation_z_90();
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 back = q.conjugate().rotate(q.rotate(v));
    EXPECT_NEAR(back.x, v.x, 1.0e-5f);
    EXPECT_NEAR(back.y, v.y, 1.0e-5f);
    EXPECT_NEAR(back.z, v.z, 1.0e-5f);
}

TEST(Quaternion, CompositionMatchesSequentialRotation) {
    const Quaternion q1 = Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), 0.6f);
    const Quaternion q2 = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), 1.3f);
    const Float3 v(1.0f, -2.0f, 0.5f);
    // A single rotation by (q2 * q1) equals applying q1 first, then q2.
    const Float3 combined = (q2 * q1).rotate(v);
    const Float3 stepwise = q2.rotate(q1.rotate(v));
    EXPECT_NEAR(combined.x, stepwise.x, 1.0e-5f);
    EXPECT_NEAR(combined.y, stepwise.y, 1.0e-5f);
    EXPECT_NEAR(combined.z, stepwise.z, 1.0e-5f);
}

TEST(Quaternion, ToMatrixThenRotateMatchesDirectRotation) {
    const Quaternion q = rotation_z_90();
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 via_matrix = q.to_matrix3x3().mul(v);
    const Float3 direct = q.rotate(v);
    EXPECT_NEAR(via_matrix.x, direct.x, 1.0e-5f);
    EXPECT_NEAR(via_matrix.y, direct.y, 1.0e-5f);
    EXPECT_NEAR(via_matrix.z, direct.z, 1.0e-5f);
}

TEST(Quaternion, EqualityIsAnEpsilonComparison) {
    EXPECT_TRUE(Quaternion(1.0f, 0.0f, 0.0f, 0.0f) == Quaternion(1.0f, 1.0e-8f, 0.0f, 0.0f));
    EXPECT_TRUE(Quaternion(1.0f, 0.0f, 0.0f, 0.0f) != Quaternion(1.0f, 0.1f, 0.0f, 0.0f));
}

TEST(Quaternion, IsFiniteRejectsANonFiniteComponent) {
    EXPECT_TRUE(atlas::isfinite(Quaternion(1.0f, 2.0f, 3.0f, 4.0f)));
    EXPECT_FALSE(atlas::isfinite(Quaternion(1.0f, atlas::inf, 0.0f, 0.0f)));
}

TEST(Quaternion, DotIsTheFourComponentInnerProduct) {
    const Quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
    const Quaternion b(5.0f, 6.0f, 7.0f, 8.0f);
    EXPECT_FLOAT_EQ(a.dot(b), 70.0f);
    // Dot with itself equals the squared norm.
    EXPECT_FLOAT_EQ(a.dot(a), a.length_squared());
}

TEST(Quaternion, FromEulerXyzWithOnlyXMatchesAxisAngleAboutX) {
    // ry = rz = 0 collapses qz * qy * qx to qx alone.
    const Quaternion euler = Quaternion::from_euler_xyz(0.7f, 0.0f, 0.0f);
    const Quaternion axis  = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), 0.7f);
    EXPECT_TRUE(euler == axis);
}

TEST(Quaternion, FromEulerXyzAppliesXThenYThenZ) {
    const float rx = 0.4f;
    const float ry = 0.9f;
    const float rz = 1.2f;
    const Quaternion composed = Quaternion::from_euler_xyz(rx, ry, rz);
    const Quaternion qx = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), rx);
    const Quaternion qy = Quaternion::from_axis_angle(Float3(0.0f, 1.0f, 0.0f), ry);
    const Quaternion qz = Quaternion::from_axis_angle(Float3(0.0f, 0.0f, 1.0f), rz);

    const Float3 v(1.0f, -2.0f, 0.5f);
    const Float3 direct   = composed.rotate(v);
    const Float3 stepwise = qz.rotate(qy.rotate(qx.rotate(v)));
    EXPECT_NEAR(direct.x, stepwise.x, 1.0e-5f);
    EXPECT_NEAR(direct.y, stepwise.y, 1.0e-5f);
    EXPECT_NEAR(direct.z, stepwise.z, 1.0e-5f);
}

TEST(Quaternion, FromMatrixRoundTripsThroughToMatrix) {
    const Quaternion q = rotation_z_90();
    const Quaternion back = Quaternion::from_matrix3x3(q.to_matrix3x3());
    // The reconstructed rotation must act identically on a probe vector.
    const Float3 v(1.0f, 2.0f, 3.0f);
    const Float3 a = q.rotate(v);
    const Float3 b = back.rotate(v);
    EXPECT_NEAR(a.x, b.x, 1.0e-5f);
    EXPECT_NEAR(a.y, b.y, 1.0e-5f);
    EXPECT_NEAR(a.z, b.z, 1.0e-5f);
}

TEST(Quaternion, LerpReturnsTheEndpointsAtZeroAndOne) {
    const Quaternion a(0.5f, 0.5f, 0.5f, 0.5f);
    const Quaternion b = rotation_z_90();
    EXPECT_TRUE(Quaternion::lerp(a, b, 0.0f) == a);
    EXPECT_TRUE(Quaternion::lerp(a, b, 1.0f) == b);
}

TEST(Quaternion, NlerpAtEndpointsYieldsTheNormalizedInputs) {
    const Quaternion a = rotation_z_90();
    const Quaternion b = Quaternion::from_axis_angle(Float3(1.0f, 0.0f, 0.0f), 1.1f);
    EXPECT_TRUE(Quaternion::nlerp(a, b, 0.0f) == a);
    EXPECT_TRUE(Quaternion::nlerp(a, b, 1.0f) == b);
}

TEST(Quaternion, SlerpMidpointIsTheHalfwayRotation) {
    // Halfway from the identity to a 90-degree z rotation is a 45-degree z rotation.
    const Quaternion mid = Quaternion::slerp(Quaternion(), rotation_z_90(), 0.5f);
    const Float3 r = mid.rotate(Float3(1.0f, 0.0f, 0.0f));
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(r.x, inv_sqrt2, 1.0e-5f);
    EXPECT_NEAR(r.y, inv_sqrt2, 1.0e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1.0e-5f);
}

TEST(Quaternion, SlerpTakesTheShortArcAcrossTheDoubleCover) {
    // q and -q are the same rotation; slerp must flip the negated endpoint back so
    // the interpolated midpoint matches the non-negated one.
    const Quaternion a;
    const Quaternion b = rotation_z_90();
    const Quaternion negated(-b.w, -b.x, -b.y, -b.z);

    const Float3 v(1.0f, 0.0f, 0.0f);
    const Float3 from_b   = Quaternion::slerp(a, b, 0.5f).rotate(v);
    const Float3 from_neg = Quaternion::slerp(a, negated, 0.5f).rotate(v);
    EXPECT_NEAR(from_b.x, from_neg.x, 1.0e-5f);
    EXPECT_NEAR(from_b.y, from_neg.y, 1.0e-5f);
    EXPECT_NEAR(from_b.z, from_neg.z, 1.0e-5f);
}
