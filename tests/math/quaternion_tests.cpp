#include <atlas/math/quaternion.h>

#include <cmath>
#include <gtest/gtest.h>

namespace {

constexpr float k_eps = 1e-5f;

constexpr float k_pi = 3.14159265358979323846f;

void
expect_vector_near(const atlas::Vector3& a, const atlas::Vector3& b, const float tolerance) {
    EXPECT_NEAR(a.x, b.x, tolerance);
    EXPECT_NEAR(a.y, b.y, tolerance);
    EXPECT_NEAR(a.z, b.z, tolerance);
}

}

TEST(Quaternion, DefaultConstructorIsIdentity) {
    const atlas::Quaternion q;
    EXPECT_FLOAT_EQ(q.w, 1.0f);
    EXPECT_FLOAT_EQ(q.x, 0.0f);
    EXPECT_FLOAT_EQ(q.y, 0.0f);
    EXPECT_FLOAT_EQ(q.z, 0.0f);
    EXPECT_TRUE(q.is_identity());
}

TEST(Quaternion, ComponentConstructor) {
    const atlas::Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_FLOAT_EQ(q.w, 1.0f);
    EXPECT_FLOAT_EQ(q.x, 2.0f);
    EXPECT_FLOAT_EQ(q.y, 3.0f);
    EXPECT_FLOAT_EQ(q.z, 4.0f);
}

TEST(Quaternion, DotLengthNormalize) {
    atlas::Quaternion q(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_FLOAT_EQ(q.length_squared(), 30.0f);
    EXPECT_FLOAT_EQ(q.dot(q), 30.0f);
    EXPECT_NEAR(q.length(), std::sqrt(30.0f), k_eps);

    const atlas::Quaternion n = q.normalized();
    EXPECT_NEAR(n.length(), 1.0f, k_eps);

    q.normalize();
    EXPECT_NEAR(q.length(), 1.0f, k_eps);
}

TEST(Quaternion, AxisAngleRotationRotatesVector) {
    const atlas::Vector3 axis(0.0f, 0.0f, 1.0f);
    const atlas::Quaternion q = atlas::Quaternion::from_axis_angle(axis, k_pi * 0.5f);

    const atlas::Vector3 r = q.rotate(atlas::Vector3(1.0f, 0.0f, 0.0f));
    expect_vector_near(r, atlas::Vector3(0.0f, 1.0f, 0.0f), k_eps);
}

TEST(Quaternion, ConjugateReversesRotation) {
    const atlas::Quaternion q = atlas::Quaternion::from_axis_angle(atlas::Vector3(0.0f, 1.0f, 0.0f), 0.7f);
    const atlas::Vector3 v(1.0f, 2.0f, 3.0f);

    const atlas::Vector3 round_trip = q.conjugate().rotate(q.rotate(v));
    expect_vector_near(round_trip, v, k_eps);
}

TEST(Quaternion, InverseUndoesMultiplication) {
    const atlas::Quaternion q = atlas::Quaternion::from_euler_xyz(0.3f, -0.2f, 0.9f);

    const atlas::Quaternion qi = q.inverse();
    EXPECT_TRUE((q * qi).is_identity(1e-4f));
}

TEST(Quaternion, MatrixRoundTrip) {
    const atlas::Quaternion q = atlas::Quaternion::from_axis_angle(
        atlas::Vector3(1.0f, 1.0f, 0.0f).normalized(), 0.8f);

    const atlas::Matrix3x3 m  = q.to_matrix3x3();
    const atlas::Quaternion r = atlas::Quaternion::from_matrix3x3(m);

    const float sign = (q.dot(r) < 0.0f) ? -1.0f : 1.0f;
    EXPECT_NEAR(r.w * sign, q.w, k_eps);
    EXPECT_NEAR(r.x * sign, q.x, k_eps);
    EXPECT_NEAR(r.y * sign, q.y, k_eps);
    EXPECT_NEAR(r.z * sign, q.z, k_eps);
}

TEST(Quaternion, MatrixMatchesRotate) {
    const atlas::Quaternion q = atlas::Quaternion::from_euler_xyz(0.1f, 0.4f, -0.3f);
    const atlas::Vector3 v(0.5f, -1.5f, 2.0f);

    const atlas::Vector3 by_quat   = q.rotate(v);
    const atlas::Vector3 by_matrix = q.to_matrix3x3() * v;
    expect_vector_near(by_quat, by_matrix, k_eps);
}

TEST(Quaternion, ArithmeticOperators) {
    const atlas::Quaternion a(1.0f, 2.0f, 3.0f, 4.0f);
    const atlas::Quaternion b(0.5f, 0.5f, 0.5f, 0.5f);

    const atlas::Quaternion s = a + b;
    EXPECT_FLOAT_EQ(s.w, 1.5f);

    const atlas::Quaternion d = a - b;
    EXPECT_FLOAT_EQ(d.z, 3.5f);

    const atlas::Quaternion m = a * 2.0f;
    EXPECT_FLOAT_EQ(m.x, 4.0f);

    const atlas::Quaternion q = a / 2.0f;
    EXPECT_FLOAT_EQ(q.y, 1.5f);

    atlas::Quaternion c = a;
    c += b;
    c -= b;
    c *= 2.0f;
    c /= 2.0f;
    EXPECT_TRUE(c == a);
}

TEST(Quaternion, IdentityIsMultiplicativeNeutral) {
    const atlas::Quaternion identity;
    const atlas::Quaternion q = atlas::Quaternion::from_euler_xyz(0.2f, 0.3f, 0.4f);

    EXPECT_TRUE((q * identity) == q);
    EXPECT_TRUE((identity * q) == q);
}

TEST(Quaternion, SlerpEndpointsAndMidpoint) {
    const atlas::Quaternion a;
    const atlas::Quaternion b = atlas::Quaternion::from_axis_angle(atlas::Vector3(0.0f, 0.0f, 1.0f), k_pi * 0.5f);

    EXPECT_TRUE(atlas::Quaternion::slerp(a, b, 0.0f) == a);
    EXPECT_TRUE(atlas::Quaternion::slerp(a, b, 1.0f) == b);

    const atlas::Quaternion mid      = atlas::Quaternion::slerp(a, b, 0.5f);
    const atlas::Quaternion expected = atlas::Quaternion::from_axis_angle(atlas::Vector3(0.0f, 0.0f, 1.0f), k_pi * 0.25f);
    EXPECT_TRUE(mid == expected);
}

TEST(Quaternion, NlerpIsNormalized) {
    const atlas::Quaternion a;
    const atlas::Quaternion b = atlas::Quaternion::from_axis_angle(atlas::Vector3(1.0f, 0.0f, 0.0f), 1.0f);

    const atlas::Quaternion n = atlas::Quaternion::nlerp(a, b, 0.3f);
    EXPECT_NEAR(n.length(), 1.0f, k_eps);
}

TEST(Quaternion, IsFinite) {
    EXPECT_TRUE(atlas::isfinite(atlas::Quaternion()));
    EXPECT_FALSE(atlas::isfinite(atlas::Quaternion(atlas::inf, 0.0f, 0.0f, 0.0f)));
}
