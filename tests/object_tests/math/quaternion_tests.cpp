#include <atlas/math/math.h>
#include <cmath>
#include <gtest/gtest.h>

using namespace atlas;

TEST(QuaternionTest, DefaultConstructor) {
    QuaternionF q;
    EXPECT_NEAR(q.w, 1.0f, 1e-5f);
    EXPECT_NEAR(q.x, 0.0f, 1e-5f);
    EXPECT_NEAR(q.y, 0.0f, 1e-5f);
    EXPECT_NEAR(q.z, 0.0f, 1e-5f);
    EXPECT_NEAR(q.length(), 1.0f, 1e-5f);
}

TEST(QuaternionTest, ScalarConstructor) {
    QuaternionF q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_NEAR(q.w, 1.0f, 1e-5f);
    EXPECT_NEAR(q.x, 2.0f, 1e-5f);
    EXPECT_NEAR(q.y, 3.0f, 1e-5f);
    EXPECT_NEAR(q.z, 4.0f, 1e-5f);
}

TEST(QuaternionTest, InitializerList) {
    QuaternionF q { 1, 2, 3, 4 };
    EXPECT_NEAR(q.w, 1.0f, 1e-5f);
    EXPECT_NEAR(q.x, 2.0f, 1e-5f);
    EXPECT_NEAR(q.y, 3.0f, 1e-5f);
    EXPECT_NEAR(q.z, 4.0f, 1e-5f);
}

TEST(QuaternionTest, FromAxisAngle_RotateAroundZ90) {
    auto q = QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(M_PI_2));
    Vector3F v(1, 0, 0);
    Vector3F r = q * v;
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, 1.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
    EXPECT_NEAR(q.length(), 1.0f, 1e-5f);
}

TEST(QuaternionTest, FromEulerXYZ_RotateAroundX90) {
    auto q = QuaternionF::from_euler_xyz(static_cast<float>(M_PI_2), 0.0f, 0.0f);
    Vector3F v(0, 0, 1);
    Vector3F r = q * v;
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, -1.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
}

TEST(QuaternionTest, FromMatrix3x3_AndBack) {
    auto q  = QuaternionF::from_axis_angle(Vector3F(0, 1, 0), static_cast<float>(M_PI) * 0.25f);
    auto m  = q.to_matrix3x3();
    auto q2 = QuaternionF::from_matrix3x3(m).normalized();
    EXPECT_NEAR(std::abs(q.dot(q2)), 1.0f, 1e-5f);
}

TEST(QuaternionTest, MultiplyComposition) {
    auto qx = QuaternionF::from_axis_angle(Vector3F(1, 0, 0), static_cast<float>(M_PI_2));
    auto qz = QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(M_PI_2));
    auto q  = qx * qz;
    Vector3F v(1, 0, 0);
    Vector3F r = q * v;
    EXPECT_NEAR(r.x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
    EXPECT_NEAR(r.z, 1.0f, 1e-5f);
}

TEST(QuaternionTest, ConjugateAndInverse) {
    auto q  = QuaternionF(0.5f, 1.0f, -2.0f, 3.0f).normalized();
    auto qi = q.inverse();
    auto id = q * qi;
    EXPECT_NEAR(id.w, 1.0f, 1e-5f);
    EXPECT_NEAR(id.x, 0.0f, 1e-5f);
    EXPECT_NEAR(id.y, 0.0f, 1e-5f);
    EXPECT_NEAR(id.z, 0.0f, 1e-5f);
}

TEST(QuaternionTest, NormalizeAndLength) {
    QuaternionF q(2.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_NEAR(q.length(), 2.0f, 1e-5f);
    q.normalize();
    EXPECT_NEAR(q.length(), 1.0f, 1e-5f);
}

TEST(QuaternionTest, LerpAndNlerp) {
    auto a = QuaternionF::from_axis_angle(Vector3F(0, 0, 1), 0.0f);
    auto b = QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(M_PI));
    auto l = QuaternionF::lerp(a, b, 0.5f);
    auto n = QuaternionF::nlerp(a, b, 0.5f);
    EXPECT_LT(l.length(), 1.0f + 1e-5f);
    EXPECT_NEAR(n.length(), 1.0f, 1e-5f);
}

TEST(QuaternionTest, SlerpHalfway) {
    auto a = QuaternionF::from_axis_angle(Vector3F(0, 1, 0), 0.0f);
    auto b = QuaternionF::from_axis_angle(Vector3F(0, 1, 0), static_cast<float>(M_PI_2));
    auto m = QuaternionF::slerp(a, b, 0.5f);
    Vector3F v(1, 0, 0);
    Vector3F r = m * v;
    EXPECT_NEAR(r.x, std::cos(static_cast<float>(M_PI_4)), 1e-5f);
    EXPECT_NEAR(r.z, -std::sin(static_cast<float>(M_PI_4)), 1e-5f);
}

TEST(QuaternionTest, RotateVectorOperator) {
    auto q = QuaternionF::from_axis_angle(Vector3F(0, 0, 1), static_cast<float>(M_PI_2));
    Vector3F v(0, 1, 0);
    Vector3F r = q * v;
    EXPECT_NEAR(r.x, -1.0f, 1e-5f);
    EXPECT_NEAR(r.y, 0.0f, 1e-5f);
    EXPECT_NEAR(r.z, 0.0f, 1e-5f);
}

TEST(QuaternionTest, DataPtrAndDot) {
    QuaternionF q(1, 2, 3, 4);
    const float* p = q.data();
    EXPECT_EQ(p[0], q.w);
    EXPECT_EQ(p[1], q.x);
    EXPECT_EQ(p[2], q.y);
    EXPECT_EQ(p[3], q.z);
    QuaternionF r(5, 6, 7, 8);
    EXPECT_NEAR(q.dot(r), 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8, 1e-6f);
}

TEST(QuaternionTest, ToMatrix4x4_RotationConsistency) {
    auto q = QuaternionF::from_axis_angle(Vector3F(1, 0, 0), static_cast<float>(M_PI_2));
    auto M = q.to_matrix4x4();
    Vector3F v(0, 1, 0);
    Vector3F r = q * v;
    Vector<float, 4> vh(v.x, v.y, v.z, 1.0f);
    Vector<float, 4> rh(
        M.m00 * vh.x + M.m01 * vh.y + M.m02 * vh.z + M.m03 * vh.w,
        M.m10 * vh.x + M.m11 * vh.y + M.m12 * vh.z + M.m13 * vh.w,
        M.m20 * vh.x + M.m21 * vh.y + M.m22 * vh.z + M.m23 * vh.w,
        M.m30 * vh.x + M.m31 * vh.y + M.m32 * vh.z + M.m33 * vh.w);
    EXPECT_NEAR(rh.x, r.x, 1e-5f);
    EXPECT_NEAR(rh.y, r.y, 1e-5f);
    EXPECT_NEAR(rh.z, r.z, 1e-5f);
}

TEST(QuaternionTest, CastToDouble) {
    QuaternionF qf(1.25f, -2.5f, 3.75f, -4.5f);
    auto qd = qf.cast_to<double>();
    EXPECT_NEAR(qd.w, 1.25, 1e-12);
    EXPECT_NEAR(qd.x, -2.5, 1e-12);
    EXPECT_NEAR(qd.y, 3.75, 1e-12);
    EXPECT_NEAR(qd.z, -4.5, 1e-12);
}