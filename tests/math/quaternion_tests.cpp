#include "../utilities/test_utils.h"
#include <cmath>
#include <testkit/testkit.h>
#include <type_traits>

TEST(Quaternion, DefaultConstructorIsIdentity) {
    const atlas::math::Quaternion<double> q;
    EXPECT_DOUBLE_EQ(q.w, 1.0);
    EXPECT_DOUBLE_EQ(q.x, 0.0);
    EXPECT_DOUBLE_EQ(q.y, 0.0);
    EXPECT_DOUBLE_EQ(q.z, 0.0);
    EXPECT_TRUE(atlas::test::is_finite_vec(atlas::Vector<double, 4>(q.x, q.y, q.z, q.w)));
}

TEST(Quaternion, ComponentConstructor) {
    const atlas::math::Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(q.w, 1.0);
    EXPECT_DOUBLE_EQ(q.x, 2.0);
    EXPECT_DOUBLE_EQ(q.y, 3.0);
    EXPECT_DOUBLE_EQ(q.z, 4.0);
}

TEST(Quaternion, InitializerListDefaults) {
    const atlas::math::Quaternion<double> q0 {};
    EXPECT_DOUBLE_EQ(q0.w, 1.0);
    EXPECT_DOUBLE_EQ(q0.x, 0.0);
    EXPECT_DOUBLE_EQ(q0.y, 0.0);
    EXPECT_DOUBLE_EQ(q0.z, 0.0);

    const atlas::math::Quaternion<double> q1 { 0.25 };
    EXPECT_DOUBLE_EQ(q1.w, 0.25);
    EXPECT_DOUBLE_EQ(q1.x, 0.0);
    EXPECT_DOUBLE_EQ(q1.y, 0.0);
    EXPECT_DOUBLE_EQ(q1.z, 0.0);

    const atlas::math::Quaternion<double> q2 { 0.25, 1.0, 2.0 };
    EXPECT_DOUBLE_EQ(q2.w, 0.25);
    EXPECT_DOUBLE_EQ(q2.x, 1.0);
    EXPECT_DOUBLE_EQ(q2.y, 2.0);
    EXPECT_DOUBLE_EQ(q2.z, 0.0);
}

TEST(Quaternion, DataPointerIsContiguous) {
    atlas::math::Quaternion<double> q(1.0, 2.0, 3.0, 4.0);

    const double* p = q.data();
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    double* pm = q.data();
    pm[2]      = 9.0;
    EXPECT_DOUBLE_EQ(q.y, 9.0);
}

TEST(Quaternion, DotLengthAndNormalize) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Quaternion<double> q(1.0, 2.0, 3.0, 4.0);

    const double d = q.dot(q);
    EXPECT_DOUBLE_EQ(d, q.length_squared());
    EXPECT_TRUE(atlas::test::near(q.length(), std::sqrt(q.length_squared()), eps));

    const atlas::math::Quaternion<double> n = q.normalized();
    EXPECT_TRUE(atlas::test::near(n.length(), 1.0, eps));

    q.normalize();
    EXPECT_TRUE(atlas::test::near(q.length(), 1.0, eps));
}

TEST(Quaternion, NormalizeZeroIsSafe) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Quaternion<double> q(0.0, 0.0, 0.0, 0.0);
    q.normalize();

    EXPECT_TRUE(atlas::test::near(q.length(), 0.0, eps));
    EXPECT_DOUBLE_EQ(q.w, 0.0);
    EXPECT_DOUBLE_EQ(q.x, 0.0);
    EXPECT_DOUBLE_EQ(q.y, 0.0);
    EXPECT_DOUBLE_EQ(q.z, 0.0);
}

TEST(Quaternion, ConjugateAndInverse) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Quaternion<double> q(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Quaternion<double> c = q.conjugate();

    EXPECT_DOUBLE_EQ(c.w, q.w);
    EXPECT_DOUBLE_EQ(c.x, -q.x);
    EXPECT_DOUBLE_EQ(c.y, -q.y);
    EXPECT_DOUBLE_EQ(c.z, -q.z);

    const atlas::math::Quaternion<double> inv = q.inverse();
    const atlas::math::Quaternion<double> id  = q * inv;

    EXPECT_TRUE(atlas::test::near(id.x, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(id.y, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(id.z, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(id.w, 1.0, eps));
}

TEST(Quaternion, IsIdentity) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Quaternion<double> q;
    EXPECT_TRUE(q.is_identity(static_cast<double>(atlas::eps)));

    const atlas::math::Quaternion<double> r(1.0, eps * 2.0, 0.0, 0.0);
    EXPECT_FALSE(r.is_identity(static_cast<double>(atlas::eps)));
}

TEST(Quaternion, AxisAngleConstructorMatchesFactory) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> axis(0.0, 0.0, 1.0);
    constexpr double angle = 0.5;

    const atlas::math::Quaternion<double> q1(axis, angle);
    const atlas::math::Quaternion<double> q2 = atlas::math::Quaternion<double>::from_axis_angle(axis, angle);

    EXPECT_TRUE(atlas::test::near(q1.w, q2.w, eps));
    EXPECT_TRUE(atlas::test::near(q1.x, q2.x, eps));
    EXPECT_TRUE(atlas::test::near(q1.y, q2.y, eps));
    EXPECT_TRUE(atlas::test::near(q1.z, q2.z, eps));
}

TEST(Quaternion, EulerXYZConstructorMatchesFactory) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    constexpr double rx = 0.25;
    constexpr double ry = -0.5;
    constexpr double rz = 0.75;

    const atlas::math::Quaternion<double> q1(rx, ry, rz);
    const atlas::math::Quaternion<double> q2 = atlas::math::Quaternion<double>::from_euler_xyz(rx, ry, rz);

    EXPECT_TRUE(atlas::test::near(q1.w, q2.w, eps));
    EXPECT_TRUE(atlas::test::near(q1.x, q2.x, eps));
    EXPECT_TRUE(atlas::test::near(q1.y, q2.y, eps));
    EXPECT_TRUE(atlas::test::near(q1.z, q2.z, eps));
}

TEST(Quaternion, RotateKeepsVectorLengthForUnitQuaternion) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> axis(0.0, 0.0, 1.0);
    atlas::math::Quaternion<double> q(axis, 1.0);
    q.normalize();

    const atlas::math::Vector<double, 3> v(3.0, 4.0, 0.0);
    const atlas::math::Vector<double, 3> r = q.rotate(v);

    EXPECT_TRUE(atlas::test::near(r.length(), v.length(), eps));
}

TEST(Quaternion, ToMatrix3x3RoundTrip) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Quaternion<double> q(atlas::math::Vector<double, 3>(0.0, 1.0, 0.0), 0.7);
    q.normalize();

    const atlas::math::Matrix<double, 3, 3> m = q.to_matrix3x3();
    const atlas::math::Quaternion<double> p(m);

    const atlas::math::Quaternion<double> qn = q.normalized();
    const atlas::math::Quaternion<double> pn = p.normalized();

    const double d = std::abs(qn.dot(pn));
    EXPECT_TRUE(atlas::test::near(d, 1.0, eps));
}

TEST(Quaternion, LerpNlerpSlerpEndpoints) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Quaternion<double> a(atlas::math::Vector<double, 3>(1.0, 0.0, 0.0), 0.3);
    const atlas::math::Quaternion<double> b(atlas::math::Vector<double, 3>(0.0, 1.0, 0.0), 1.1);

    const atlas::math::Quaternion<double> l0 = atlas::math::Quaternion<double>::lerp(a, b, 0.0);
    const atlas::math::Quaternion<double> l1 = atlas::math::Quaternion<double>::lerp(a, b, 1.0);
    EXPECT_TRUE(atlas::test::near(std::abs(l0.dot(a)), a.length_squared(), eps));
    EXPECT_TRUE(atlas::test::near(std::abs(l1.dot(b)), b.length_squared(), eps));

    const atlas::math::Quaternion<double> n0 = atlas::math::Quaternion<double>::nlerp(a, b, 0.0);
    const atlas::math::Quaternion<double> n1 = atlas::math::Quaternion<double>::nlerp(a, b, 1.0);
    EXPECT_TRUE(atlas::test::near(std::abs(n0.dot(a.normalized())), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(std::abs(n1.dot(b.normalized())), 1.0, eps));

    const atlas::math::Quaternion<double> s0 = atlas::math::Quaternion<double>::slerp(a, b, 0.0);
    const atlas::math::Quaternion<double> s1 = atlas::math::Quaternion<double>::slerp(a, b, 1.0);
    EXPECT_TRUE(atlas::test::near(std::abs(s0.dot(a.normalized())), 1.0, eps));
    EXPECT_TRUE(atlas::test::near(std::abs(s1.dot(b.normalized())), 1.0, eps));
}

TEST(Quaternion, OperatorsBasic) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Quaternion<double> a(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Quaternion<double> b(0.5, -1.0, 2.0, 0.25);

    const atlas::math::Quaternion<double> s1 = a + b;
    EXPECT_TRUE(atlas::test::near(s1.w, 1.5, eps));
    EXPECT_TRUE(atlas::test::near(s1.x, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(s1.y, 5.0, eps));
    EXPECT_TRUE(atlas::test::near(s1.z, 4.25, eps));

    const atlas::math::Quaternion<double> s2 = a - b;
    EXPECT_TRUE(atlas::test::near(s2.w, 0.5, eps));
    EXPECT_TRUE(atlas::test::near(s2.x, 3.0, eps));
    EXPECT_TRUE(atlas::test::near(s2.y, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(s2.z, 3.75, eps));

    const atlas::math::Quaternion<double> s3 = a * 2.0;
    EXPECT_TRUE(atlas::test::near(s3.w, 2.0, eps));
    EXPECT_TRUE(atlas::test::near(s3.x, 4.0, eps));
    EXPECT_TRUE(atlas::test::near(s3.y, 6.0, eps));
    EXPECT_TRUE(atlas::test::near(s3.z, 8.0, eps));

    const atlas::math::Quaternion<double> s4 = a / 2.0;
    EXPECT_TRUE(atlas::test::near(s4.w, 0.5, eps));
    EXPECT_TRUE(atlas::test::near(s4.x, 1.0, eps));
    EXPECT_TRUE(atlas::test::near(s4.y, 1.5, eps));
    EXPECT_TRUE(atlas::test::near(s4.z, 2.0, eps));
}

TEST(Quaternion, CastTo) {
    const atlas::math::Quaternion<float> qf(1.25f, -2.5f, 3.75f, 0.5f);
    const atlas::math::Quaternion<double> qd = qf.cast_to<double>();

    EXPECT_DOUBLE_EQ(qd.w, 1.25);
    EXPECT_DOUBLE_EQ(qd.x, -2.5);
    EXPECT_DOUBLE_EQ(qd.y, 3.75);
    EXPECT_DOUBLE_EQ(qd.z, 0.5);
}