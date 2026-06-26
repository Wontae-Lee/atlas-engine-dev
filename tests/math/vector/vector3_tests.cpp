#include "../../utilities/test_utils.h"
#include <cmath>
#include <cstddef>
#include <testkit/testkit.h>
#include <type_traits>

TEST(Vector3, DefaultConstructible) {
    const atlas::Vector<float, 3> v {};
    (void)v;
    SUCCEED();
}

TEST(Vector3, UniformScalarConstructor) {
    const atlas::Vector<float, 3> v(2.5f);
    EXPECT_FLOAT_EQ(v.x, 2.5f);
    EXPECT_FLOAT_EQ(v.y, 2.5f);
    EXPECT_FLOAT_EQ(v.z, 2.5f);
}

TEST(Vector3, ComponentConstructor) {
    const atlas::Vector<double, 3> v(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(v.x, 1.0);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
    EXPECT_DOUBLE_EQ(v.z, 3.0);
}

TEST(Vector3, InitializerListConstructor) {
    const atlas::Vector<float, 3> v { 1.0f, 2.0f, 3.0f };
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, DataAndIndexAccess) {
    atlas::Vector<float, 3> v(1.0f, 2.0f, 3.0f);

    const float* p = v.data();
    ASSERT_NE(p, nullptr);
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);

    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);

    v[0] = 10.0f;
    v[1] = 20.0f;
    v[2] = 30.0f;

    EXPECT_FLOAT_EQ(v.x, 10.0f);
    EXPECT_FLOAT_EQ(v.y, 20.0f);
    EXPECT_FLOAT_EQ(v.z, 30.0f);
}

TEST(Vector3, SetAndSetZero) {
    atlas::Vector<float, 3> v(1.0f, 2.0f, 3.0f);

    v.set(7.0f);
    EXPECT_FLOAT_EQ(v.x, 7.0f);
    EXPECT_FLOAT_EQ(v.y, 7.0f);
    EXPECT_FLOAT_EQ(v.z, 7.0f);

    v.set_zero();
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);

    v.set_values(1.0f, 2.0f, 3.0f);
    EXPECT_FLOAT_EQ(v.x, 1.0f);
    EXPECT_FLOAT_EQ(v.y, 2.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, ScalarOpsInPlace) {
    atlas::Vector<float, 3> v(1.0f, 2.0f, 3.0f);

    v.add(1.0f);
    EXPECT_FLOAT_EQ(v.x, 2.0f);
    EXPECT_FLOAT_EQ(v.y, 3.0f);
    EXPECT_FLOAT_EQ(v.z, 4.0f);

    v.sub(2.0f);
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 1.0f);
    EXPECT_FLOAT_EQ(v.z, 2.0f);

    v.mul(3.0f);
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 3.0f);
    EXPECT_FLOAT_EQ(v.z, 6.0f);

    v.div(2.0f);
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 1.5f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST(Vector3, VectorOpsInPlace) {
    atlas::Vector<float, 3> a(1.0f, 2.0f, 3.0f);
    const atlas::Vector<float, 3> b(10.0f, 20.0f, 30.0f);

    a.add(b);
    EXPECT_FLOAT_EQ(a.x, 11.0f);
    EXPECT_FLOAT_EQ(a.y, 22.0f);
    EXPECT_FLOAT_EQ(a.z, 33.0f);

    a.sub(atlas::Vector<float, 3>(1.0f, 2.0f, 3.0f));
    EXPECT_FLOAT_EQ(a.x, 10.0f);
    EXPECT_FLOAT_EQ(a.y, 20.0f);
    EXPECT_FLOAT_EQ(a.z, 30.0f);

    a.mul(atlas::Vector<float, 3>(2.0f, 3.0f, 4.0f));
    EXPECT_FLOAT_EQ(a.x, 20.0f);
    EXPECT_FLOAT_EQ(a.y, 60.0f);
    EXPECT_FLOAT_EQ(a.z, 120.0f);

    a.div(atlas::Vector<float, 3>(2.0f, 3.0f, 4.0f));
    EXPECT_FLOAT_EQ(a.x, 10.0f);
    EXPECT_FLOAT_EQ(a.y, 20.0f);
    EXPECT_FLOAT_EQ(a.z, 30.0f);
}

TEST(Vector3, MinMax) {
    const atlas::Vector<float, 3> v(-1.0f, 3.0f, 2.0f);
    EXPECT_FLOAT_EQ(v.min(), -1.0f);
    EXPECT_FLOAT_EQ(v.max(), 3.0f);
}

TEST(Vector3, DotAndCross) {
    const atlas::Vector<double, 3> a(1.0, 2.0, 3.0);
    const atlas::Vector<double, 3> b(4.0, 5.0, 6.0);

    const double d = a.dot(b);
    EXPECT_DOUBLE_EQ(d, 32.0);

    const atlas::Vector<double, 3> c = a.cross(b);
    EXPECT_DOUBLE_EQ(c.x, -3.0);
    EXPECT_DOUBLE_EQ(c.y, 6.0);
    EXPECT_DOUBLE_EQ(c.z, -3.0);

    const atlas::Vector<double, 3> c2 = atlas::cross(a, b);
    EXPECT_DOUBLE_EQ(c2.x, -3.0);
    EXPECT_DOUBLE_EQ(c2.y, 6.0);
    EXPECT_DOUBLE_EQ(c2.z, -3.0);
}

TEST(Vector3, LengthAndNormalize) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::Vector<double, 3> v(3.0, 4.0, 12.0);

    const double l2 = v.length_squared();
    EXPECT_DOUBLE_EQ(l2, 169.0);

    const double l = v.length();
    EXPECT_DOUBLE_EQ(l, 13.0);

    const atlas::Vector<double, 3> n = v.normalized();
    EXPECT_TRUE(atlas::test::near(n.length(), 1.0, eps));

    v.normalize();
    EXPECT_TRUE(atlas::test::near(v.length(), 1.0, eps));
}

TEST(Vector3, MajorMinorAxisDefinition) {
    const atlas::Vector<float, 3> v(1.0f, -5.0f, 3.0f);

    const std::size_t maj = v.major_axis();
    const std::size_t min = v.minor_axis();

    ASSERT_LT(maj, static_cast<std::size_t>(3));
    ASSERT_LT(min, static_cast<std::size_t>(3));

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_GE(v[maj], v[i]);
        EXPECT_LE(v[min], v[i]);
    }
}

TEST(Vector3, ReflectedAndProjected) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector<double, 3> v(1.0, -2.0, 3.0);
    const atlas::Vector<double, 3> n(0.0, 1.0, 0.0);

    const atlas::Vector<double, 3> r = v.reflected(n);
    EXPECT_TRUE(atlas::test::vec_near(r, atlas::Vector<double, 3>(1.0, 2.0, 3.0), eps));

    const atlas::Vector<double, 3> nn = n.normalized();

    const atlas::Vector<double, 3> p = v.projected(nn);

    const atlas::Vector<double, 3> proj_dir   = nn * v.dot(nn);
    const atlas::Vector<double, 3> proj_plane = v - proj_dir;

    const bool ok = atlas::test::vec_near(p, proj_dir, eps) || atlas::test::vec_near(p, proj_plane, eps);
    EXPECT_TRUE(ok);

    const atlas::Vector<double, 3> r2 = atlas::reflected(v, nn);
    EXPECT_TRUE(atlas::test::vec_near(r2, atlas::Vector<double, 3>(1.0, 2.0, 3.0), eps));

    const atlas::Vector<double, 3> p2 = atlas::projected(v, nn);
    const bool ok2                          = atlas::test::vec_near(p2, proj_dir, eps) || atlas::test::vec_near(p2, proj_plane, eps);
    EXPECT_TRUE(ok2);
}

TEST(Vector3, TangentialBasisOrthogonality) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Vector<double, 3> n(0.0, 0.0, 1.0);

    const auto [t0, t1] = n.tangential();

    EXPECT_TRUE(std::abs(n.dot(t0)) < eps);
    EXPECT_TRUE(std::abs(n.dot(t1)) < eps);
    EXPECT_TRUE(std::abs(t0.dot(t1)) < eps);

    EXPECT_GT(t0.length(), 0.0);
    EXPECT_GT(t1.length(), 0.0);

    const auto [u0, u1] = atlas::tangential(n);
    EXPECT_TRUE(std::abs(n.dot(u0)) < eps);
    EXPECT_TRUE(std::abs(n.dot(u1)) < eps);
}

TEST(Vector3, CastTo) {
    const atlas::Vector<float, 3> vf(1.25f, -2.5f, 3.75f);
    const atlas::Vector<double, 3> vd = vf.cast_to<double>();

    EXPECT_DOUBLE_EQ(vd.x, 1.25);
    EXPECT_DOUBLE_EQ(vd.y, -2.5);
    EXPECT_DOUBLE_EQ(vd.z, 3.75);

    const auto vi = atlas::cast_to<int>(vf);
    EXPECT_EQ(vi.x, 1);
    EXPECT_EQ(vi.y, -2);
    EXPECT_EQ(vi.z, 3);
}

TEST(Vector3, FreeOperatorsBasic) {
    constexpr float eps = atlas::eps;

    const atlas::Vector<float, 3> a(1.0f, 2.0f, 3.0f);
    const atlas::Vector<float, 3> b(10.0f, 20.0f, 30.0f);

    const atlas::Vector<float, 3> s1 = a + b;
    EXPECT_TRUE(atlas::test::vec_near(s1, atlas::Vector<float, 3>(11.0f, 22.0f, 33.0f), eps));

    const atlas::Vector<float, 3> s2 = a - b;
    EXPECT_TRUE(atlas::test::vec_near(s2, atlas::Vector<float, 3>(-9.0f, -18.0f, -27.0f), eps));

    const atlas::Vector<float, 3> s3 = a * 2.0f;
    EXPECT_TRUE(atlas::test::vec_near(s3, atlas::Vector<float, 3>(2.0f, 4.0f, 6.0f), eps));

    const atlas::Vector<float, 3> s4 = 2.0f * a;
    EXPECT_TRUE(atlas::test::vec_near(s4, atlas::Vector<float, 3>(2.0f, 4.0f, 6.0f), eps));

    const atlas::Vector<float, 3> s5 = b / 10.0f;
    EXPECT_TRUE(atlas::test::vec_near(s5, atlas::Vector<float, 3>(1.0f, 2.0f, 3.0f), eps));

    const atlas::Vector<float, 3> s6 = atlas::cmin(a, b);
    EXPECT_TRUE(atlas::test::vec_near(s6, atlas::Vector<float, 3>(1.0f, 2.0f, 3.0f), eps));

    const atlas::Vector<float, 3> s7 = atlas::cmax(a, b);
    EXPECT_TRUE(atlas::test::vec_near(s7, atlas::Vector<float, 3>(10.0f, 20.0f, 30.0f), eps));
}

TEST(Vector3, Point3UIAliasCompiles) {
    const atlas::Point3UI p {};
    (void)p;
    SUCCEED();
}