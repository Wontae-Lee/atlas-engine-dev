#include "../../utilities/tests_utils.h"
#include <cmath>
#include <cstddef>
#include <testkit/testkit.h>

TEST(Vector4, DefaultConstructorIsZero) {
    const atlas::math::Vector<double, 4> v;
    EXPECT_DOUBLE_EQ(v.x, 0.0);
    EXPECT_DOUBLE_EQ(v.y, 0.0);
    EXPECT_DOUBLE_EQ(v.z, 0.0);
    EXPECT_DOUBLE_EQ(v.w, 0.0);
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
}

TEST(Vector4, ScalarFillConstructor) {
    const atlas::math::Vector<double, 4> v(3.5);
    EXPECT_DOUBLE_EQ(v.x, 3.5);
    EXPECT_DOUBLE_EQ(v.y, 3.5);
    EXPECT_DOUBLE_EQ(v.z, 3.5);
    EXPECT_DOUBLE_EQ(v.w, 3.5);
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
}

TEST(Vector4, ComponentConstructorAndIndexing) {
    const atlas::math::Vector<double, 4> v(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
    EXPECT_DOUBLE_EQ(v[3], 4.0);

    EXPECT_DOUBLE_EQ(v.at(0), 1.0);
    EXPECT_DOUBLE_EQ(v.at(1), 2.0);
    EXPECT_DOUBLE_EQ(v.at(2), 3.0);
    EXPECT_DOUBLE_EQ(v.at(3), 4.0);
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
}

TEST(Vector4, InitializerListMissingDefaultsToZero) {
    const atlas::math::Vector<double, 4> v { 1.0, 2.0 };
    EXPECT_DOUBLE_EQ(v.x, 1.0);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
    EXPECT_DOUBLE_EQ(v.z, 0.0);
    EXPECT_DOUBLE_EQ(v.w, 0.0);
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
}

TEST(Vector4, SetAndSetValuesAndZero) {
    atlas::math::Vector<double, 4> v;

    v.set(2.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::math::Vector<double, 4>(2.0, 2.0, 2.0, 2.0), static_cast<double>(atlas::eps)));

    v.set_values(1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::math::Vector<double, 4>(1.0, 2.0, 3.0, 4.0), static_cast<double>(atlas::eps)));

    v.set_zero();
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::math::Vector<double, 4>(0.0, 0.0, 0.0, 0.0), static_cast<double>(atlas::eps)));
}

TEST(Vector4, DotAndLengthSquared) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 4> a(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Vector<double, 4> b(2.0, 0.5, -1.0, 3.0);

    const double d1 = a.dot(b);
    const double d2 = atlas::math::dot(a, b);

    EXPECT_DOUBLE_EQ(d1, 1.0 * 2.0 + 2.0 * 0.5 + 3.0 * (-1.0) + 4.0 * 3.0);
    EXPECT_DOUBLE_EQ(d2, d1);

    const double ls = a.length_squared();
    EXPECT_DOUBLE_EQ(ls, 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0);

    const double len = a.length();
    EXPECT_TRUE(atlas::test::near(len, std::sqrt(ls), eps));
}

TEST(Vector4, NormalizeAndNormalized) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    atlas::math::Vector<double, 4> v(3.0, 4.0, 0.0, 0.0);
    const double ls0 = v.length_squared();
    EXPECT_DOUBLE_EQ(ls0, 25.0);

    const atlas::math::Vector<double, 4> u = v.normalized();
    EXPECT_TRUE(atlas::test::is_finite_vec(u));
    EXPECT_TRUE(atlas::test::near(u.length(), 1.0, eps));

    v.normalize();
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
    EXPECT_TRUE(atlas::test::near(v.length(), 1.0, eps));

    atlas::math::Vector<double, 4> z(0.0, 0.0, 0.0, 0.0);
    z.normalize();
    EXPECT_TRUE(atlas::test::vec_near(z, atlas::math::Vector<double, 4>(0.0, 0.0, 0.0, 0.0), eps));
}

TEST(Vector4, ReflectedAboutUnitNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 4> v(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Vector<double, 4> n(1.0, 0.0, 0.0, 0.0);

    const atlas::math::Vector<double, 4> r1 = v.reflected(n);
    const atlas::math::Vector<double, 4> r2 = atlas::math::reflected(v, n);

    EXPECT_TRUE(atlas::test::is_finite_vec(r1));
    EXPECT_TRUE(atlas::test::is_finite_vec(r2));

    EXPECT_TRUE(atlas::test::vec_near(r1, atlas::math::Vector<double, 4>(-1.0, 2.0, 3.0, 4.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(r2, atlas::math::Vector<double, 4>(-1.0, 2.0, 3.0, 4.0), eps));
}

TEST(Vector4, ProjectedIsOrthogonalToNormal_RejectionDefinition) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 4> v(3.0, 4.0, 5.0, 6.0);
    const atlas::math::Vector<double, 4> n(2.0, 0.0, 0.0, 0.0);

    const atlas::math::Vector<double, 4> p1 = v.projected(n);
    const atlas::math::Vector<double, 4> p2 = atlas::math::projected(v, n);

    EXPECT_TRUE(atlas::test::is_finite_vec(p1));
    EXPECT_TRUE(atlas::test::is_finite_vec(p2));

    EXPECT_TRUE(atlas::test::near(p1.dot(n), 0.0, eps));
    EXPECT_TRUE(atlas::test::near(p2.dot(n), 0.0, eps));

    const atlas::math::Vector<double, 4> vp1 = v - p1;
    const atlas::math::Vector<double, 4> vp2 = v - p2;

    EXPECT_TRUE(atlas::test::near(vp1.y, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(vp1.z, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(vp1.w, 0.0, eps));

    EXPECT_TRUE(atlas::test::near(vp2.y, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(vp2.z, 0.0, eps));
    EXPECT_TRUE(atlas::test::near(vp2.w, 0.0, eps));

    EXPECT_TRUE(atlas::test::vec_near(p1, atlas::math::Vector<double, 4>(0.0, 4.0, 5.0, 6.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(p2, atlas::math::Vector<double, 4>(0.0, 4.0, 5.0, 6.0), eps));
}

TEST(Vector4, ProjectedWithZeroNormalReturnsSelf) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 4> v(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Vector<double, 4> n(0.0, 0.0, 0.0, 0.0);

    const atlas::math::Vector<double, 4> p1 = v.projected(n);
    const atlas::math::Vector<double, 4> p2 = atlas::math::projected(v, n);

    EXPECT_TRUE(atlas::test::vec_near(p1, v, eps));
    EXPECT_TRUE(atlas::test::vec_near(p2, v, eps));
}

TEST(Vector4, MajorMinorAxis_UsesAbsAndLateTieBreak) {
    const atlas::math::Vector<double, 4> v(-1.0, 5.0, 3.0, -5.0);
    const std::size_t maj = v.major_axis();
    EXPECT_EQ(maj, static_cast<std::size_t>(3));

    const std::size_t min = v.minor_axis();
    EXPECT_EQ(min, static_cast<std::size_t>(0));

    const double mags[4] = { std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w) };
    for (double mag : mags) {
        EXPECT_LE(mags[min], mag);
        EXPECT_GE(mags[maj], mag);
    }

    const atlas::math::Vector<double, 4> t(2.0, -2.0, 7.0, 2.0);
    EXPECT_EQ(t.minor_axis(), static_cast<std::size_t>(3));
}

TEST(Vector4, OperatorsBasic) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 4> a(1.0, 2.0, 3.0, 4.0);
    const atlas::math::Vector<double, 4> b(5.0, -1.0, 2.0, 0.5);

    EXPECT_TRUE(atlas::test::vec_near(+a, a, eps));
    EXPECT_TRUE(atlas::test::vec_near(-a, atlas::math::Vector<double, 4>(-1.0, -2.0, -3.0, -4.0), eps));

    EXPECT_TRUE(atlas::test::vec_near(a + b, atlas::math::Vector<double, 4>(6.0, 1.0, 5.0, 4.5), eps));
    EXPECT_TRUE(atlas::test::vec_near(a - b, atlas::math::Vector<double, 4>(-4.0, 3.0, 1.0, 3.5), eps));

    EXPECT_TRUE(atlas::test::vec_near(a * 2.0, atlas::math::Vector<double, 4>(2.0, 4.0, 6.0, 8.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(2.0 * a, atlas::math::Vector<double, 4>(2.0, 4.0, 6.0, 8.0), eps));

    EXPECT_TRUE(atlas::test::vec_near(a * b, atlas::math::Vector<double, 4>(5.0, -2.0, 6.0, 2.0), eps));

    EXPECT_TRUE(atlas::test::vec_near(a / 2.0, atlas::math::Vector<double, 4>(0.5, 1.0, 1.5, 2.0), eps));
    EXPECT_TRUE(atlas::test::vec_near(2.0 / a, atlas::math::Vector<double, 4>(2.0, 1.0, 2.0 / 3.0, 0.5), eps));
    EXPECT_TRUE(atlas::test::vec_near(a / b, atlas::math::Vector<double, 4>(0.2, -2.0, 1.5, 8.0), eps));
}