#include "../../utilities/tests_utils.h"
#include <atlas/math/math.h>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

using namespace atlas;

TEST(Vector2, ScalarCtor) {
    const atlas::math::Vector<double, 2> v(3.0);
    EXPECT_DOUBLE_EQ(v.x, 3.0);
    EXPECT_DOUBLE_EQ(v.y, 3.0);
}

TEST(Vector2, ComponentCtorAndIndex) {
    atlas::math::Vector<double, 2> v(1.0, -2.0);

    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], -2.0);

    v[0] = -3.0;
    v[1] = 4.0;
    EXPECT_DOUBLE_EQ(v.x, -3.0);
    EXPECT_DOUBLE_EQ(v.y, 4.0);
}

TEST(Vector2, DotAndCrossBasic) {
    const atlas::math::Vector<double, 2> a(1.0, 2.0);
    const atlas::math::Vector<double, 2> b(3.0, -4.0);

    EXPECT_DOUBLE_EQ(a.dot(b), 1.0 * 3.0 + 2.0 * (-4.0));
    EXPECT_DOUBLE_EQ(atlas::math::dot(a, b), a.dot(b));

    const double cr = a.cross(b);
    EXPECT_DOUBLE_EQ(cr, 1.0 * (-4.0) - 2.0 * 3.0);
    EXPECT_DOUBLE_EQ(atlas::math::cross(a, b), cr);
}

TEST(Vector2, LengthAndNormalize) {

    atlas::math::Vector<double, 2> v(3.0, 4.0);
    EXPECT_DOUBLE_EQ(v.length_squared(), 25.0);
    EXPECT_DOUBLE_EQ(v.length(), 5.0);

    const atlas::math::Vector<double, 2> n = v.normalized();
    EXPECT_TRUE(atlas::test::is_finite_vec(n));
    EXPECT_TRUE(atlas::test::near(n.length(), 1.0, static_cast<double>(atlas::eps)));

    v.normalize();
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
    EXPECT_TRUE(atlas::test::near(v.length(), 1.0, static_cast<double>(atlas::eps)));
}

TEST(Vector2, ProjectedIsOrthogonalToNormal) {

    const atlas::math::Vector<double, 2> v(3.0, 4.0);
    const atlas::math::Vector<double, 2> n(2.0, 0.0);

    const atlas::math::Vector<double, 2> p1 = v.projected(n);
    const atlas::math::Vector<double, 2> p2 = atlas::math::projected(v, n);

    EXPECT_TRUE(atlas::test::is_finite_vec(p1));
    EXPECT_TRUE(atlas::test::is_finite_vec(p2));

    EXPECT_TRUE(atlas::test::near(p1.dot(n), 0.0, static_cast<double>(atlas::eps)));
    EXPECT_TRUE(atlas::test::near(p2.dot(n), 0.0, static_cast<double>(atlas::eps)));

    EXPECT_TRUE(atlas::test::near((v - p1).cross(n), 0.0, static_cast<double>(atlas::eps)));
    EXPECT_TRUE(atlas::test::near((v - p2).cross(n), 0.0, static_cast<double>(atlas::eps)));
}

TEST(Vector2, TangentialIsPerpendicular) {

    const atlas::math::Vector<double, 2> v(1.0, 2.0);
    const atlas::math::Vector<double, 2> t = v.tangential();

    EXPECT_TRUE(atlas::test::is_finite_vec(t));
    EXPECT_TRUE(atlas::test::near(v.dot(t), 0.0, static_cast<double>(atlas::eps)));
    EXPECT_GT(t.length_squared(), 0.0);
}

TEST(Vector2, MajorMinorAxisUsesAbsMagnitude) {
    const atlas::math::Vector<double, 2> v(1.0, -5.0);

    const std::size_t maj = v.major_axis();
    const std::size_t min = v.minor_axis();

    ASSERT_LT(maj, static_cast<std::size_t>(2));
    ASSERT_LT(min, static_cast<std::size_t>(2));

    const double mags[2] = { std::abs(v[0]), std::abs(v[1]) };

    for (double mag : mags) {
        EXPECT_GE(mags[maj] + static_cast<double>(atlas::eps), mag);
        EXPECT_LE(mags[min] - static_cast<double>(atlas::eps), mag);
    }
}

TEST(Vector2, ReflectedBasicProperty) {

    const atlas::math::Vector<double, 2> n(0.0, 1.0);
    const atlas::math::Vector<double, 2> v(1.0, -2.0);

    const atlas::math::Vector<double, 2> r1 = v.reflected(n);
    const atlas::math::Vector<double, 2> r2 = atlas::math::reflected(v, n);

    EXPECT_TRUE(atlas::test::is_finite_vec(r1));
    EXPECT_TRUE(atlas::test::is_finite_vec(r2));

    EXPECT_TRUE(atlas::test::near(r1.dot(n), -v.dot(n), static_cast<double>(atlas::eps)));
    EXPECT_TRUE(atlas::test::near(r2.dot(n), -v.dot(n), static_cast<double>(atlas::eps)));
}

TEST(Vector2, CastTo) {
    const atlas::math::Vector<float, 2> vf(1.25f, -2.5f);
    const atlas::math::Vector<double, 2> vd = vf.cast_to<double>();

    EXPECT_NEAR(vd.x, 1.25, static_cast<double>(atlas::eps));
    EXPECT_NEAR(vd.y, -2.5, static_cast<double>(atlas::eps));
}