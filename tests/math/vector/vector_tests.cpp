#include "../../utilities/tests_utils.h"
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

using namespace atlas;

TEST(VectorN, DefaultConstructorIsZero) {
    atlas::Vector<double, 5> v;

    EXPECT_EQ(v.size(), static_cast<std::size_t>(5));
    for (std::size_t i = 0; i < atlas::Vector<double, 5>::size(); ++i) {
        EXPECT_DOUBLE_EQ(v[i], 0.0);
    }
    EXPECT_TRUE(atlas::test::is_finite_vec(v));
}

TEST(VectorN, FillConstructor) {
    atlas::Vector<float, 7> v(3.5f);

    for (std::size_t i = 0; i < atlas::Vector<float, 7>::size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 3.5f);
    }
}

TEST(VectorN, VariadicConstructor) {
    atlas::Vector<int, 4> v(1, 2, 3, 4);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
    EXPECT_EQ(v[3], 4);
}

TEST(VectorN, InitializerListPadsWithZero) {
    atlas::Vector<double, 4> v { 1.0, 2.0 };

    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 0.0);
    EXPECT_DOUBLE_EQ(v[3], 0.0);
}

TEST(VectorN, CopyAssignAndEquality) {
    const atlas::Vector<int, 3> a(1, 2, 3);
    atlas::Vector<int, 3> b = a;

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    b[1] = 7;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
}

TEST(VectorN, DataPointerIsContiguous) {
    atlas::Vector<double, 4> v(1.0, 2.0, 3.0, 4.0);

    const double* p = v.data();
    EXPECT_DOUBLE_EQ(p[0], 1.0);
    EXPECT_DOUBLE_EQ(p[1], 2.0);
    EXPECT_DOUBLE_EQ(p[2], 3.0);
    EXPECT_DOUBLE_EQ(p[3], 4.0);

    double* pm = v.data();
    pm[2]      = 9.0;
    EXPECT_DOUBLE_EQ(v[2], 9.0);
}

TEST(VectorN, SetAndSetZero) {
    atlas::Vector<float, 3> v;
    v.set(2.0f);
    for (std::size_t i = 0; i < atlas::Vector<float, 3>::size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 2.0f);
    }

    v.set_zero();
    for (std::size_t i = 0; i < atlas::Vector<float, 3>::size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 0.0f);
    }
}

TEST(VectorN, SetValues) {
    atlas::Vector<double, 3> v;
    v.set_values(1.25, -2.5, 7.0);

    EXPECT_DOUBLE_EQ(v[0], 1.25);
    EXPECT_DOUBLE_EQ(v[1], -2.5);
    EXPECT_DOUBLE_EQ(v[2], 7.0);
}

TEST(VectorN, VectorArithmeticInPlace) {
    atlas::Vector<int, 3> a(1, 2, 3);
    const atlas::Vector<int, 3> b(10, 20, 30);

    a.add(b);
    EXPECT_EQ(a[0], 11);
    EXPECT_EQ(a[1], 22);
    EXPECT_EQ(a[2], 33);

    a.sub(atlas::Vector<int, 3>(1, 1, 1));
    EXPECT_EQ(a[0], 10);
    EXPECT_EQ(a[1], 21);
    EXPECT_EQ(a[2], 32);

    a.mul(atlas::Vector<int, 3>(2, 3, 4));
    EXPECT_EQ(a[0], 20);
    EXPECT_EQ(a[1], 63);
    EXPECT_EQ(a[2], 128);

    a.div(atlas::Vector<int, 3>(2, 7, 8));
    EXPECT_EQ(a[0], 10);
    EXPECT_EQ(a[1], 9);
    EXPECT_EQ(a[2], 16);
}

TEST(VectorN, OperatorCompoundAssignments) {

    atlas::Vector<double, 3> v(1.0, 2.0, 3.0);
    v += 1.0;
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(2.0, 3.0, 4.0), static_cast<double>(eps)));

    v -= 2.0;
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 1.0, 2.0), static_cast<double>(eps)));

    v *= 2.0;
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 2.0, 4.0), static_cast<double>(eps)));

    v /= 2.0;
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 1.0, 2.0), static_cast<double>(eps)));

    v += atlas::Vector<double, 3>(1.0, 1.0, 1.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(1.0, 2.0, 3.0), static_cast<double>(eps)));

    v -= atlas::Vector<double, 3>(1.0, 2.0, 1.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 0.0, 2.0), static_cast<double>(eps)));

    v *= atlas::Vector<double, 3>(3.0, 4.0, 5.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 0.0, 10.0), static_cast<double>(eps)));

    v /= atlas::Vector<double, 3>(1.0, 2.0, 2.0);
    EXPECT_TRUE(atlas::test::vec_near(v, atlas::Vector<double, 3>(0.0, 0.0, 5.0), static_cast<double>(eps)));
}

TEST(VectorN, DotLengthAndNormalize) {

    atlas::Vector<double, 3> v(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(v.dot(v), 25.0);
    EXPECT_DOUBLE_EQ(v.length_squared(), 25.0);
    EXPECT_TRUE(atlas::test::near(v.length(), 5.0, static_cast<double>(eps)));

    const atlas::Vector<double, 3> vn = v.normalized();
    EXPECT_TRUE(atlas::test::is_finite_vec(vn));
    EXPECT_TRUE(atlas::test::near(vn.length(), 1.0, static_cast<double>(eps)));

    v.normalize();
    EXPECT_TRUE(atlas::test::near(v.length(), 1.0, static_cast<double>(eps)));
}

TEST(VectorN, NormalizeZeroIsNoNan) {
    atlas::Vector<double, 3> z;
    z.normalize();
    EXPECT_TRUE(atlas::test::is_finite_vec(z));
    EXPECT_DOUBLE_EQ(z.length(), 0.0);
}

TEST(VectorN, MajorMinorAxisAreByAbsValue) {
    atlas::Vector<double, 4> v(-1.0, 5.0, -3.0, 2.0);

    const std::size_t maj = v.major_axis();
    const std::size_t min = v.minor_axis();

    EXPECT_EQ(maj, static_cast<std::size_t>(1));
    EXPECT_EQ(min, static_cast<std::size_t>(0));

    const double mags[4] = { std::abs(v[0]), std::abs(v[1]), std::abs(v[2]), std::abs(v[3]) };
    for (double mag : mags) {
        EXPECT_LE(mags[min], mag + 1e-12);
        EXPECT_GE(mags[maj] + 1e-12, mag);
    }
}

TEST(VectorN, CastToWorks) {
    const atlas::Vector<double, 3> vd(1.25, -2.5, 7.0);
    atlas::Vector<float, 3> vf = vd.template cast_to<float>();

    EXPECT_FLOAT_EQ(vf[0], 1.25f);
    EXPECT_FLOAT_EQ(vf[1], -2.5f);
    EXPECT_FLOAT_EQ(vf[2], 7.0f);
}

TEST(VectorSpecializations, Vector2Vector3Vector4CompileSmoke) {
    atlas::Vector<double, 2> v2(1.0, 2.0);
    atlas::Vector<double, 3> v3(1.0, 2.0, 3.0);
    atlas::Vector<double, 4> v4(1.0, 2.0, 3.0, 4.0);

    EXPECT_EQ(decltype(v2)::size(), static_cast<std::size_t>(2));
    EXPECT_EQ(decltype(v3)::size(), static_cast<std::size_t>(3));
    EXPECT_EQ(decltype(v4)::size(), static_cast<std::size_t>(4));

    EXPECT_TRUE(atlas::test::is_finite_vec(v2));
    EXPECT_TRUE(atlas::test::is_finite_vec(v3));
    EXPECT_TRUE(atlas::test::is_finite_vec(v4));
}