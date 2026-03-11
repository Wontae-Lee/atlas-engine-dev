#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>
#include <gtest/gtest.h>

TEST(Ray3, DefaultConstructorHasExpectedOriginAndDirection) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::Ray<double> r;

    EXPECT_TRUE(atlas::test::vec_near(r.origin,
                                      atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                                      eps));

    EXPECT_TRUE(atlas::test::vec_near(r.direction,
                                      atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
                                      eps));

    EXPECT_TRUE(atlas::test::near(r.direction.length(), 1.0, eps));
}

TEST(Ray3, ConstructorNormalizesDirection) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> origin(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> dir(3.0, 4.0, 0.0);

    const atlas::Ray<double> r(origin, dir);

    EXPECT_TRUE(atlas::test::vec_near(r.origin, origin, eps));

    EXPECT_TRUE(atlas::test::vec_near(r.direction,
                                      atlas::math::Vector<double, 3>(0.6, 0.8, 0.0),
                                      eps));

    EXPECT_TRUE(atlas::test::near(r.direction.length(), 1.0, eps));
}

TEST(Ray3, PointAtTZeroReturnsOrigin) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> origin(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> dir(0.0, 2.0, 0.0);

    const atlas::Ray<double> r(origin, dir);

    const atlas::math::Vector<double, 3> p = r.point_at(0.0);
    EXPECT_TRUE(atlas::test::vec_near(p, origin, eps));
}

TEST(Ray3, PointAtPositiveTAdvancesAlongDirection) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> origin(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> dir(0.0, 2.0, 0.0);

    const atlas::Ray<double> r(origin, dir);

    const atlas::math::Vector<double, 3> p = r.point_at(5.0);
    EXPECT_TRUE(atlas::test::vec_near(p,
                                      atlas::math::Vector<double, 3>(1.0, 7.0, 3.0),
                                      eps));
}

TEST(Ray3, PointAtNegativeTMovesBehindOrigin) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> origin(1.0, 2.0, 3.0);
    const atlas::math::Vector<double, 3> dir(0.0, 2.0, 0.0);

    const atlas::Ray<double> r(origin, dir);

    const atlas::math::Vector<double, 3> p = r.point_at(-2.0);
    EXPECT_TRUE(atlas::test::vec_near(p,
                                      atlas::math::Vector<double, 3>(1.0, 0.0, 3.0),
                                      eps));
}

TEST(Ray3, UnitDirectionInterpretsTAsDistance) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> origin(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> dir(1.0, 1.0, 0.0);

    const atlas::Ray<double> r(origin, dir);

    const atlas::math::Vector<double, 3> p0 = r.point_at(0.0);
    const atlas::math::Vector<double, 3> p1 = r.point_at(2.0);

    const atlas::math::Vector<double, 3> dp = p1 - p0;
    EXPECT_TRUE(atlas::test::near(dp.length(), 2.0, eps));
}