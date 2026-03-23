#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>

#include <gtest/gtest.h>

TEST(GeometryOperator_Box, MissingParamsReturnsMiss) {
    const atlas::math::Vector<double, 3> o(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> d(1.0, 0.0, 0.0);
    const atlas::Ray<double> r(o, d);

    constexpr atlas::geometry::BoxGeometryOperator<double> op;
    const auto h = op(r);

    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Box, HitFromOutsideReturnsEnterDistancePointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::geometry::BoxGeometryOperator<double> op;
    op.lower_corner = &lo;
    op.upper_corner = &hi;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 0.2, -0.3),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 2.0, eps));

    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(-1.0, 0.2, -0.3),
                                      eps));

    EXPECT_TRUE(atlas::test::vec_near(normal,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
}

TEST(GeometryOperator_Box, FromInsideReturnsExitDistanceAndOutwardNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::geometry::BoxGeometryOperator<double> op;
    op.lower_corner = &lo;
    op.upper_corner = &hi;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.25, -0.5),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 1.0, eps));

    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(1.0, 0.25, -0.5),
                                      eps));

    EXPECT_TRUE(atlas::test::vec_near(normal,
                                      atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
                                      eps));
}

TEST(GeometryOperator_Box, MissWhenIntervalsDoNotOverlap) {
    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::geometry::BoxGeometryOperator<double> op;
    op.lower_corner = &lo;
    op.upper_corner = &hi;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 2.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Cylinder, MissingParamsReturnsMiss) {
    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    constexpr atlas::geometry::CylinderGeometryOperator<double> op;
    const auto h = op(r);

    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Cylinder, SideHitReturnsCorrectDistancePointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;
    constexpr double height = 2.0;

    atlas::geometry::CylinderGeometryOperator<double> op;
    op.center = &c;
    op.radius = &radius;
    op.height = &height;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 2.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));

    EXPECT_TRUE(atlas::test::vec_near(normal,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
}

TEST(GeometryOperator_Cylinder, CapHitReturnsNormalAlongZ) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;
    constexpr double height = 2.0;

    atlas::geometry::CylinderGeometryOperator<double> op;
    op.center = &c;
    op.radius = &radius;
    op.height = &height;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.25, 0.25, 3.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op(r);

    EXPECT_TRUE(h.is_intersecting);
    EXPECT_TRUE(atlas::test::near(h.point.z, 1.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(h.normal,
                                      atlas::math::Vector<double, 3>(0.0, 0.0, 1.0),
                                      eps));
}

TEST(GeometryOperator_Plane, MissingParamsReturnsMiss) {
    constexpr atlas::geometry::PlaneGeometryOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(0.0, 1.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Plane, ParallelNoHitWhenNotOnPlane) {
    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::geometry::PlaneGeometryOperator<double> op;
    op.normal = &n;
    op.offset = &d;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Plane, ParallelOnPlaneReturnsTZero) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::geometry::PlaneGeometryOperator<double> op;
    op.normal = &n;
    op.offset = &d;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(2.0, 1.0, -3.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);
    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 0.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point, r.origin, eps));
    EXPECT_TRUE(atlas::test::vec_near(normal, n, eps));
}

TEST(GeometryOperator_Plane, HitComputesCorrectTPointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::geometry::PlaneGeometryOperator<double> op;
    op.normal = &n;
    op.offset = &d;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(0.0, 1.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 1.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(0.0, 1.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(normal, n, eps));
}

TEST(GeometryOperator_Sphere, MissingParamsReturnsMiss) {
    constexpr atlas::geometry::SphereGeometryOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Sphere, HitReturnsNearestPositiveRootAndUnitNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::geometry::SphereGeometryOperator<double> op;
    op.center = &c;
    op.radius = &radius;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 2.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(normal,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::near(normal.length(), 1.0, eps));
}

TEST(GeometryOperator_Sphere, MissWhenDiscriminantNegative) {
    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::geometry::SphereGeometryOperator<double> op;
    op.center = &c;
    op.radius = &radius;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 2.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Circle, MissingParamsReturnsMiss) {
    constexpr atlas::geometry::CircleGeometryOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 1.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Circle, HitReturnsPlaneIntersectionInsideRadius) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> center(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> normal(0.0, 0.0, 1.0);
    constexpr double radius = 2.0;

    atlas::geometry::CircleGeometryOperator<double> op;
    op.center = &center;
    op.normal = &normal;
    op.radius = &radius;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.5, 0.5, 3.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto [is_intersecting, distance, point, hit_normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 3.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(0.5, 0.5, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(hit_normal,
                                      atlas::math::Vector<double, 3>(0.0, 0.0, 1.0),
                                      eps));
}

TEST(GeometryOperator_Circle, MissWhenPlaneIntersectionFallsOutsideRadius) {
    const atlas::math::Vector<double, 3> center(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> normal(0.0, 0.0, 1.0);
    constexpr double radius = 2.0;

    atlas::geometry::CircleGeometryOperator<double> op;
    op.center = &center;
    op.normal = &normal;
    op.radius = &radius;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(3.0, 0.0, 3.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Triangle, MissingParamsReturnsMiss) {
    constexpr atlas::geometry::TriangleGeometryOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Triangle, HitReturnsTPointAndUnitNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> c(0.0, 1.0, 0.0);

    atlas::geometry::TriangleGeometryOperator<double> op;
    op.a = &a;
    op.b = &b;
    op.c = &c;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.25, 0.25, 1.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto [is_intersecting, distance, point, normal] = op(r);

    EXPECT_TRUE(is_intersecting);
    EXPECT_TRUE(atlas::test::near(distance, 1.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(point,
                                      atlas::math::Vector<double, 3>(0.25, 0.25, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::near(normal.length(), 1.0, eps));

    EXPECT_TRUE(atlas::test::vec_near(normal,
                                      atlas::math::Vector<double, 3>(0.0, 0.0, 1.0),
                                      eps));
}

TEST(GeometryOperator_Triangle, MissWhenRayHitsPlaneOutsideTriangle) {
    const atlas::math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> c(0.0, 1.0, 0.0);

    atlas::geometry::TriangleGeometryOperator<double> op;
    op.a = &a;
    op.b = &b;
    op.c = &c;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(2.0, 2.0, 1.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Dispatch, DefaultIsSphereAndReturnsMissWithNoParams) {
    const atlas::GeometryOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(GeometryOperator_Dispatch, TaggedConstructorsSelectCorrectVariant) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::geometry::SphereGeometryOperator<double> sphere;
    sphere.center = &c;
    sphere.radius = &radius;

    const atlas::GeometryOperator<double> op_s(sphere);

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op_s.trace(r);

    EXPECT_TRUE(h.is_intersecting);
    EXPECT_TRUE(atlas::test::near(h.distance, 2.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(h.point,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
}

TEST(GeometryOperator_Dispatch, CircleTaggedConstructorDispatchesTraceCorrectly) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> center(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> normal(0.0, 0.0, 1.0);
    constexpr double radius = 2.0;

    atlas::geometry::CircleGeometryOperator<double> circle;
    circle.center = &center;
    circle.normal = &normal;
    circle.radius = &radius;

    const atlas::GeometryOperator<double> op(circle);
    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(1.0, 0.0, 2.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op.trace(r);

    EXPECT_TRUE(h.is_intersecting);
    EXPECT_TRUE(atlas::test::near(h.distance, 2.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(h.point,
                                      atlas::math::Vector<double, 3>(1.0, 0.0, 0.0),
                                      eps));
    EXPECT_TRUE(atlas::test::vec_near(h.normal,
                                      atlas::math::Vector<double, 3>(0.0, 0.0, 1.0),
                                      eps));
}