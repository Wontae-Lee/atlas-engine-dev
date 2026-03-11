#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <cmath>

#include <gtest/gtest.h>

TEST(TraceOperator_Box, MissingParamsReturnsMiss) {
    const atlas::math::Vector<double, 3> o(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> d(1.0, 0.0, 0.0);
    const atlas::Ray<double> r(o, d);

    constexpr atlas::spatial::BoxTraceOperator<double> op;
    const auto h = op(r);

    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Box, HitFromOutsideReturnsEnterDistancePointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::spatial::BoxTraceOperator<double> op;
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

TEST(TraceOperator_Box, FromInsideReturnsExitDistanceAndOutwardNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::spatial::BoxTraceOperator<double> op;
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

TEST(TraceOperator_Box, MissWhenIntervalsDoNotOverlap) {
    const atlas::math::Vector<double, 3> lo(-1.0, -1.0, -1.0);
    const atlas::math::Vector<double, 3> hi(1.0, 1.0, 1.0);

    atlas::spatial::BoxTraceOperator<double> op;
    op.lower_corner = &lo;
    op.upper_corner = &hi;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 2.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Cylinder, MissingParamsReturnsMiss) {
    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    constexpr atlas::spatial::CylinderTraceOperator<double> op;
    const auto h = op(r);

    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Cylinder, SideHitReturnsCorrectDistancePointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;
    constexpr double height = 2.0;

    atlas::spatial::CylinderTraceOperator<double> op;
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

TEST(TraceOperator_Cylinder, CapHitReturnsNormalAlongZ) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;
    constexpr double height = 2.0;

    atlas::spatial::CylinderTraceOperator<double> op;
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

TEST(TraceOperator_Plane, MissingParamsReturnsMiss) {
    constexpr atlas::spatial::PlaneTraceOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(0.0, 1.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Plane, ParallelNoHitWhenNotOnPlane) {
    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::spatial::PlaneTraceOperator<double> op;
    op.normal = &n;
    op.offset = &d;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Plane, ParallelOnPlaneReturnsTZero) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::spatial::PlaneTraceOperator<double> op;
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

TEST(TraceOperator_Plane, HitComputesCorrectTPointAndNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> n(0.0, 1.0, 0.0);
    constexpr double d = -1.0;

    atlas::spatial::PlaneTraceOperator<double> op;
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

TEST(TraceOperator_Sphere, MissingParamsReturnsMiss) {
    constexpr atlas::spatial::SphereTraceOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Sphere, HitReturnsNearestPositiveRootAndUnitNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::spatial::SphereTraceOperator<double> op;
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

TEST(TraceOperator_Sphere, MissWhenDiscriminantNegative) {
    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::spatial::SphereTraceOperator<double> op;
    op.center = &c;
    op.radius = &radius;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 2.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Triangle, MissingParamsReturnsMiss) {
    constexpr atlas::spatial::TriangleTraceOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, 1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Triangle, HitReturnsTPointAndUnitNormal) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> c(0.0, 1.0, 0.0);

    atlas::spatial::TriangleTraceOperator<double> op;
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

TEST(TraceOperator_Triangle, MissWhenRayHitsPlaneOutsideTriangle) {
    const atlas::math::Vector<double, 3> a(0.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> b(1.0, 0.0, 0.0);
    const atlas::math::Vector<double, 3> c(0.0, 1.0, 0.0);

    atlas::spatial::TriangleTraceOperator<double> op;
    op.a = &a;
    op.b = &b;
    op.c = &c;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(2.0, 2.0, 1.0),
                               atlas::math::Vector<double, 3>(0.0, 0.0, -1.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Dispatch, DefaultIsSphereAndReturnsMissWithNoParams) {
    const atlas::spatial::TraceOperator<double> op;

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(0.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op(r);
    EXPECT_FALSE(h.is_intersecting);
}

TEST(TraceOperator_Dispatch, TaggedConstructorsSelectCorrectVariant) {
    constexpr auto eps = static_cast<double>(atlas::eps);

    const atlas::math::Vector<double, 3> c(0.0, 0.0, 0.0);
    constexpr double radius = 1.0;

    atlas::spatial::SphereTraceOperator<double> sphere;
    sphere.center = &c;
    sphere.radius = &radius;

    const atlas::spatial::TraceOperator<double> op_s(sphere);

    const atlas::Ray<double> r(atlas::math::Vector<double, 3>(-3.0, 0.0, 0.0),
                               atlas::math::Vector<double, 3>(1.0, 0.0, 0.0));

    const auto h = op_s.trace(r);

    EXPECT_TRUE(h.is_intersecting);
    EXPECT_TRUE(atlas::test::near(h.distance, 2.0, eps));
    EXPECT_TRUE(atlas::test::vec_near(h.point,
                                      atlas::math::Vector<double, 3>(-1.0, 0.0, 0.0),
                                      eps));
}