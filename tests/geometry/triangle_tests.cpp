#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath>

using namespace atlas;

TEST(Triangle, ConstructorSetsVerticesA) {

    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    EXPECT_TRUE(test::vec_near(t.a, a, eps));
}

TEST(Triangle, ConstructorSetsVerticesB) {

    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    EXPECT_TRUE(test::vec_near(t.b, b, eps));
}

TEST(Triangle, ConstructorSetsVerticesC) {

    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    EXPECT_TRUE(test::vec_near(t.c, c, eps));
}

TEST(Triangle, ConstructorComputesCachedNormal) {

    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    const Vector3<double> expected(0.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(t.normal, expected, eps));
    EXPECT_NEAR(t.normal.length(), 1.0, 1e-12);
}

TEST(Triangle, MakeGeometryOperatorProducesValidGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto op = t.make_geometry_operator();

    EXPECT_EQ(t.type(), geometry::GeometryType::Triangle);

    (void)op;
}

TEST(Triangle, ClosestPointMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> p(0.25, 0.25, 2.0);

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.closest_point(p);
    const auto got      = t.closest_point(p);

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, ClosestNormalMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> p(0.25, 0.25, 2.0);

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.closest_normal(p);
    const auto got      = t.closest_normal(p);

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, SignedDistanceMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> p(0.25, 0.25, 2.0);

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const double expected = qop.signed_distance(p);
    const double got      = t.signed_distance(p);

    EXPECT_NEAR(got, expected, eps);
}

TEST(Triangle, IsInsideMatchesGeometryOperatorClassification) {
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto qop = t.make_geometry_operator();

    EXPECT_EQ(t.is_inside(Vector3<double>(0.25, 0.25, -0.1), 0.0),
              qop.is_inside(Vector3<double>(0.25, 0.25, -0.1), 0.0));
    EXPECT_EQ(t.is_inside(Vector3<double>(0.25, 0.25, 0.1), 0.15),
              qop.is_inside(Vector3<double>(0.25, 0.25, 0.1), 0.15));
}

TEST(Triangle, IsOnSurfaceMatchesGeometryOperatorClassification) {
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto qop = t.make_geometry_operator();

    EXPECT_EQ(t.is_on_surface(Vector3<double>(0.25, 0.25, 0.0), 0.0),
              qop.is_on_surface(Vector3<double>(0.25, 0.25, 0.0), 0.0));
    EXPECT_EQ(t.is_on_surface(Vector3<double>(0.25, 0.25, 0.1), 0.15),
              qop.is_on_surface(Vector3<double>(0.25, 0.25, 0.1), 0.15));
}

TEST(Triangle, CentroidMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 6.0, 0.0));

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.centroid();
    const auto got      = t.centroid();

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, BoundMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(-2.0, 3.0, 1.0),
        Vector3<double>(5.0, -4.0, 2.0),
        Vector3<double>(1.0, 2.0, -6.0));

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.bound();
    const auto got      = t.bound();

    EXPECT_TRUE(test::vec_near(got.lower_corner, expected.lower_corner, eps));
    EXPECT_TRUE(test::vec_near(got.upper_corner, expected.upper_corner, eps));
}

TEST(Triangle, IsValidMatchesTriangleGeometryOperator) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    geometry::TriangleGeometryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    EXPECT_EQ(t.is_valid(), qop.is_valid());
}

TEST(Triangle, TypeReturnsTriangle) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    EXPECT_EQ(t.type(), geometry::GeometryType::Triangle);
}

TEST(Triangle, SetVerticesUpdatesVertexA) {

    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> a2(2.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 2.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 2.0);

    t.set_vertices(a2, b2, c2);

    EXPECT_TRUE(test::vec_near(t.a, a2, eps));
}

TEST(Triangle, SetVerticesUpdatesVertexB) {

    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> a2(2.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 2.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 2.0);

    t.set_vertices(a2, b2, c2);

    EXPECT_TRUE(test::vec_near(t.b, b2, eps));
}

TEST(Triangle, SetVerticesUpdatesVertexC) {

    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> a2(2.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 2.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 2.0);

    t.set_vertices(a2, b2, c2);

    EXPECT_TRUE(test::vec_near(t.c, c2, eps));
}

TEST(Triangle, SetVerticesRecomputesCachedNormal) {

    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> a2(0.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 1.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 1.0);

    t.set_vertices(a2, b2, c2);

    const Vector3<double> expected = math::cross(b2 - a2, c2 - a2).normalized();

    EXPECT_TRUE(test::vec_near(t.normal, expected, eps));
}

TEST(Triangle, BarycentricReturnsFalseForDegenerateTriangle) {

    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0));

    double u = -1.0;
    double v = -1.0;
    double w = -1.0;

    const bool ok = t.barycentric(Vector3<double>(0.5, 0.0, 0.0), u, v, w);

    EXPECT_FALSE(ok);

    EXPECT_NEAR(u, 1.0, eps);
    EXPECT_NEAR(v, 0.0, eps);
    EXPECT_NEAR(w, 0.0, eps);
}

TEST(Triangle, BarycentricComputesWeightsForInteriorPoint) {

    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(2.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 2.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    double u = 0.0;
    double v = 0.0;
    double w = 0.0;

    const Vector3<double> p(0.5, 0.5, 0.0);

    const bool ok = t.barycentric(p, u, v, w);

    EXPECT_TRUE(ok);

    EXPECT_NEAR(u + v + w, 1.0, 1e-12);

    const Vector3<double> recon = a * u + b * v + c * w;
    EXPECT_TRUE(test::vec_near(recon, p, 1e-12));
}