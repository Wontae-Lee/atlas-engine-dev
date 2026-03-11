#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <cmath> // std::abs (implicitly used by some geometry codepaths)

using namespace atlas;

TEST(Triangle, ConstructorSetsVerticesA) {

    // Build a simple right triangle on the XY plane.
    // This test focuses only on verifying that the constructor stores vertex "a".
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    // Construct triangle from three vertices (a,b,c).
    const geometry::Triangle<double> t(a, b, c);

    // Vertex storage must preserve the exact input (within eps).
    EXPECT_TRUE(test::vec_near(t.a, a, eps));
}

TEST(Triangle, ConstructorSetsVerticesB) {

    // Same setup as above, but verify vertex "b" is stored correctly.
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    EXPECT_TRUE(test::vec_near(t.b, b, eps));
}

TEST(Triangle, ConstructorSetsVerticesC) {

    // Same setup as above, but verify vertex "c" is stored correctly.
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    EXPECT_TRUE(test::vec_near(t.c, c, eps));
}

TEST(Triangle, ConstructorComputesCachedNormal) {

    // For a right triangle on the XY plane with CCW winding (a->b->c),
    // the normal should be +Z.
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(1.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 1.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    // Expected unit normal for CCW triangle in XY plane.
    const Vector3<double> expected(0.0, 0.0, 1.0);

    // Verify cached normal direction and that it is normalized.
    EXPECT_TRUE(test::vec_near(t.normal, expected, eps));
    EXPECT_NEAR(t.normal.length(), 1.0, 1e-12);
}

TEST(Triangle, MakeTraceOperatorProducesValidTraceOperator) {

    // Trace operator is used for ray intersection / tracing.
    // This test mainly checks the call is safe and that the geometry type is Triangle.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto op = t.make_trace_operator();

    // type() must return Triangle for runtime dispatch.
    EXPECT_EQ(t.type(), geometry::GeometryType::Triangle);

    // Suppress unused warning: the returned operator object is still constructed.
    (void)op;
}

TEST(Triangle, MakeQueryOperatorProducesValidQueryOperator) {

    // Query operator is used for closest point / distance / bounds queries.
    // Similar to trace operator test: ensure type is Triangle and the call is safe.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const auto op = t.make_query_operator();

    EXPECT_EQ(t.type(), geometry::GeometryType::Triangle);
    (void)op;
}

TEST(Triangle, ClosestPointMatchesTriangleQueryOperator) {

    // Ensure Triangle::closest_point() delegates to TriangleQueryOperator consistently.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    // Point above the triangle (in +Z). Closest point should lie on triangle plane.
    const Vector3<double> p(0.25, 0.25, 2.0);

    // Build a query operator that points at the triangle's internal storage.
    // raw_pointer_cast is used to provide device/host-safe pointer representations.
    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    // "Expected" computed through operator; "got" computed through Triangle wrapper.
    const auto expected = qop.closest_point(p);
    const auto got      = t.closest_point(p);

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, ClosestNormalMatchesTriangleQueryOperator) {

    // Ensure closest_normal() matches the query operator implementation.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> p(0.25, 0.25, 2.0);

    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.closest_normal(p);
    const auto got      = t.closest_normal(p);

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, SignedDistanceMatchesTriangleQueryOperator) {

    // Ensure signed_distance() matches the query operator implementation.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    const Vector3<double> p(0.25, 0.25, 2.0);

    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const double expected = qop.signed_distance(p);
    const double got      = t.signed_distance(p);

    EXPECT_NEAR(got, expected, eps);
}

TEST(Triangle, CentroidMatchesTriangleQueryOperator) {

    // Use a non-unit triangle to ensure centroid computation isn't accidentally
    // hardcoded to specific positions.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(0.0, 6.0, 0.0));

    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    // Both paths should produce the same centroid.
    const auto expected = qop.centroid();
    const auto got      = t.centroid();

    EXPECT_TRUE(test::vec_near(got, expected, eps));
}

TEST(Triangle, BoundMatchesTriangleQueryOperator) {

    // Use coordinates with mixed signs to ensure min/max logic on each axis is exercised.
    const geometry::Triangle<double> t(
        Vector3<double>(-2.0, 3.0, 1.0),
        Vector3<double>(5.0, -4.0, 2.0),
        Vector3<double>(1.0, 2.0, -6.0));

    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    const auto expected = qop.bound();
    const auto got      = t.bound();

    // Compare both corners of the AABB.
    EXPECT_TRUE(test::vec_near(got.lower_corner, expected.lower_corner, eps));
    EXPECT_TRUE(test::vec_near(got.upper_corner, expected.upper_corner, eps));
}

TEST(Triangle, IsValidMatchesTriangleQueryOperator) {

    // Validity checks should be consistent between Triangle wrapper and query operator.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    geometry::TriangleQueryOperator<double> qop;
    qop.a = atlas::raw_pointer_cast(&t.a);
    qop.b = atlas::raw_pointer_cast(&t.b);
    qop.c = atlas::raw_pointer_cast(&t.c);
    qop.n = atlas::raw_pointer_cast(&t.normal);

    EXPECT_EQ(t.is_valid(), qop.is_valid());
}

TEST(Triangle, TypeReturnsTriangle) {

    // type() returns a runtime geometry tag used for dispatch/serialization.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    EXPECT_EQ(t.type(), geometry::GeometryType::Triangle);
}

TEST(Triangle, SetVerticesUpdatesVertexA) {

    // set_vertices() should update all stored vertices.
    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    // New vertex set (not coplanar with the original to avoid accidental reuse).
    const Vector3<double> a2(2.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 2.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 2.0);

    t.set_vertices(a2, b2, c2);

    EXPECT_TRUE(test::vec_near(t.a, a2, eps));
}

TEST(Triangle, SetVerticesUpdatesVertexB) {

    // Verify vertex "b" is updated by set_vertices().
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

    // Verify vertex "c" is updated by set_vertices().
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

    // Changing vertices must recompute cached normal to remain consistent.
    geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0));

    // New triangle in the YZ plane (still non-degenerate) so normal changes.
    const Vector3<double> a2(0.0, 0.0, 0.0);
    const Vector3<double> b2(0.0, 1.0, 0.0);
    const Vector3<double> c2(0.0, 0.0, 1.0);

    t.set_vertices(a2, b2, c2);

    // Expected normal computed from right-handed cross product of edges.
    // Note: The sign depends on vertex winding convention (a->b->c).
    const Vector3<double> expected = math::cross(b2 - a2, c2 - a2).normalized();

    EXPECT_TRUE(test::vec_near(t.normal, expected, eps));
}

TEST(Triangle, BarycentricReturnsFalseForDegenerateTriangle) {

    // Degenerate triangle (collinear points) has zero area and cannot produce stable
    // barycentric coordinates via area-based formulas.
    const geometry::Triangle<double> t(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(2.0, 0.0, 0.0));

    double u = -1.0;
    double v = -1.0;
    double w = -1.0;

    // Query a point on the same line. Implementation policy:
    // - return false to signal degeneracy
    // - set (u,v,w) to a safe fallback (here: (1,0,0)) so callers have deterministic values.
    const bool ok = t.barycentric(Vector3<double>(0.5, 0.0, 0.0), u, v, w);

    EXPECT_FALSE(ok);

    // Expect the documented/assumed fallback weights.
    EXPECT_NEAR(u, 1.0, eps);
    EXPECT_NEAR(v, 0.0, eps);
    EXPECT_NEAR(w, 0.0, eps);
}

TEST(Triangle, BarycentricComputesWeightsForInteriorPoint) {

    // Use a large right triangle on the XY plane to make barycentric weights easy to interpret.
    const Vector3<double> a(0.0, 0.0, 0.0);
    const Vector3<double> b(2.0, 0.0, 0.0);
    const Vector3<double> c(0.0, 2.0, 0.0);

    const geometry::Triangle<double> t(a, b, c);

    double u = 0.0;
    double v = 0.0;
    double w = 0.0;

    // Choose a point strictly inside the triangle.
    const Vector3<double> p(0.5, 0.5, 0.0);

    // barycentric() should succeed for non-degenerate triangles.
    const bool ok = t.barycentric(p, u, v, w);

    EXPECT_TRUE(ok);

    // Barycentric coordinates must sum to 1.
    EXPECT_NEAR(u + v + w, 1.0, 1e-12);

    // Reconstruct point from barycentric weights and ensure it matches.
    const Vector3<double> recon = a * u + b * v + c * w;
    EXPECT_TRUE(test::vec_near(recon, p, 1e-12));
}