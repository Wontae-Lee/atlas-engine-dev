#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Plane, DefaultConstructorSetsCanonicalParams) {

    // Default-constructed Plane<T> should follow the canonical convention:
    // - normal = (0,0,1)  (Z-up plane)
    // - offset = 0        (plane passes through the origin)
    //
    // Under the typical implicit plane form:
    //   n · x + d = 0
    // this corresponds to:
    //   (0,0,1) · (x,y,z) + 0 = 0  =>  z = 0
    const geometry::Plane<double> p;

    EXPECT_TRUE(test::vec_near(p.normal, Vector3<double>(0.0, 0.0, 1.0), eps));
    EXPECT_NEAR(p.offset, 0.0, eps);
}

TEST(Plane, NormalOffsetConstructorCopiesMemberParams) {

    // Provide a non-unit, non-trivial normal and a non-zero offset to ensure the
    // constructor stores values verbatim (no implicit normalization assumed here).
    const Vector3<double> n(0.0, 2.0, 0.0);
    constexpr double d = -3.5;

    const geometry::Plane<double> p(n, d);

    // The plane should store exactly the provided normal and offset.
    EXPECT_TRUE(test::vec_near(p.normal, n, eps));
    EXPECT_NEAR(p.offset, d, eps);
}

TEST(Plane, PointNormalConstructorSetsOffsetAsNegativeDot) {

    // A common plane construction is from a point on the plane and a normal:
    //   n · (x - p0) = 0
    // Expanding to implicit form n · x + d = 0 gives:
    //   d = -n · p0
    const Vector3<double> point(0.0, 0.0, 2.0);
    const Vector3<double> n(0.0, 0.0, 3.0);

    const geometry::Plane<double> p(point, n);

    // Normal should match the provided normal.
    EXPECT_TRUE(test::vec_near(p.normal, n, eps));

    // Offset must be -dot(n, point).
    EXPECT_NEAR(p.offset, -(n.dot(point)), eps);
}

TEST(Plane, NormalMemberIsWritable) {

    // Plane parameters are expected to be directly writable (public or mutable API).
    // This test ensures assigning normal updates the stored normal.
    geometry::Plane<double> p;

    const Vector3<double> n(1.0, 0.0, 0.0);
    p.normal = n;

    EXPECT_TRUE(test::vec_near(p.normal, n, eps));
}

TEST(Plane, OffsetMemberIsWritable) {

    // Offset must be writable to allow updating plane position along its normal.
    geometry::Plane<double> p;

    p.offset = 4.25;

    EXPECT_NEAR(p.offset, 4.25, eps);
}

TEST(Plane, MakeTraceOperatorReturnsPlaneTraceOperatorVariant) {

    // make_trace_operator() must return a TraceOperator tagged as Plane for runtime dispatch.
    const geometry::Plane<double> p;

    const auto op = p.make_trace_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Plane);
}

TEST(Plane, MakeQueryOperatorReturnsPlaneQueryOperatorVariant) {

    // make_query_operator() must return a QueryOperator tagged as Plane for runtime dispatch.
    const geometry::Plane<double> p;

    const auto op = p.make_query_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Plane);
}

TEST(Plane, ClosestPointProjectsToPlaneForZPlane) {

    // Use z=0 plane: n=(0,0,1), d=0.
    // The closest point should be the orthogonal projection onto the plane:
    // - x/y unchanged
    // - z becomes 0
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> q(1.25, -3.5, 7.0);
    const Vector3<double> expected(1.25, -3.5, 0.0);

    EXPECT_TRUE(test::vec_near(p.closest_point(q), expected, eps));
}

TEST(Plane, ClosestNormalIsParallelToStoredNormalForZPlane) {

    // For a plane, the closest normal is typically the plane's (unit) normal direction
    // (up to sign, depending on convention). Here we check it is parallel to +Z.
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> q(3.0, 4.0, 5.0);
    const Vector3<double> n = p.closest_normal(q);

    // Check parallel to Z by verifying x and y are (approximately) zero and |z| is 1.
    // We use abs() to allow either +Z or -Z as long as the direction is parallel.
    const double nx = std::abs(n.x);
    const double ny = std::abs(n.y);
    const double nz = std::abs(n.z);

    EXPECT_NEAR(nx, 0.0, eps);
    EXPECT_NEAR(ny, 0.0, eps);
    EXPECT_NEAR(nz, 1.0, eps);
}

TEST(Plane, SignedDistanceHasCorrectSignForZPlane) {

    // For z=0 plane with normal +Z and offset 0, signed distance should follow:
    // - positive when z > 0
    // - negative when z < 0
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> above(0.0, 0.0, 2.0);
    const Vector3<double> below(0.0, 0.0, -2.0);

    EXPECT_GT(p.signed_distance(above), 0.0);
    EXPECT_LT(p.signed_distance(below), 0.0);
}

TEST(Plane, IsInsideUsesHalfSpaceAndTolerance) {
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    EXPECT_TRUE(p.is_inside(Vector3<double>(0.0, 0.0, -1.0), 0.0));
    EXPECT_FALSE(p.is_inside(Vector3<double>(0.0, 0.0, 0.2), 0.0));
    EXPECT_TRUE(p.is_inside(Vector3<double>(0.0, 0.0, 0.2), 0.25));
}

TEST(Plane, IsOnSurfaceDetectsPlaneBand) {
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    EXPECT_TRUE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.0), 0.0));
    EXPECT_FALSE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.5), 0.0));
    EXPECT_TRUE(p.is_on_surface(Vector3<double>(1.0, 2.0, 0.1), 0.15));
}

TEST(Plane, CentroidLiesOnPlaneForZPlane) {

    // Planes are infinite primitives, so "centroid" is convention-based.
    // Whatever convention is used, it must lie on the plane (satisfy n·c + d = 0).
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 1.0), 0.0);

    const Vector3<double> c = p.centroid();

    // Validate c lies on plane: n·c + offset == 0.
    // Use a tight tolerance because this should be exact for z=0 plane in most implementations.
    EXPECT_NEAR(p.normal.dot(c) + p.offset, 0.0, 1e-10);
}

TEST(Plane, BoundIsWellFormedForInfinitePrimitive) {

    // A plane is infinite, so bound() cannot be a true tight AABB.
    // The library likely returns a large-but-finite bounding box (or a sentinel AABB)
    // to keep downstream code robust.
    const geometry::Plane<double> p;

    const auto aabb = p.bound();

    // Whatever convention is used, the AABB must be well-formed and finite so it can
    // participate safely in broad-phase structures and sanity checks.
    EXPECT_TRUE(test::is_finite_vec(aabb.lower_corner));
    EXPECT_TRUE(test::is_finite_vec(aabb.upper_corner));

    // Ensure bounds are ordered (lower <= upper) on each axis.
    EXPECT_LE(aabb.lower_corner.x, aabb.upper_corner.x);
    EXPECT_LE(aabb.lower_corner.y, aabb.upper_corner.y);
    EXPECT_LE(aabb.lower_corner.z, aabb.upper_corner.z);
}

TEST(Plane, IsValidTrueForNonZeroFiniteNormalAndFiniteOffset) {

    // Validity policy is typically:
    // - normal must be finite and non-zero (direction defined)
    // - offset must be finite
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 2.0), 1.0);

    EXPECT_TRUE(p.is_valid());
}

TEST(Plane, IsValidFalseForZeroNormal) {

    // A zero normal does not define a plane orientation, so it must be invalid.
    const geometry::Plane<double> p(Vector3<double>(0.0, 0.0, 0.0), 0.0);

    EXPECT_FALSE(p.is_valid());
}

TEST(Plane, TypeReturnsPlaneGeometryType) {

    // type() returns the runtime tag used for geometry dispatch/serialization.
    const geometry::Plane<double> p;

    EXPECT_EQ(p.type(), geometry::GeometryType::Plane);
}
