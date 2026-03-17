#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Box, DefaultConstructorSetsCanonicalBounds) {

    // Default-constructed Box<T> should use canonical bounds by convention:
    // - lower_corner = (-1,-1,-1)
    // - upper_corner = (+1,+1,+1)
    // This provides a convenient unit/canonical box for many algorithms and tests.
    const geometry::Box<double> b;

    // Verify both corners match the canonical AABB bounds.
    EXPECT_TRUE(test::vec_near(b.lower_corner, Vector3<double>(-1.0, -1.0, -1.0), eps));
    EXPECT_TRUE(test::vec_near(b.upper_corner, Vector3<double>(+1.0, +1.0, +1.0), eps));
}

TEST(Box, BoundsConstructorCopiesMemberBounds) {

    // Provide explicit lower/upper bounds with non-trivial values to ensure they
    // are copied exactly into the Box members.
    const Vector3<double> lo(-2.0, -3.0, -4.0);
    const Vector3<double> hi(+5.0, +6.0, +7.0);

    // Construct a box from explicit bounds.
    const geometry::Box<double> b(lo, hi);

    // The constructor must store the given vectors verbatim.
    EXPECT_TRUE(test::vec_near(b.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(b.upper_corner, hi, eps));
}

TEST(Box, LowerCornerMemberIsWritable) {

    // Box corners are stored as public members (or otherwise directly writable).
    // This test ensures modifying lower_corner updates the object state.
    geometry::Box<double> b;

    // Set a new lower corner and verify the assignment sticks.
    const Vector3<double> lo(-9.0, -8.0, -7.0);
    b.lower_corner = lo;

    EXPECT_TRUE(test::vec_near(b.lower_corner, lo, eps));
}

TEST(Box, UpperCornerMemberIsWritable) {

    // Similar to LowerCornerMemberIsWritable, but for upper_corner.
    geometry::Box<double> b;

    const Vector3<double> hi(9.0, 8.0, 7.0);
    b.upper_corner = hi;

    EXPECT_TRUE(test::vec_near(b.upper_corner, hi, eps));
}

TEST(Box, MakeTraceOperatorReturnsBoxTraceOperatorVariant) {

    // make_trace_operator() must return a TraceOperator variant tagged as Box.
    // This tag is used for runtime dispatch across geometry types.
    const geometry::Box<double> b;

    const auto op = b.make_trace_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Box);
}

TEST(Box, MakeQueryOperatorReturnsBoxQueryOperatorVariant) {

    // make_query_operator() must return a QueryOperator variant tagged as Box.
    // This tag is used for runtime dispatch across query operator implementations.
    const geometry::Box<double> b;

    const auto op = b.make_query_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Box);
}

TEST(Box, ClosestPointDelegatesToQueryOperator) {

    // Use a symmetric box centered at origin with bounds [-1, +1] on each axis.
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    // Choose a point that is outside the box in +X and -Z, but inside in Y.
    // The closest point should clamp X to +1 and Z to -1, while keeping Y unchanged.
    const Vector3<double> p(2.0, 0.5, -3.0);
    const Vector3<double> expected(1.0, 0.5, -1.0);

    // Box::closest_point() is expected to delegate to the underlying query operator.
    EXPECT_TRUE(test::vec_near(b.closest_point(p), expected, eps));
}

TEST(Box, ClosestNormalReturnsAxisUnitNormalForOutsidePoint) {

    // For an axis-aligned box, the closest normal for an outside point should be
    // one of the axis-aligned unit normals (+/-X, +/-Y, +/-Z), depending on which
    // face is closest.
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    // This point lies outside on +X side, while Y/Z remain inside the slab.
    // The closest face is the +X face, so the normal should be (1,0,0).
    const Vector3<double> p(2.0, 0.25, 0.0);

    const Vector3<double> n = b.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(Box, SignedDistanceIsPositiveOutsideAndNegativeInside) {

    // Signed distance convention for SDFs is usually:
    // - positive outside
    // - negative inside
    // - zero on the surface
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(+1.0, +1.0, +1.0));

    // Clearly outside on +X by a margin of 2 (x=3 vs upper x=1).
    const Vector3<double> p_out(3.0, 0.0, 0.0);

    // Clearly inside (center of the box).
    const Vector3<double> p_in(0.0, 0.0, 0.0);

    // Outside distance must be positive.
    EXPECT_GT(b.signed_distance(p_out), 0.0);

    // Inside distance must be negative.
    EXPECT_LT(b.signed_distance(p_in), 0.0);
}

TEST(Box, IsInsideClassifiesInteriorAndToleranceBand) {
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0));

    EXPECT_TRUE(b.is_inside(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_FALSE(b.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.0));
    EXPECT_TRUE(b.is_inside(Vector3<double>(1.2, 0.0, 0.0), 0.25));
}

TEST(Box, IsOnSurfaceDetectsBoundaryWithTolerance) {
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0));

    EXPECT_TRUE(b.is_on_surface(Vector3<double>(1.0, 0.25, 0.0), 0.0));
    EXPECT_FALSE(b.is_on_surface(Vector3<double>(0.0, 0.0, 0.0), 0.0));
    EXPECT_TRUE(b.is_on_surface(Vector3<double>(1.1, 0.0, 0.0), 0.15));
}

TEST(Box, CentroidIsMidpointOfBounds) {

    // The centroid of an axis-aligned box is the midpoint of its lower/upper corners:
    // centroid = (lo + hi) * 0.5
    const geometry::Box<double> b(
        Vector3<double>(-2.0, -4.0, -6.0),
        Vector3<double>(+6.0, +2.0, +4.0));

    // Midpoint:
    // x: (-2 + 6)/2 =  2
    // y: (-4 + 2)/2 = -1
    // z: (-6 + 4)/2 = -1
    const Vector3<double> expected(2.0, -1.0, -1.0);

    EXPECT_TRUE(test::vec_near(b.centroid(), expected, eps));
}

TEST(Box, BoundReturnsAxisAlignedBoundingBoxWithSameBounds) {

    // bound() is expected to return an AxisAlignedBoundingBox with the same lower/upper corners.
    const Vector3<double> lo(-2.0, -3.0, -4.0);
    const Vector3<double> hi(+5.0, +6.0, +7.0);

    const geometry::Box<double> b(lo, hi);

    const auto aabb = b.bound();

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(Box, IsValidTrueForOrderedBounds) {

    // A valid box should have ordered bounds on every axis:
    // lower_corner[i] <= upper_corner[i] for i in {x,y,z}.
    const geometry::Box<double> b(
        Vector3<double>(-1.0, -2.0, -3.0),
        Vector3<double>(+1.0, +2.0, +3.0));

    EXPECT_TRUE(b.is_valid());
}

TEST(Box, IsValidFalseForInvertedBounds) {

    // Inverted bounds on any axis should render the box invalid.
    // Here x is inverted: lower.x = +1, upper.x = -1.
    const geometry::Box<double> b(
        Vector3<double>(+1.0, 0.0, 0.0),
        Vector3<double>(-1.0, 0.0, 0.0));

    EXPECT_FALSE(b.is_valid());
}

TEST(Box, TypeReturnsBoxGeometryType) {

    // type() provides a runtime tag to identify geometry type in dispatch.
    const geometry::Box<double> b;

    EXPECT_EQ(b.type(), geometry::GeometryType::Box);
}
