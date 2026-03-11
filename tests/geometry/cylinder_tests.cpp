#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Cylinder, DefaultConstructorSetsCanonicalParams) {

    // Default-constructed Cylinder<T> should use canonical parameters by convention:
    // - center = (0,0,0)
    // - radius = 1
    // - height = 1
    //
    // This provides a convenient unit cylinder for algorithms and smoke tests.
    const geometry::Cylinder<double> c;

    // Verify canonical defaults.
    EXPECT_TRUE(test::vec_near(c.center, Vector3<double>(0.0, 0.0, 0.0), eps));
    EXPECT_NEAR(c.radius, 1.0, eps);
    EXPECT_NEAR(c.height, 1.0, eps);
}

TEST(Cylinder, ParamsConstructorCopiesMemberParams) {

    // Provide explicit parameters with mixed signs / non-trivial magnitudes so
    // we can detect accidental reorderings or overwrites.
    const Vector3<double> center(1.0, -2.0, 3.0);
    constexpr double radius = 2.5;
    constexpr double height = 7.0;

    // Constructor should copy parameters into members verbatim.
    const geometry::Cylinder<double> c(center, radius, height);

    EXPECT_TRUE(test::vec_near(c.center, center, eps));
    EXPECT_NEAR(c.radius, radius, eps);
    EXPECT_NEAR(c.height, height, eps);
}

TEST(Cylinder, CenterMemberIsWritable) {

    // Members are expected to be writable (public or via direct access).
    // This test ensures assignments update object state.
    geometry::Cylinder<double> c;

    const Vector3<double> center(9.0, 8.0, 7.0);
    c.center = center;

    EXPECT_TRUE(test::vec_near(c.center, center, eps));
}

TEST(Cylinder, RadiusMemberIsWritable) {

    // Radius must be writable so users can modify geometry in-place.
    geometry::Cylinder<double> c;

    c.radius = 3.0;

    EXPECT_NEAR(c.radius, 3.0, eps);
}

TEST(Cylinder, HeightMemberIsWritable) {

    // Height must be writable so users can modify geometry in-place.
    geometry::Cylinder<double> c;

    c.height = 4.0;

    EXPECT_NEAR(c.height, 4.0, eps);
}

TEST(Cylinder, MakeTraceOperatorReturnsCylinderTraceOperatorVariant) {

    // make_trace_operator() must return a TraceOperator variant tagged as Cylinder.
    // This tag is used for runtime dispatch across geometry types.
    const geometry::Cylinder<double> c;

    const auto op = c.make_trace_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Cylinder);
}

TEST(Cylinder, MakeQueryOperatorReturnsCylinderQueryOperatorVariant) {

    // make_query_operator() must return a QueryOperator variant tagged as Cylinder.
    // This tag is used for runtime dispatch across query operator implementations.
    const geometry::Cylinder<double> c;

    const auto op = c.make_query_operator();

    EXPECT_EQ(op.type, geometry::GeometryType::Cylinder);
}

TEST(Cylinder, ClosestPointClampsToSideWallInXYAndCapsInZ) {

    // Use a simple cylinder aligned with the Z axis:
    // - center at origin
    // - radius = 1
    // - height = 2
    //
    // Under typical conventions, the cylinder spans:
    // - z in [-height/2, +height/2] => [-1, +1]
    // - radial distance in XY clamped to radius
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    // Point is outside radially (x=2) and above the top cap (z=5).
    const Vector3<double> p(2.0, 0.0, 5.0);

    // Expected closest point:
    // - clamp radial projection to radius => (1, 0) in XY
    // - clamp z to +height/2 => +1
    const Vector3<double> expected(1.0, 0.0, 1.0);

    EXPECT_TRUE(test::vec_near(c.closest_point(p), expected, eps));
}

TEST(Cylinder, ClosestNormalIsRadialForSideWallOutsidePoint) {

    // Same simple cylinder centered at origin.
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    // Point is outside on +X direction, with z inside the height range.
    // Closest surface should be the side wall (not a cap).
    const Vector3<double> p(2.0, 0.0, 0.25);

    // Closest normal for the side wall should be radial in the XY plane:
    // here, +X => (1,0,0).
    const Vector3<double> n = c.closest_normal(p);

    EXPECT_TRUE(test::vec_near(n, Vector3<double>(1.0, 0.0, 0.0), eps));
}

TEST(Cylinder, SignedDistanceIsPositiveOutsideAndNegativeInside) {

    // Signed distance convention for SDFs is usually:
    // - positive outside
    // - negative inside
    // - zero on the surface
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    // Clearly outside (radially).
    const Vector3<double> p_out(3.0, 0.0, 0.0);

    // Clearly inside (center).
    const Vector3<double> p_in(0.0, 0.0, 0.0);

    EXPECT_GT(c.signed_distance(p_out), 0.0);
    EXPECT_LT(c.signed_distance(p_in), 0.0);
}

TEST(Cylinder, CentroidEqualsCenter) {

    // For an axis-aligned cylinder parameterized by its center, the centroid
    // should equal that center (assuming symmetric height about center.z).
    const Vector3<double> center(1.0, -2.0, 3.0);
    const geometry::Cylinder<double> c(center, 2.0, 4.0);

    EXPECT_TRUE(test::vec_near(c.centroid(), center, eps));
}

TEST(Cylinder, BoundMatchesAxisAlignedExtents) {

    // AABB bounds for an axis-aligned cylinder centered at `center` are expected to be:
    // - x in [center.x - radius, center.x + radius]
    // - y in [center.y - radius, center.y + radius]
    // - z in [center.z - height/2, center.z + height/2]
    const Vector3<double> center(1.0, -2.0, 3.0);
    constexpr double radius = 2.0;
    constexpr double height = 6.0;

    const geometry::Cylinder<double> c(center, radius, height);

    const auto aabb = c.bound();

    // Compute expected corners based on the cylinder extents.
    const Vector3<double> lo(center.x - radius, center.y - radius, center.z - height * 0.5);
    const Vector3<double> hi(center.x + radius, center.y + radius, center.z + height * 0.5);

    EXPECT_TRUE(test::vec_near(aabb.lower_corner, lo, eps));
    EXPECT_TRUE(test::vec_near(aabb.upper_corner, hi, eps));
}

TEST(Cylinder, IsValidTrueForPositiveRadiusAndHeight) {

    // A cylinder is valid only if radius > 0 and height > 0 (and center is finite).
    const geometry::Cylinder<double> c(
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0);

    EXPECT_TRUE(c.is_valid());
}

TEST(Cylinder, IsValidFalseForNonPositiveRadiusOrHeight) {

    // Non-positive radius/height should invalidate the cylinder.
    // We test zero and negative cases for both parameters.
    const geometry::Cylinder<double> r0(Vector3<double>(0.0, 0.0, 0.0), 0.0, 2.0);
    const geometry::Cylinder<double> h0(Vector3<double>(0.0, 0.0, 0.0), 1.0, 0.0);
    const geometry::Cylinder<double> rn(Vector3<double>(0.0, 0.0, 0.0), -1.0, 2.0);
    const geometry::Cylinder<double> hn(Vector3<double>(0.0, 0.0, 0.0), 1.0, -2.0);

    EXPECT_FALSE(r0.is_valid());
    EXPECT_FALSE(h0.is_valid());
    EXPECT_FALSE(rn.is_valid());
    EXPECT_FALSE(hn.is_valid());
}

TEST(Cylinder, TypeReturnsCylinderGeometryType) {

    // type() provides a runtime geometry tag for dispatch/serialization.
    const geometry::Cylinder<double> c;

    EXPECT_EQ(c.type(), geometry::GeometryType::Cylinder);
}