#include <atlas/geometry/triangle.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Ray;
using atlas::Triangle;

/**
 * A canonical right triangle in the z = 0 plane, oriented so its derived normal
 * is +z. Vertices a=(0,0,0), b=(1,0,0), c=(0,1,0); area 0.5, centroid
 * (1/3,1/3,0).
 */
Triangle
reference_triangle() noexcept {
    return Triangle(Float3(0.0f, 0.0f, 0.0f),
                    Float3(1.0f, 0.0f, 0.0f),
                    Float3(0.0f, 1.0f, 0.0f));
}

constexpr float kTol = 1e-5f;

}

TEST(Triangle, DefaultConstructsDegenerateAtOrigin) {
    const Triangle tri;

    EXPECT_FLOAT_EQ(tri.a.x, 0.0f);
    EXPECT_FLOAT_EQ(tri.a.y, 0.0f);
    EXPECT_FLOAT_EQ(tri.a.z, 0.0f);
    EXPECT_TRUE(tri.a == tri.b);
    EXPECT_TRUE(tri.b == tri.c);

    // The default normal slot faces +z even though the triangle is degenerate.
    EXPECT_FLOAT_EQ(tri.normal.z, 1.0f);

    // Collinear (coincident) vertices: not a valid surface.
    EXPECT_FALSE(tri.is_valid());
}

TEST(Triangle, ThreeVertexConstructorDerivesNormal) {
    const Triangle tri = reference_triangle();

    EXPECT_NEAR(tri.normal.x, 0.0f, kTol);
    EXPECT_NEAR(tri.normal.y, 0.0f, kTol);
    EXPECT_NEAR(tri.normal.z, 1.0f, kTol);
}

TEST(Triangle, NormalIsUnitLength) {
    const Triangle tri(Float3(0.0f, 0.0f, 0.0f),
                       Float3(2.0f, 0.0f, 0.0f),
                       Float3(0.0f, 3.0f, 0.0f));

    EXPECT_NEAR(tri.normal.length(), 1.0f, kTol);
}

TEST(Triangle, NormalOrientationFollowsWinding) {
    const Triangle ccw(Float3(0.0f, 0.0f, 0.0f),
                       Float3(1.0f, 0.0f, 0.0f),
                       Float3(0.0f, 1.0f, 0.0f));

    // Swapping the last two vertices reverses the winding and flips the normal.
    const Triangle cw(Float3(0.0f, 0.0f, 0.0f),
                      Float3(0.0f, 1.0f, 0.0f),
                      Float3(1.0f, 0.0f, 0.0f));

    EXPECT_NEAR(ccw.normal.z, 1.0f, kTol);
    EXPECT_NEAR(cw.normal.z, -1.0f, kTol);
}

TEST(Triangle, IsValidTrueForNonDegenerate) {
    EXPECT_TRUE(reference_triangle().is_valid());
}

TEST(Triangle, IsValidFalseForCollinearVertices) {
    const Triangle collinear(Float3(0.0f, 0.0f, 0.0f),
                             Float3(1.0f, 0.0f, 0.0f),
                             Float3(2.0f, 0.0f, 0.0f));

    EXPECT_FALSE(collinear.is_valid());
}

TEST(Triangle, CentroidIsVertexAverage) {
    const Triangle tri = reference_triangle();
    const Float3 c     = tri.centroid();

    EXPECT_NEAR(c.x, 1.0f / 3.0f, kTol);
    EXPECT_NEAR(c.y, 1.0f / 3.0f, kTol);
    EXPECT_NEAR(c.z, 0.0f, kTol);
}

TEST(Triangle, BoundEnclosesVertices) {
    const Triangle tri(Float3(-1.0f, 2.0f, 0.5f),
                       Float3(3.0f, -4.0f, 1.0f),
                       Float3(0.0f, 0.0f, -2.0f));

    const AABB box = tri.bound();

    EXPECT_FLOAT_EQ(box.lower_corner.x, -1.0f);
    EXPECT_FLOAT_EQ(box.lower_corner.y, -4.0f);
    EXPECT_FLOAT_EQ(box.lower_corner.z, -2.0f);
    EXPECT_FLOAT_EQ(box.upper_corner.x, 3.0f);
    EXPECT_FLOAT_EQ(box.upper_corner.y, 2.0f);
    EXPECT_FLOAT_EQ(box.upper_corner.z, 1.0f);
}

TEST(Triangle, ClosestPointInteriorProjection) {
    const Triangle tri = reference_triangle();

    // (0.25,0.25,1) projects to (0.25,0.25,0), which lies inside the triangle.
    const Float3 cp = tri.closest_point(Float3(0.25f, 0.25f, 1.0f));

    EXPECT_NEAR(cp.x, 0.25f, kTol);
    EXPECT_NEAR(cp.y, 0.25f, kTol);
    EXPECT_NEAR(cp.z, 0.0f, kTol);
}

TEST(Triangle, ClosestPointOnEdge) {
    const Triangle tri = reference_triangle();

    // (0.5,-0.5,0) sits in the Voronoi region of edge ab; nearest point (0.5,0,0).
    const Float3 cp = tri.closest_point(Float3(0.5f, -0.5f, 0.0f));

    EXPECT_NEAR(cp.x, 0.5f, kTol);
    EXPECT_NEAR(cp.y, 0.0f, kTol);
    EXPECT_NEAR(cp.z, 0.0f, kTol);
}

TEST(Triangle, ClosestPointAtVertex) {
    const Triangle tri = reference_triangle();

    // (-1,-1,0) lies in the Voronoi region of vertex a=(0,0,0).
    const Float3 cp = tri.closest_point(Float3(-1.0f, -1.0f, 0.0f));

    EXPECT_NEAR(cp.x, 0.0f, kTol);
    EXPECT_NEAR(cp.y, 0.0f, kTol);
    EXPECT_NEAR(cp.z, 0.0f, kTol);
}

TEST(Triangle, ClosestNormalReturnsStoredNormal) {
    const Triangle tri = reference_triangle();

    // closest_normal ignores the query point and returns the stored normal.
    const Float3 n = tri.closest_normal(Float3(10.0f, -7.0f, 3.0f));

    EXPECT_NEAR(n.x, 0.0f, kTol);
    EXPECT_NEAR(n.y, 0.0f, kTol);
    EXPECT_NEAR(n.z, 1.0f, kTol);
}

TEST(Triangle, SignedDistancePositiveOnFrontSide) {
    const Triangle tri = reference_triangle();

    // A point one unit along +normal projects inside: distance +1.
    const float sd = tri.signed_distance(Float3(0.25f, 0.25f, 1.0f));

    EXPECT_NEAR(sd, 1.0f, kTol);
}

TEST(Triangle, SignedDistanceNegativeOnBackSide) {
    const Triangle tri = reference_triangle();

    // Same in-plane location but behind the plane: distance -1.
    const float sd = tri.signed_distance(Float3(0.25f, 0.25f, -1.0f));

    EXPECT_NEAR(sd, -1.0f, kTol);
}

TEST(Triangle, IsInsideBackHalfSpace) {
    const Triangle tri = reference_triangle();

    // The back half-space (negative side of the plane) is interior.
    EXPECT_TRUE(tri.is_inside(Float3(0.25f, 0.25f, -1.0f)));
}

TEST(Triangle, IsInsideFrontRejectedBeyondTolerance) {
    const Triangle tri = reference_triangle();

    // In front of the plane and farther than the (zero) tolerance: not inside.
    EXPECT_FALSE(tri.is_inside(Float3(0.25f, 0.25f, 1.0f)));

    // Widening the shell past the distance accepts the same point.
    EXPECT_TRUE(tri.is_inside(Float3(0.25f, 0.25f, 1.0f), 2.0f));
}

TEST(Triangle, IsInsideDegenerateAlwaysFalse) {
    // Collinear vertices yield a zero stored normal, so the inside test bails out.
    const Triangle degenerate(Float3(0.0f, 0.0f, 0.0f),
                              Float3(1.0f, 0.0f, 0.0f),
                              Float3(2.0f, 0.0f, 0.0f));

    EXPECT_NEAR(degenerate.normal.length_squared(), 0.0f, kTol);
    EXPECT_FALSE(degenerate.is_inside(Float3(0.0f, 0.0f, -1.0f)));
    EXPECT_FALSE(degenerate.is_inside(Float3(0.0f, 0.0f, -1.0f), 5.0f));
}

TEST(Triangle, IsOnSurfaceWithinTolerance) {
    const Triangle tri = reference_triangle();

    // Exactly on the face.
    EXPECT_TRUE(tri.is_on_surface(Float3(0.25f, 0.25f, 0.0f)));

    // 0.1 above the face: outside a zero shell, inside a 0.2 shell.
    EXPECT_FALSE(tri.is_on_surface(Float3(0.25f, 0.25f, 0.1f)));
    EXPECT_TRUE(tri.is_on_surface(Float3(0.25f, 0.25f, 0.1f), 0.2f));
}

TEST(Triangle, IsOnSurfaceNegativeToleranceFalse) {
    const Triangle tri = reference_triangle();

    EXPECT_FALSE(tri.is_on_surface(Float3(0.25f, 0.25f, 0.0f), -0.1f));
}

TEST(Triangle, TraceHitsThroughInterior) {
    const Triangle tri = reference_triangle();

    const Ray ray(Float3(0.25f, 0.25f, 1.0f), Float3(0.0f, 0.0f, -1.0f));
    const HitSurface hit = tri.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kTol);
    EXPECT_NEAR(hit.point.x, 0.25f, kTol);
    EXPECT_NEAR(hit.point.y, 0.25f, kTol);
    EXPECT_NEAR(hit.point.z, 0.0f, kTol);
    // Reported normal is the stored face normal.
    EXPECT_NEAR(hit.normal.z, 1.0f, kTol);
}

TEST(Triangle, TraceMissPastEdge) {
    const Triangle tri = reference_triangle();

    // Aimed straight down but well outside the triangle's barycentric domain.
    const Ray ray(Float3(2.0f, 2.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f));

    EXPECT_FALSE(tri.trace(ray).is_intersecting);
}

TEST(Triangle, TraceParallelRayMisses) {
    const Triangle tri = reference_triangle();

    // Direction lies in the triangle plane: |det| <= eps, treated as a miss.
    const Ray ray(Float3(0.25f, 0.25f, 1.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_FALSE(tri.trace(ray).is_intersecting);
}

TEST(Triangle, TraceBackFaceHitsWithUnflippedNormal) {
    const Triangle tri = reference_triangle();

    // Ray approaches from behind (-z) travelling toward +z. Möller-Trumbore here
    // does not cull back faces, so it still registers a hit and the reported
    // normal is the stored +z normal, NOT flipped toward the ray.
    const Ray ray(Float3(0.25f, 0.25f, -1.0f), Float3(0.0f, 0.0f, 1.0f));
    const HitSurface hit = tri.trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kTol);
    EXPECT_NEAR(hit.point.z, 0.0f, kTol);
    EXPECT_NEAR(hit.normal.z, 1.0f, kTol);
}

TEST(Triangle, BuilderBuildsValidTriangle) {
    const Triangle tri = Triangle::builder()
                             .with_a(Float3(0.0f, 0.0f, 0.0f))
                             .with_b(Float3(1.0f, 0.0f, 0.0f))
                             .with_c(Float3(0.0f, 1.0f, 0.0f))
                             .build();

    EXPECT_TRUE(tri.is_valid());
    EXPECT_NEAR(tri.normal.z, 1.0f, kTol);
}

TEST(Triangle, BuilderThrowsOnDegenerate) {
    auto builder = Triangle::builder().with_vertices(
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f));

    EXPECT_THROW(builder.build(), std::runtime_error);
}
