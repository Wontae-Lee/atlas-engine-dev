#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::HitAABB;
using atlas::Ray;

void
expect_float3_eq(const Float3& actual, const Float3& expected) {
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
}

// The box every ray/AABB case below is fired against.
AABB
unit_cube() {
    return AABB(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f));
}

}

TEST(Ray, DefaultConstructsAtOriginAlongPlusX) {
    const Ray ray;

    expect_float3_eq(ray.origin, Float3(0.0f, 0.0f, 0.0f));
    expect_float3_eq(ray.direction, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Ray, ConstructorNormalizesTheDirection) {
    const Ray ray(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 0.0f, 5.0f));

    expect_float3_eq(ray.origin, Float3(1.0f, 2.0f, 3.0f));
    expect_float3_eq(ray.direction, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FLOAT_EQ(ray.direction.length(), 1.0f);
}

TEST(Ray, ZeroDirectionCollapsesToTheDegenerateZeroVector) {
    const Ray ray(Float3(1.0f, 1.0f, 1.0f), Float3(0.0f, 0.0f, 0.0f));

    expect_float3_eq(ray.direction, Float3(0.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(ray.direction.length_squared(), 0.0f);
}

TEST(Ray, PointAtEvaluatesOriginPlusTDirection) {
    const Ray ray(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 1.0f, 0.0f));

    expect_float3_eq(ray.point_at(0.0f), Float3(1.0f, 2.0f, 3.0f));
    expect_float3_eq(ray.point_at(4.0f), Float3(1.0f, 6.0f, 3.0f));
    // Negative t walks backward along the direction.
    expect_float3_eq(ray.point_at(-2.0f), Float3(1.0f, 0.0f, 3.0f));
}

TEST(RayPlaneDistance, ReportsForwardHitDistance) {
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, -1.0f));
    float     distance = -1.0f;

    EXPECT_TRUE(atlas::ray_plane_distance(Float3(0.0f, 0.0f, 0.0f),
                                          Float3(0.0f, 0.0f, 1.0f),
                                          ray,
                                          distance));
    EXPECT_FLOAT_EQ(distance, 5.0f);
}

TEST(RayPlaneDistance, ParallelRayMisses) {
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(1.0f, 0.0f, 0.0f));
    float     distance = 42.0f;

    EXPECT_FALSE(atlas::ray_plane_distance(Float3(0.0f, 0.0f, 0.0f),
                                           Float3(0.0f, 0.0f, 1.0f),
                                           ray,
                                           distance));
    // A miss leaves the out-parameter untouched.
    EXPECT_FLOAT_EQ(distance, 42.0f);
}

TEST(RayPlaneDistance, IntersectionBehindOriginIsRejected) {
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, 1.0f));
    float     distance = 0.0f;

    EXPECT_FALSE(atlas::ray_plane_distance(Float3(0.0f, 0.0f, 0.0f),
                                           Float3(0.0f, 0.0f, 1.0f),
                                           ray,
                                           distance));
}

TEST(RayAabb, HitReportsTheEnterExitInterval) {
    const HitAABB hit = unit_cube().trace(Ray(Float3(-5.0f, 1.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_FLOAT_EQ(hit.enter, 5.0f);
    EXPECT_FLOAT_EQ(hit.exit, 7.0f);
}

TEST(RayAabb, IntersectsWrapperAgreesWithTrace) {
    const AABB box = unit_cube();
    const Ray  ray(Float3(-5.0f, 1.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f));

    EXPECT_TRUE(box.intersects(ray));
    EXPECT_EQ(box.intersects(ray), box.trace(ray).is_intersecting);
}

TEST(RayAabb, MissWhenTheRayPassesBesideTheBox) {
    // Offset in y beyond the box; the x-slab overlaps but y never does.
    const HitAABB hit = unit_cube().trace(Ray(Float3(-5.0f, 5.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(RayAabb, MissWhenTheRayPointsAwayFromTheBox) {
    // Origin beside the box, marching away: the hit interval collapses.
    const HitAABB hit = unit_cube().trace(Ray(Float3(-5.0f, 1.0f, 1.0f), Float3(-1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(RayAabb, ParallelToSlabHitsWhenOriginLiesBetweenThePlanes) {
    // Direction has zero y/z components: those slabs are parallel. The origin's
    // y and z sit inside the box, so those axes impose no constraint and the ray
    // still enters through the x face.
    const HitAABB hit = unit_cube().trace(Ray(Float3(-5.0f, 1.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_TRUE(hit.is_intersecting);
}

TEST(RayAabb, ParallelToSlabMissesWhenOriginIsOutsideThePlanes) {
    // Same parallel x-ray, but the origin's y is above the box: the parallel
    // y-slab rejects immediately.
    const HitAABB hit = unit_cube().trace(Ray(Float3(-5.0f, 3.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(RayAabb, OriginInsideTheBoxClampsEntryToZero) {
    const HitAABB hit = unit_cube().trace(Ray(Float3(1.0f, 1.0f, 1.0f), Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_FLOAT_EQ(hit.enter, 0.0f);
    EXPECT_FLOAT_EQ(hit.exit, 1.0f);
}
