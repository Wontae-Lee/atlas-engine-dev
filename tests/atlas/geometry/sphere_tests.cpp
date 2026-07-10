#include <atlas/geometry/sphere.h>

#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::HitSurface;
using atlas::Ray;
using atlas::Sphere;

constexpr float kTol = 1e-5f;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, kTol);
    EXPECT_NEAR(actual.y, expected.y, kTol);
    EXPECT_NEAR(actual.z, expected.z, kTol);
}

}

TEST(Sphere, DefaultIsUnitSphereAtOrigin) {
    const Sphere sphere;

    expect_vec_near(sphere.center, Float3(0.0f, 0.0f, 0.0f));
    EXPECT_FLOAT_EQ(sphere.radius, 1.0f);
    EXPECT_TRUE(sphere.is_valid());
}

TEST(Sphere, ConstructorStoresCenterAndRadius) {
    const Sphere sphere(Float3(2.0f, 3.0f, 4.0f), 2.0f);

    expect_vec_near(sphere.center, Float3(2.0f, 3.0f, 4.0f));
    EXPECT_FLOAT_EQ(sphere.radius, 2.0f);
}

TEST(Sphere, IsValidRejectsNonPositiveRadius) {
    EXPECT_FALSE(Sphere(Float3(0.0f, 0.0f, 0.0f), 0.0f).is_valid());
    EXPECT_FALSE(Sphere(Float3(0.0f, 0.0f, 0.0f), -1.0f).is_valid());
}

TEST(Sphere, SignedDistanceIsNegativeInside) {
    const Sphere sphere; // unit sphere at origin.

    EXPECT_NEAR(sphere.signed_distance(Float3(2.0f, 0.0f, 0.0f)), 1.0f, kTol);
    EXPECT_NEAR(sphere.signed_distance(Float3(0.5f, 0.0f, 0.0f)), -0.5f, kTol);
    EXPECT_NEAR(sphere.signed_distance(Float3(1.0f, 0.0f, 0.0f)), 0.0f, kTol);
}

TEST(Sphere, ClosestPointLandsOnTheSurface) {
    const Sphere sphere;

    expect_vec_near(sphere.closest_point(Float3(2.0f, 0.0f, 0.0f)),
                    Float3(1.0f, 0.0f, 0.0f));
}

TEST(Sphere, ClosestPointFallsBackAtCenter) {
    const Sphere sphere;

    // Degenerate query at the center returns the deterministic +x rim point.
    expect_vec_near(sphere.closest_point(Float3(0.0f, 0.0f, 0.0f)),
                    Float3(1.0f, 0.0f, 0.0f));
}

TEST(Sphere, ClosestNormalIsOutwardUnitVector) {
    const Sphere sphere;

    const Float3 n = sphere.closest_normal(Float3(0.0f, 3.0f, 0.0f));

    expect_vec_near(n, Float3(0.0f, 1.0f, 0.0f));
    EXPECT_NEAR(n.length(), 1.0f, kTol);
}

TEST(Sphere, IsInsideRespectsRadius) {
    const Sphere sphere;

    EXPECT_TRUE(sphere.is_inside(Float3(0.5f, 0.0f, 0.0f)));
    EXPECT_FALSE(sphere.is_inside(Float3(1.5f, 0.0f, 0.0f)));
}

TEST(Sphere, IsInsideToleranceDilatesTheBall) {
    const Sphere sphere;

    // Radius grows to 1.6, so a point at distance 1.5 becomes interior.
    EXPECT_TRUE(sphere.is_inside(Float3(1.5f, 0.0f, 0.0f), 0.6f));
    // An adjusted radius below zero always rejects.
    EXPECT_FALSE(sphere.is_inside(Float3(0.0f, 0.0f, 0.0f), -2.0f));
}

TEST(Sphere, IsOnSurfaceShellInsideAndOutside) {
    const Sphere sphere;

    // On the shell.
    EXPECT_TRUE(sphere.is_on_surface(Float3(1.0f, 0.0f, 0.0f), 0.1f));
    // Just outside the outer shell radius (1.1).
    EXPECT_FALSE(sphere.is_on_surface(Float3(1.2f, 0.0f, 0.0f), 0.1f));
    // Just inside the inner shell radius (0.9).
    EXPECT_FALSE(sphere.is_on_surface(Float3(0.85f, 0.0f, 0.0f), 0.1f));
}

TEST(Sphere, IsOnSurfaceRejectsNegativeToleranceAndBadRadius) {
    EXPECT_FALSE(Sphere().is_on_surface(Float3(1.0f, 0.0f, 0.0f), -0.1f));
    EXPECT_FALSE(Sphere(Float3(0.0f, 0.0f, 0.0f), 0.0f).is_on_surface(Float3(0.0f, 0.0f, 0.0f), 0.1f));
}

TEST(Sphere, CentroidIsTheCenter) {
    const Sphere sphere(Float3(2.0f, 3.0f, 4.0f), 2.0f);

    expect_vec_near(sphere.centroid(), Float3(2.0f, 3.0f, 4.0f));
}

TEST(Sphere, BoundEnclosesTheSphere) {
    const Sphere sphere(Float3(2.0f, 3.0f, 4.0f), 2.0f);
    const AABB box = sphere.bound();

    expect_vec_near(box.lower_corner, Float3(0.0f, 1.0f, 2.0f));
    expect_vec_near(box.upper_corner, Float3(4.0f, 5.0f, 6.0f));
    EXPECT_TRUE(box.contains(sphere.closest_point(Float3(100.0f, 3.0f, 4.0f))));
}

TEST(Sphere, TraceHitsNearSurfaceFromOutside) {
    const Sphere sphere; // unit sphere at origin.

    const HitSurface hit = sphere.trace(Ray(Float3(0.0f, 0.0f, -5.0f),
                                            Float3(0.0f, 0.0f, 1.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    // Nearest root: enters the near surface at z = -1, four units ahead.
    EXPECT_NEAR(hit.distance, 4.0f, kTol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, -1.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, -1.0f));
}

TEST(Sphere, TraceMissesWhenPointingAway) {
    const Sphere sphere;

    // Both roots lie behind the origin.
    const HitSurface hit = sphere.trace(Ray(Float3(0.0f, 0.0f, -5.0f),
                                            Float3(0.0f, 0.0f, -1.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Sphere, TraceMissesWhenRayAvoidsSphere) {
    const Sphere sphere;

    const HitSurface hit = sphere.trace(Ray(Float3(0.0f, 5.0f, 0.0f),
                                            Float3(1.0f, 0.0f, 0.0f)));

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(Sphere, TraceFromInsideReturnsForwardExit) {
    const Sphere sphere;

    // Origin at the center: the only non-negative root is the exit at t = radius.
    const HitSurface hit = sphere.trace(Ray(Float3(0.0f, 0.0f, 0.0f),
                                            Float3(1.0f, 0.0f, 0.0f)));

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kTol);
    expect_vec_near(hit.point, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Sphere, BuilderBuildsValidatedSphere) {
    const Sphere sphere = Sphere::builder()
                              .with_center(Float3(1.0f, 2.0f, 3.0f))
                              .with_radius(4.0f)
                              .build();

    expect_vec_near(sphere.center, Float3(1.0f, 2.0f, 3.0f));
    EXPECT_FLOAT_EQ(sphere.radius, 4.0f);
}

TEST(Sphere, BuilderRejectsNonPositiveRadius) {
    EXPECT_THROW(
        static_cast<void>(Sphere::builder().with_radius(-1.0f).build()),
        std::runtime_error);
}
