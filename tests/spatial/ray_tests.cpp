#include <atlas/spatial/ray.h>

#include <cmath>
#include <gtest/gtest.h>

namespace {

using atlas::HitSurface;
using atlas::Ray;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Ray, DefaultConstructorCreatesCanonicalRay) {
    const Ray ray;

    expect_vec_near(ray.origin, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(ray.direction, Float3(1.0f, 0.0f, 0.0f));
}

TEST(Ray, ConstructorNormalizesDirection) {
    const Ray ray(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 3.0f, 4.0f));

    expect_vec_near(ray.origin, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(ray.direction, Float3(0.0f, 0.6f, 0.8f));
    EXPECT_NEAR(ray.direction.length(), 1.0f, tol);
}

TEST(Ray, CopyConstructorPreservesOriginAndDirection) {
    const Ray source(Float3(-1.0f, 4.0f, 2.0f), Float3(2.0f, 0.0f, 0.0f));

    const Ray copy(source);

    expect_vec_near(copy.origin, source.origin);
    expect_vec_near(copy.direction, source.direction);
}

TEST(Ray, PointAtEvaluatesParametricLine) {
    const Ray ray(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 0.0f, 2.0f));

    expect_vec_near(ray.point_at(0.0f), Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(ray.point_at(2.5f), Float3(1.0f, 2.0f, 5.5f));
    expect_vec_near(ray.point_at(-1.0f), Float3(1.0f, 2.0f, 2.0f));
}

TEST(Ray, HitSurfaceDefaultStateIsWellDefined) {
    const HitSurface hit;

    EXPECT_FALSE(hit.is_intersecting);
    EXPECT_TRUE(std::isfinite(hit.distance));
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}
