#include "../utilities/tests_utils.h"

#include <atlas/spatial/ray.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using Ray = atlas::spatial::Ray<T>;
using Hit = atlas::spatial::SurfaceRayIntersection<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Ray, DefaultConstructorCreatesCanonicalRay) {
    const Ray ray;

    EXPECT_TRUE(atlas::test::vec_near(ray.origin, Vec3(0, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(ray.direction, Vec3(1, 0, 0), kEps));
}

TEST(Ray, ConstructorNormalizesDirection) {
    const Ray ray(Vec3(1, 2, 3), Vec3(0, 3, 4));

    EXPECT_TRUE(atlas::test::vec_near(ray.origin, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(ray.direction, Vec3(0, 0.6f, 0.8f), 1e-4f));
    EXPECT_NEAR(ray.direction.length(), 1.0f, 1e-4f);
}

TEST(Ray, CopyConstructorPreservesOriginAndDirection) {
    const Ray source(Vec3(-1, 4, 2), Vec3(2, 0, 0));
    const Ray copy(source);

    EXPECT_TRUE(atlas::test::vec_near(copy.origin, source.origin, kEps));
    EXPECT_TRUE(atlas::test::vec_near(copy.direction, source.direction, kEps));
}

TEST(Ray, PointAtEvaluatesParametricLine) {
    const Ray ray(Vec3(1, 2, 3), Vec3(0, 0, 2));

    EXPECT_TRUE(atlas::test::vec_near(ray.point_at(0.0f), Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(ray.point_at(2.5f), Vec3(1, 2, 5.5f), 1e-4f));
    EXPECT_TRUE(atlas::test::vec_near(ray.point_at(-1.0f), Vec3(1, 2, 2), 1e-4f));
}

TEST(Ray, HitSurfaceDefaultStateIsWellDefined) {
    const Hit hit;

    EXPECT_FALSE(hit.is_intersecting);
    EXPECT_TRUE(std::isfinite(static_cast<double>(hit.distance)));
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(0, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(hit.normal, Vec3(0, 0, 1), kEps));
}
