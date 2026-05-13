#include "../utilities/test_utils.h"

#include <atlas/spatial/ray.h>

#include <testkit/testkit.h>

namespace {

using atlas::HitSurface;
using atlas::RayF;
using atlas::Vector3F;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Ray, DefaultConstructorCreatesCanonicalRay) {
    // Arrange: create a default ray.
    const RayF ray;

    // Assert: default origin and direction use the canonical +X ray.
    EXPECT_TRUE(vec_near(ray.origin, Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(ray.direction, Vector3F(1, 0, 0), tol));
}

TEST(Ray, ConstructorNormalizesDirection) {
    // Arrange and act: build a ray with a non-unit direction.
    const RayF ray(Vector3F(1, 2, 3), Vector3F(0, 3, 4));

    // Assert: construction preserves origin and normalizes direction.
    EXPECT_TRUE(vec_near(ray.origin, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(ray.direction, Vector3F(0, 0.6f, 0.8f), tol));
    EXPECT_NEAR(ray.direction.length(), 1.0f, tol);
}

TEST(Ray, CopyConstructorPreservesOriginAndDirection) {
    // Arrange: build a ray with non-default origin and direction.
    const RayF source(Vector3F(-1, 4, 2), Vector3F(2, 0, 0));

    // Act: copy-construct a second ray.
    const RayF copy(source);

    // Assert: copy construction preserves the full ray state.
    EXPECT_TRUE(vec_near(copy.origin, source.origin, tol));
    EXPECT_TRUE(vec_near(copy.direction, source.direction, tol));
}

TEST(Ray, PointAtEvaluatesParametricLine) {
    // Arrange: build a ray whose direction normalizes to +Z.
    const RayF ray(Vector3F(1, 2, 3), Vector3F(0, 0, 2));

    // Assert: point_at evaluates origin + direction * t.
    EXPECT_TRUE(vec_near(ray.point_at(0.0f), Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(ray.point_at(2.5f), Vector3F(1, 2, 5.5f), tol));
    EXPECT_TRUE(vec_near(ray.point_at(-1.0f), Vector3F(1, 2, 2), tol));
}

TEST(Ray, HitSurfaceDefaultStateIsWellDefined) {
    // Arrange: create a default surface hit record.
    const HitSurface<float> hit;

    // Assert: default hit state is finite and non-intersecting.
    EXPECT_FALSE(hit.is_intersecting);
    EXPECT_TRUE(std::isfinite(static_cast<double>(hit.distance)));
    EXPECT_TRUE(vec_near(hit.point, Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(hit.normal, Vector3F(0, 0, 1), tol));
}
