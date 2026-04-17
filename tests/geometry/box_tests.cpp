#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/box.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Box, DefaultConstructorCreatesCanonicalBox) {
    const atlas::geometry::Box<T> box;

    EXPECT_TRUE(atlas::test::vec_near(box.lower_corner, Vec3(-1, -1, -1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.upper_corner, Vec3(1, 1, 1), kEps));
    EXPECT_EQ(box.type(), atlas::geometry::GeometryType::Box);
    EXPECT_TRUE(box.is_valid());
}

TEST(Box, BuilderConstructsConfiguredBox) {
    const auto box = atlas::geometry::Box<T>::builder()
                         .with_lower_corner(Vec3(-2, -3, -4))
                         .with_upper_corner(Vec3(2, 3, 4))
                         .build();

    EXPECT_TRUE(atlas::test::vec_near(box.lower_corner, Vec3(-2, -3, -4), kEps));
    EXPECT_TRUE(atlas::test::vec_near(box.upper_corner, Vec3(2, 3, 4), kEps));
}

TEST(Box, BuilderRejectsInvalidBounds) {
    EXPECT_THROW(
        atlas::geometry::Box<T>::builder()
            .with_lower_corner(Vec3(1, 0, 0))
            .with_upper_corner(Vec3(0, 1, 1))
            .build(),
        std::runtime_error);
}

TEST(Box, ClosestPointProjectsOutsidePointToSurface) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    const Vec3 closest = box.closest_point(Vec3(3, 0.25f, -0.5f));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(1, 0.25f, -0.5f), kEps));
}

TEST(Box, ClosestPointProjectsInsidePointToNearestFace) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    const Vec3 closest = box.closest_point(Vec3(0.2f, 0.7f, -0.1f));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(0.2f, 1.0f, -0.1f), kEps));
}

TEST(Box, ClosestNormalReturnsExpectedDirections) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    const Vec3 outside_normal = box.closest_normal(Vec3(4, 0, 0));
    const Vec3 inside_normal = box.closest_normal(Vec3(0.1f, -0.8f, 0.0f));

    EXPECT_TRUE(atlas::test::vec_near(outside_normal, Vec3(1, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(inside_normal, Vec3(0, -1, 0), kEps));
}

TEST(Box, SignedDistanceMatchesInsideOutsideCases) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    EXPECT_NEAR(box.signed_distance(Vec3(0, 0, 0)), -1.0f, kEps);
    EXPECT_NEAR(box.signed_distance(Vec3(1, 0, 0)), 0.0f, kEps);
    EXPECT_NEAR(box.signed_distance(Vec3(3, 0, 0)), 2.0f, kEps);
}

TEST(Box, InsideAndSurfaceQueriesRespectTolerance) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    EXPECT_TRUE(box.is_inside(Vec3(0, 0, 0), 0.0f));
    EXPECT_FALSE(box.is_inside(Vec3(1.1f, 0, 0), 0.0f));
    EXPECT_TRUE(box.is_inside(Vec3(1.1f, 0, 0), 0.11f));

    EXPECT_TRUE(box.is_on_surface(Vec3(1, 0, 0), 0.0f));
    EXPECT_FALSE(box.is_on_surface(Vec3(1.05f, 0, 0), 0.01f));
    EXPECT_TRUE(box.is_on_surface(Vec3(1.05f, 0, 0), 0.051f));
}

TEST(Box, CentroidAndBoundMatchCorners) {
    const atlas::geometry::Box<T> box(Vec3(-2, -4, -6), Vec3(4, 2, 8));

    const Vec3 center = box.centroid();
    const auto bounds = box.bound();

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1, -1, 1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.lower_corner, Vec3(-2, -4, -6), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.upper_corner, Vec3(4, 2, 8), kEps));
}

TEST(Box, GeometryOperatorAndRayTraceWork) {
    const atlas::geometry::Box<T> box(Vec3(-1, -1, -1), Vec3(1, 1, 1));

    const auto geometry_operator = box.make_geometry_operator();
    const atlas::spatial::Ray<T> ray(Vec3(-3, 0, 0), Vec3(1, 0, 0));
    const auto hit = box.make_geometry_operator().trace(ray);

    EXPECT_EQ(box.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 2.0f, kEps);
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(-1, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(hit.normal, Vec3(-1, 0, 0), kEps));
}
