#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::Ray;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Box, DefaultConstructorCreatesCanonicalBox) {
    const Box<float> box;

    EXPECT_TRUE(vec_near(box.lower_corner, Vector3F(-1, -1, -1), tol));
    EXPECT_TRUE(vec_near(box.upper_corner, Vector3F(1, 1, 1), tol));
    EXPECT_EQ(box.type(), GeometryType::Box);
    EXPECT_TRUE(box.is_valid());
}

TEST(Box, BuilderConstructsConfiguredBox) {
    const auto box = Box<float>::builder()
                         .with_lower_corner(Vector3F(-2, -3, -4))
                         .with_upper_corner(Vector3F(2, 3, 4))
                         .build();

    EXPECT_TRUE(vec_near(box.lower_corner, Vector3F(-2, -3, -4), tol));
    EXPECT_TRUE(vec_near(box.upper_corner, Vector3F(2, 3, 4), tol));
}

TEST(Box, BuilderRejectsInvalidBounds) {
    EXPECT_THROW(
        Box<float>::builder()
            .with_lower_corner(Vector3F(1, 0, 0))
            .with_upper_corner(Vector3F(0, 1, 1))
            .build(),
        std::runtime_error);
}

TEST(Box, ClosestPointProjectsOutsidePointToSurface) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    const Vector3F closest = box.closest_point(Vector3F(3, 0.25f, -0.5f));

    EXPECT_TRUE(vec_near(closest, Vector3F(1, 0.25f, -0.5f), tol));
}

TEST(Box, ClosestPointProjectsInsidePointToNearestFace) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    const Vector3F closest = box.closest_point(Vector3F(0.2f, 0.7f, -0.1f));

    EXPECT_TRUE(vec_near(closest, Vector3F(0.2f, 1.0f, -0.1f), tol));
}

TEST(Box, ClosestNormalReturnsExpectedDirections) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    const Vector3F outside_normal = box.closest_normal(Vector3F(4, 0, 0));
    const Vector3F inside_normal = box.closest_normal(Vector3F(0.1f, -0.8f, 0.0f));

    EXPECT_TRUE(vec_near(outside_normal, Vector3F(1, 0, 0), tol));
    EXPECT_TRUE(vec_near(inside_normal, Vector3F(0, -1, 0), tol));
}

TEST(Box, SignedDistanceMatchesInsideOutsideCases) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    EXPECT_NEAR(box.signed_distance(Vector3F(0, 0, 0)), -1.0f, tol);
    EXPECT_NEAR(box.signed_distance(Vector3F(1, 0, 0)), 0.0f, tol);
    EXPECT_NEAR(box.signed_distance(Vector3F(3, 0, 0)), 2.0f, tol);
}

TEST(Box, InsideAndSurfaceQueriesRespectTolerance) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    EXPECT_TRUE(box.is_inside(Vector3F(0, 0, 0), 0.0f));
    EXPECT_FALSE(box.is_inside(Vector3F(1.1f, 0, 0), 0.0f));
    EXPECT_TRUE(box.is_inside(Vector3F(1.1f, 0, 0), 0.11f));

    EXPECT_TRUE(box.is_on_surface(Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(box.is_on_surface(Vector3F(1.05f, 0, 0), 0.01f));
    EXPECT_TRUE(box.is_on_surface(Vector3F(1.05f, 0, 0), 0.051f));
}

TEST(Box, CentroidAndBoundMatchCorners) {
    const Box<float> box(Vector3F(-2, -4, -6), Vector3F(4, 2, 8));

    const Vector3F center = box.centroid();
    const auto bounds = box.bound();

    EXPECT_TRUE(vec_near(center, Vector3F(1, -1, 1), tol));
    EXPECT_TRUE(vec_near(bounds.lower_corner, Vector3F(-2, -4, -6), tol));
    EXPECT_TRUE(vec_near(bounds.upper_corner, Vector3F(4, 2, 8), tol));
}

TEST(Box, GeometryOperatorAndRayTraceWork) {
    const Box<float> box(Vector3F(-1, -1, -1), Vector3F(1, 1, 1));

    const auto geometry_operator = box.make_device_geometry_view();
    const Ray<float> ray(Vector3F(-3, 0, 0), Vector3F(1, 0, 0));
    const auto hit = box.make_device_geometry_view().trace(ray);

    EXPECT_EQ(box.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 2.0f, tol);
    EXPECT_TRUE(vec_near(hit.point, Vector3F(-1, 0, 0), tol));
    EXPECT_TRUE(vec_near(hit.normal, Vector3F(-1, 0, 0), tol));
}
