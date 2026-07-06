#include <atlas/geometry/box.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Box;
using atlas::GeometryType;
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

TEST(Box, DefaultConstructorCreatesCanonicalBox) {
    const Box box;

    expect_vec_near(box.lower_corner, Float3(-1.0f, -1.0f, -1.0f));
    expect_vec_near(box.upper_corner, Float3(1.0f, 1.0f, 1.0f));
    const atlas::Geometry geometry_operator(box);
    EXPECT_EQ(geometry_operator.type, GeometryType::box);
    EXPECT_TRUE(box.is_valid());
}

TEST(Box, BuilderConstructsConfiguredBox) {
    const auto box = Box::builder()
                         .with_lower_corner(Float3(-2.0f, -3.0f, -4.0f))
                         .with_upper_corner(Float3(2.0f, 3.0f, 4.0f))
                         .build();

    expect_vec_near(box.lower_corner, Float3(-2.0f, -3.0f, -4.0f));
    expect_vec_near(box.upper_corner, Float3(2.0f, 3.0f, 4.0f));
}

TEST(Box, BuilderRejectsInvalidBounds) {
    EXPECT_THROW(
        (void)Box::builder()
            .with_lower_corner(Float3(1.0f, 0.0f, 0.0f))
            .with_upper_corner(Float3(0.0f, 1.0f, 1.0f))
            .build(),
        std::runtime_error);
}

TEST(Box, ClosestPointProjectsOutsidePointToSurface) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    const Float3 closest = box.closest_point(Float3(3.0f, 0.25f, -0.5f));

    expect_vec_near(closest, Float3(1.0f, 0.25f, -0.5f));
}

TEST(Box, ClosestPointProjectsInsidePointToNearestFace) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    const Float3 closest = box.closest_point(Float3(0.2f, 0.7f, -0.1f));

    expect_vec_near(closest, Float3(0.2f, 1.0f, -0.1f));
}

TEST(Box, ClosestNormalReturnsExpectedDirections) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    const Float3 outside_normal = box.closest_normal(Float3(4.0f, 0.0f, 0.0f));
    const Float3 inside_normal  = box.closest_normal(Float3(0.1f, -0.8f, 0.0f));

    expect_vec_near(outside_normal, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(inside_normal, Float3(0.0f, -1.0f, 0.0f));
}

TEST(Box, SignedDistanceMatchesInsideOutsideCases) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    EXPECT_NEAR(box.signed_distance(Float3(0.0f, 0.0f, 0.0f)), -1.0f, tol);
    EXPECT_NEAR(box.signed_distance(Float3(1.0f, 0.0f, 0.0f)), 0.0f, tol);
    EXPECT_NEAR(box.signed_distance(Float3(3.0f, 0.0f, 0.0f)), 2.0f, tol);
}

TEST(Box, InsideAndSurfaceQueriesRespectTolerance) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    EXPECT_TRUE(box.is_inside(Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(box.is_inside(Float3(1.1f, 0.0f, 0.0f), 0.0f));
    EXPECT_TRUE(box.is_inside(Float3(1.1f, 0.0f, 0.0f), 0.11f));

    EXPECT_TRUE(box.is_on_surface(Float3(1.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(box.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.01f));
    EXPECT_TRUE(box.is_on_surface(Float3(1.05f, 0.0f, 0.0f), 0.051f));
}

TEST(Box, CentroidAndBoundMatchCorners) {
    const Box box(Float3(-2.0f, -4.0f, -6.0f), Float3(4.0f, 2.0f, 8.0f));

    const Float3 center = box.centroid();
    const auto bounds    = box.bound();

    expect_vec_near(center, Float3(1.0f, -1.0f, 1.0f));
    expect_vec_near(bounds.lower_corner, Float3(-2.0f, -4.0f, -6.0f));
    expect_vec_near(bounds.upper_corner, Float3(4.0f, 2.0f, 8.0f));
}

TEST(Box, GeometryOperatorAndRayTraceWork) {
    const Box box(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f));

    const atlas::Geometry geometry_operator(box);
    const Ray ray(Float3(-3.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(GeometryType::box, geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 2.0f, tol);
    expect_vec_near(hit.point, Float3(-1.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(-1.0f, 0.0f, 0.0f));
}
