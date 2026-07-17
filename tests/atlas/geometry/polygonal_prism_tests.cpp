#include <atlas/geometry/polygonal_prism.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::Float3;
using atlas::PolygonalPrism;
using atlas::Ray;

void expect_vec_near(const Float3& actual, const Float3& expected, float tolerance = 1.0e-5f) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

}

TEST(PolygonalPrism, DefaultIsAUnitTriangularPrism) {
    const PolygonalPrism prism;
    EXPECT_EQ(prism.side_count, 3);
    EXPECT_FLOAT_EQ(prism.radius, 1.0f);
    EXPECT_FLOAT_EQ(prism.height, 1.0f);
    EXPECT_TRUE(prism.is_valid());
}

TEST(PolygonalPrism, RejectsFewerThanThreeSidesAndNonPositiveExtents) {
    EXPECT_FALSE(PolygonalPrism(Float3(0.0f), 2, 1.0f, 1.0f).is_valid());
    EXPECT_FALSE(PolygonalPrism(Float3(0.0f), 3, 0.0f, 1.0f).is_valid());
    EXPECT_FALSE(PolygonalPrism(Float3(0.0f), 3, 1.0f, 0.0f).is_valid());
}

TEST(PolygonalPrism, SquareCrossSectionUsesCircumradiusAndFixedOrientation) {
    const PolygonalPrism prism(Float3(0.0f), 4, 1.0f, 2.0f);
    const float apothem = 0.70710678118f;

    EXPECT_NEAR(prism.signed_distance(Float3(0.0f)), -apothem, 1.0e-5f);
    EXPECT_NEAR(prism.signed_distance(Float3(0.5f, 0.5f, 0.0f)), 0.0f, 1.0e-5f);
    expect_vec_near(prism.closest_point(Float3(2.0f, 0.0f, 0.0f)), Float3(1.0f, 0.0f, 0.0f));
}

TEST(PolygonalPrism, ClosestPointSelectsSideOrCap) {
    const PolygonalPrism prism(Float3(0.0f), 4, 1.0f, 2.0f);
    const float inset = 0.01f / 1.41421356237f;

    expect_vec_near(prism.closest_point(Float3(0.0f, 0.0f, 0.9f)), Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(prism.closest_point(Float3(0.5f - inset, 0.5f - inset, 0.0f)),
                    Float3(0.5f, 0.5f, 0.0f));
}

TEST(PolygonalPrism, ClassificationUsesSignedTolerance) {
    const PolygonalPrism prism(Float3(0.0f), 6, 1.0f, 2.0f);
    EXPECT_TRUE(prism.is_inside(Float3(0.0f)));
    EXPECT_FALSE(prism.is_inside(Float3(0.0f, 0.0f, 1.05f)));
    EXPECT_TRUE(prism.is_inside(Float3(0.0f, 0.0f, 1.05f), 0.1f));
    EXPECT_TRUE(prism.is_on_surface(Float3(0.0f, 0.0f, 1.0f)));
    EXPECT_FALSE(prism.is_on_surface(Float3(0.0f), 0.1f));
}

TEST(PolygonalPrism, IsInsideOffsetsEveryFaceWithoutRoundingCorners) {
    const PolygonalPrism prism(Float3(0.0f), 4, 1.0f, 2.0f);

    EXPECT_FALSE(prism.is_inside(Float3(1.05f, 0.0f, 0.0f)));
    EXPECT_TRUE(prism.is_inside(Float3(1.05f, 0.0f, 0.0f), 0.04f));
    EXPECT_FALSE(prism.is_inside(Float3(0.0f), -0.8f));
}

TEST(PolygonalPrism, IsOnSurfaceChecksSidesAndCapsIndependently) {
    const PolygonalPrism prism(Float3(0.0f), 4, 1.0f, 2.0f);

    EXPECT_TRUE(prism.is_on_surface(Float3(0.5f, 0.5f, 0.0f), 1.0e-5f));
    EXPECT_TRUE(prism.is_on_surface(Float3(0.0f, 0.0f, 1.0f), 1.0e-5f));
    EXPECT_FALSE(prism.is_on_surface(Float3(0.0f), 0.1f));
    EXPECT_FALSE(prism.is_on_surface(Float3(0.5f, 0.5f, 0.0f), -0.1f));
}

TEST(PolygonalPrism, ClassificationRejectsInvalidPrisms) {
    const PolygonalPrism prism(Float3(0.0f), 2, 1.0f, 1.0f);
    EXPECT_FALSE(prism.is_inside(Float3(0.0f)));
    EXPECT_FALSE(prism.is_on_surface(Float3(0.0f)));
}

TEST(PolygonalPrism, BoundUsesCircumradiusAndHalfHeight) {
    const PolygonalPrism prism(Float3(1.0f, 2.0f, 3.0f), 5, 2.0f, 4.0f);
    const auto bound = prism.bound();
    expect_vec_near(bound.lower_corner, Float3(-1.0f, 0.0f, 1.0f));
    expect_vec_near(bound.upper_corner, Float3(3.0f, 4.0f, 5.0f));
}

TEST(PolygonalPrism, TraceHitsSideAndCapAndExitsFromInside) {
    const PolygonalPrism prism(Float3(0.0f), 4, 1.0f, 2.0f);

    const auto side = prism.trace(Ray(Float3(-2.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f)));
    ASSERT_TRUE(side.is_intersecting);
    EXPECT_NEAR(side.distance, 1.0f, 1.0e-5f);
    expect_vec_near(side.point, Float3(-1.0f, 0.0f, 0.0f));

    const auto cap = prism.trace(Ray(Float3(0.0f, 0.0f, 3.0f), Float3(0.0f, 0.0f, -1.0f)));
    ASSERT_TRUE(cap.is_intersecting);
    EXPECT_NEAR(cap.distance, 2.0f, 1.0e-5f);
    expect_vec_near(cap.normal, Float3(0.0f, 0.0f, 1.0f));

    const auto exit = prism.trace(Ray(Float3(0.0f), Float3(0.0f, 0.0f, 1.0f)));
    ASSERT_TRUE(exit.is_intersecting);
    EXPECT_NEAR(exit.distance, 1.0f, 1.0e-5f);
}

TEST(PolygonalPrism, TraceMissesOutsideTheHeightBand) {
    const PolygonalPrism prism(Float3(0.0f), 5, 1.0f, 2.0f);
    EXPECT_FALSE(prism.trace(Ray(Float3(-2.0f, 0.0f, 2.0f), Float3(1.0f, 0.0f, 0.0f))).is_intersecting);
}

TEST(PolygonalPrism, BuilderBuildsAndRejectsInvalidParameters) {
    const PolygonalPrism prism = PolygonalPrism::builder()
        .with_center(Float3(1.0f, 2.0f, 3.0f))
        .with_side_count(8)
        .with_radius(2.0f)
        .with_height(4.0f)
        .build();
    EXPECT_EQ(prism.side_count, 8);
    EXPECT_TRUE(prism.is_valid());
    EXPECT_THROW(static_cast<void>(PolygonalPrism::builder().with_side_count(2).build()),
                 std::runtime_error);
}
