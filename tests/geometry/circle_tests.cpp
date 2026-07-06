#include <atlas/geometry/circle.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Circle;
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

TEST(Circle, DefaultConstructorCreatesValidCircle) {
    const Circle circle;

    const atlas::Geometry geometry_operator(circle);
    EXPECT_EQ(geometry_operator.type, GeometryType::circle);
    EXPECT_TRUE(circle.is_valid());
    EXPECT_NEAR(circle.radius, 1.0f, tol);
}

TEST(Circle, BuilderConstructsConfiguredCircle) {
    const auto circle = Circle::builder()
                            .with_center(Float3(1.0f, 2.0f, 3.0f))
                            .with_normal(Float3(0.0f, 0.0f, 2.0f))
                            .with_radius(4.0f)
                            .build();

    expect_vec_near(circle.center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(circle.normal, Float3(0.0f, 0.0f, 2.0f));
    EXPECT_NEAR(circle.radius, 4.0f, tol);
}

TEST(Circle, BuilderRejectsInvalidCircle) {
    EXPECT_THROW(
        (void)Circle::builder()
            .with_normal(Float3(0.0f, 0.0f, 0.0f))
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Circle, ClosestPointProjectsToDiskAndRim) {
    const Circle circle(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);

    const Float3 interior_projection = circle.closest_point(Float3(1.0f, 0.0f, 3.0f));
    const Float3 rim_projection      = circle.closest_point(Float3(3.0f, 0.0f, 0.0f));

    expect_vec_near(interior_projection, Float3(1.0f, 0.0f, 0.0f));
    expect_vec_near(rim_projection, Float3(2.0f, 0.0f, 0.0f));
}

TEST(Circle, ClosestNormalAndSignedDistanceAreConsistent) {
    const Circle circle(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 2.0f), 2.0f);

    const Float3 normal = circle.closest_normal(Float3(0.0f, 0.0f, 5.0f));

    expect_vec_near(normal, Float3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(circle.signed_distance(Float3(0.0f, 0.0f, 3.0f)), 3.0f, tol);
    EXPECT_NEAR(circle.signed_distance(Float3(0.0f, 0.0f, -3.0f)), -3.0f, tol);
}

TEST(Circle, InsideSurfaceCentroidAndBoundWork) {
    const Circle circle(Float3(1.0f, 2.0f, 3.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);

    EXPECT_TRUE(circle.is_inside(Float3(1.0f, 2.0f, 3.0f), 0.0f));
    EXPECT_TRUE(circle.is_on_surface(Float3(3.0f, 2.0f, 3.0f), 0.0f));

    const Float3 center = circle.centroid();
    const auto bounds    = circle.bound();

    expect_vec_near(center, Float3(1.0f, 2.0f, 3.0f));
    expect_vec_near(bounds.lower_corner, Float3(-1.0f, 0.0f, 3.0f));
    expect_vec_near(bounds.upper_corner, Float3(3.0f, 4.0f, 3.0f));
}

TEST(Circle, GeometryOperatorAndTraceWork) {
    const Circle circle(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);

    const atlas::Geometry geometry_operator(circle);
    const Ray ray(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, -1.0f));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(GeometryType::circle, geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, tol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
}
