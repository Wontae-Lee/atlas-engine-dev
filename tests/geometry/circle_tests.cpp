#include <atlas/geometry/circle.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Circle;
using atlas::GeometryType;
using atlas::Ray;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
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
                            .with_center(Vector3(1.0f, 2.0f, 3.0f))
                            .with_normal(Vector3(0.0f, 0.0f, 2.0f))
                            .with_radius(4.0f)
                            .build();

    expect_vec_near(circle.center, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(circle.normal, Vector3(0.0f, 0.0f, 2.0f));
    EXPECT_NEAR(circle.radius, 4.0f, tol);
}

TEST(Circle, BuilderRejectsInvalidCircle) {
    EXPECT_THROW(
        (void)Circle::builder()
            .with_normal(Vector3(0.0f, 0.0f, 0.0f))
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Circle, ClosestPointProjectsToDiskAndRim) {
    const Circle circle(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f), 2.0f);

    const Vector3 interior_projection = circle.closest_point(Vector3(1.0f, 0.0f, 3.0f));
    const Vector3 rim_projection      = circle.closest_point(Vector3(3.0f, 0.0f, 0.0f));

    expect_vec_near(interior_projection, Vector3(1.0f, 0.0f, 0.0f));
    expect_vec_near(rim_projection, Vector3(2.0f, 0.0f, 0.0f));
}

TEST(Circle, ClosestNormalAndSignedDistanceAreConsistent) {
    const Circle circle(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 2.0f), 2.0f);

    const Vector3 normal = circle.closest_normal(Vector3(0.0f, 0.0f, 5.0f));

    expect_vec_near(normal, Vector3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(circle.signed_distance(Vector3(0.0f, 0.0f, 3.0f)), 3.0f, tol);
    EXPECT_NEAR(circle.signed_distance(Vector3(0.0f, 0.0f, -3.0f)), -3.0f, tol);
}

TEST(Circle, InsideSurfaceCentroidAndBoundWork) {
    const Circle circle(Vector3(1.0f, 2.0f, 3.0f), Vector3(0.0f, 0.0f, 1.0f), 2.0f);

    EXPECT_TRUE(circle.is_inside(Vector3(1.0f, 2.0f, 3.0f), 0.0f));
    EXPECT_TRUE(circle.is_on_surface(Vector3(3.0f, 2.0f, 3.0f), 0.0f));

    const Vector3 center = circle.centroid();
    const auto bounds    = circle.bound();

    expect_vec_near(center, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(bounds.lower_corner, Vector3(-1.0f, 0.0f, 3.0f));
    expect_vec_near(bounds.upper_corner, Vector3(3.0f, 4.0f, 3.0f));
}

TEST(Circle, GeometryOperatorAndTraceWork) {
    const Circle circle(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f), 2.0f);

    const atlas::Geometry geometry_operator(circle);
    const Ray ray(Vector3(0.0f, 0.0f, 5.0f), Vector3(0.0f, 0.0f, -1.0f));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(GeometryType::circle, geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, tol);
    expect_vec_near(hit.point, Vector3(0.0f, 0.0f, 0.0f));
}
