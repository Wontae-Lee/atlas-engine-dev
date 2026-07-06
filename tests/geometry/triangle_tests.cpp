#include <atlas/geometry/triangle.h>

#include <atlas/geometry/geometry.h>

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Geometry;
using atlas::GeometryType;
using atlas::Ray;
using atlas::Triangle;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

Triangle
make_triangle() {
    return Triangle(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f));
}

}

TEST(Triangle, DefaultAndBuilderCreateValidTriangle) {
    const Triangle default_triangle;
    const auto built_triangle = Triangle::builder()
                                    .with_vertices(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f))
                                    .build();

    EXPECT_FALSE(default_triangle.is_valid());
    EXPECT_TRUE(built_triangle.is_valid());
}

TEST(Triangle, BuilderRejectsDegenerateTriangle) {
    EXPECT_THROW(
        (void)Triangle::builder()
            .with_vertices(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), Vector3(2.0f, 0.0f, 0.0f))
            .build(),
        std::runtime_error);
}

TEST(Triangle, ClosestPointNormalAndDistanceWork) {
    const auto triangle = make_triangle();

    const Vector3 closest = triangle.closest_point(Vector3(0.2f, 0.2f, 1.0f));
    const Vector3 normal  = triangle.closest_normal(Vector3(0.2f, 0.2f, 1.0f));

    expect_vec_near(closest, Vector3(0.2f, 0.2f, 0.0f));
    expect_vec_near(normal, Vector3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(std::abs(triangle.signed_distance(Vector3(0.2f, 0.2f, 1.0f))), 1.0f, tol);
}

TEST(Triangle, ClassificationCentroidBoundAndOperatorWork) {
    const auto triangle = make_triangle();

    EXPECT_TRUE(triangle.is_on_surface(Vector3(0.2f, 0.2f, 0.0f), tol));

    const Vector3 center         = triangle.centroid();
    const auto bounds            = triangle.bound();
    const Geometry geometry_operator(triangle);

    expect_vec_near(center, Vector3(1.0f / 3.0f, 1.0f / 3.0f, 0.0f));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(geometry_operator.type, GeometryType::triangle);
}

TEST(Triangle, TraceHitsFrontFace) {
    const auto triangle = make_triangle();

    const auto hit = Geometry(triangle).trace(Ray(Vector3(0.2f, 0.2f, 1.0f), Vector3(0.0f, 0.0f, -1.0f)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, tol);
    expect_vec_near(hit.point, Vector3(0.2f, 0.2f, 0.0f));
}
