#include <atlas/geometry/square.h>

#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

using atlas::Geometry;
using atlas::GeometryType;
using atlas::Ray;
using atlas::Square;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(Square, DefaultConstructorCreatesValidSquare) {
    const Square square;

    EXPECT_TRUE(square.is_valid());
    EXPECT_NEAR(square.side_length, 1.0f, tol);
}

TEST(Square, BuilderConstructsConfiguredSquare) {
    const auto square = Square::builder()
                            .with_center(Vector3(1.0f, 2.0f, 3.0f))
                            .with_normal(Vector3(0.0f, 0.0f, 2.0f))
                            .with_side_length(4.0f)
                            .build();

    expect_vec_near(square.center, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(square.normal, Vector3(0.0f, 0.0f, 2.0f));
    EXPECT_NEAR(square.side_length, 4.0f, tol);
}

TEST(Square, BuilderRejectsInvalidSquare) {
    EXPECT_THROW(
        (void)Square::builder()
            .with_normal(Vector3(0.0f, 0.0f, 0.0f))
            .with_side_length(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Square, ClosestPointProjectsAndClampsToEdges) {
    const Square square(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f), 2.0f);

    const Vector3 interior_projection = square.closest_point(Vector3(0.25f, 0.50f, 3.0f));
    const Vector3 edge_projection     = square.closest_point(Vector3(3.0f, 0.0f, 0.0f));

    expect_vec_near(interior_projection, Vector3(0.25f, 0.50f, 0.0f));
    expect_vec_near(edge_projection, Vector3(1.0f, 0.0f, 0.0f));
}

TEST(Square, ClosestNormalAndSignedDistanceAreConsistent) {
    const Square square(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 2.0f), 2.0f);

    const Vector3 normal = square.closest_normal(Vector3(0.0f, 0.0f, 5.0f));

    expect_vec_near(normal, Vector3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(square.signed_distance(Vector3(0.0f, 0.0f, 3.0f)), 3.0f, tol);
    EXPECT_NEAR(square.signed_distance(Vector3(0.0f, 0.0f, -3.0f)), -3.0f, tol);
}

TEST(Square, InsideSurfaceCentroidAndBoundWork) {
    const Square square(Vector3(1.0f, 2.0f, 3.0f), Vector3(0.0f, 0.0f, 1.0f), 4.0f);

    EXPECT_TRUE(square.is_inside(Vector3(1.0f, 2.0f, 3.0f), 0.0f));
    EXPECT_TRUE(square.is_on_surface(Vector3(3.0f, 4.0f, 3.0f), 0.0f));

    const Vector3 center = square.centroid();
    const auto bounds    = square.bound();

    expect_vec_near(center, Vector3(1.0f, 2.0f, 3.0f));
    expect_vec_near(bounds.lower_corner, Vector3(-1.0f, 0.0f, 3.0f));
    expect_vec_near(bounds.upper_corner, Vector3(3.0f, 4.0f, 3.0f));
}

TEST(Square, GeometryOperatorAndTraceWork) {
    const Square square(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f), 2.0f);

    const Geometry geometry_operator(square);
    const Ray ray(Vector3(0.0f, 0.0f, 5.0f), Vector3(0.0f, 0.0f, -1.0f));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(geometry_operator.type, GeometryType::square);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, tol);
    expect_vec_near(hit.point, Vector3(0.0f, 0.0f, 0.0f));
}
