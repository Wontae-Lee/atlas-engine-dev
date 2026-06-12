#include "../utilities/test_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/square.h>

#include <testkit/testkit.h>

namespace {

using atlas::Ray;
using atlas::Square;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Square, DefaultConstructorCreatesValidSquare) {
    const Square<float> square;

    EXPECT_EQ(square.type(), GeometryType::Square);
    EXPECT_TRUE(square.is_valid());
    EXPECT_NEAR(square.side_length, 1.0f, tol);
}

TEST(Square, BuilderConstructsConfiguredSquare) {
    const auto square = Square<float>::builder()
                            .with_center(Vector3F(1, 2, 3))
                            .with_normal(Vector3F(0, 0, 2))
                            .with_side_length(4.0f)
                            .build();

    EXPECT_TRUE(vec_near(square.center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(square.normal, Vector3F(0, 0, 2), tol));
    EXPECT_NEAR(square.side_length, 4.0f, tol);
}

TEST(Square, BuilderRejectsInvalidSquare) {
    EXPECT_THROW(
        Square<float>::builder()
            .with_normal(Vector3F(0, 0, 0))
            .with_side_length(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Square, ClosestPointProjectsAndClampsToEdges) {
    const Square<float> square(Vector3F(0, 0, 0), Vector3F(0, 0, 1), 2.0f);

    const Vector3F interior_projection = square.closest_point(Vector3F(0.25f, 0.50f, 3.0f));
    const Vector3F edge_projection = square.closest_point(Vector3F(3.0f, 0.0f, 0.0f));

    EXPECT_TRUE(vec_near(interior_projection, Vector3F(0.25f, 0.50f, 0.0f), tol));
    EXPECT_TRUE(vec_near(edge_projection, Vector3F(1.0f, 0.0f, 0.0f), tol));
}

TEST(Square, ClosestNormalAndSignedDistanceAreConsistent) {
    const Square<float> square(Vector3F(0, 0, 0), Vector3F(0, 0, 2), 2.0f);

    const Vector3F normal = square.closest_normal(Vector3F(0, 0, 5));

    EXPECT_TRUE(vec_near(normal, Vector3F(0, 0, 1), tol));
    EXPECT_NEAR(square.signed_distance(Vector3F(0, 0, 3)), 3.0f, tol);
    EXPECT_NEAR(square.signed_distance(Vector3F(0, 0, -3)), -3.0f, tol);
}

TEST(Square, InsideSurfaceCentroidAndBoundWork) {
    const Square<float> square(Vector3F(1, 2, 3), Vector3F(0, 0, 1), 4.0f);

    EXPECT_TRUE(square.is_inside(Vector3F(1, 2, 3), 0.0f));
    EXPECT_TRUE(square.is_on_surface(Vector3F(3, 4, 3), 0.0f));

    const Vector3F center = square.centroid();
    const auto bounds = square.bound();

    EXPECT_TRUE(vec_near(center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(bounds.lower_corner, Vector3F(-1, 0, 3), tol));
    EXPECT_TRUE(vec_near(bounds.upper_corner, Vector3F(3, 4, 3), tol));
}

TEST(Square, GeometryOperatorAndTraceWork) {
    const Square<float> square(Vector3F(0, 0, 0), Vector3F(0, 0, 1), 2.0f);

    const auto geometry_operator = square.make_device_geometry_view();
    const Ray<float> ray(Vector3F(0, 0, 5), Vector3F(0, 0, -1));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(square.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, tol);
    EXPECT_TRUE(vec_near(hit.point, Vector3F(0, 0, 0), tol));
}
