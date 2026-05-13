#include "../utilities/test_utils.h"

#include <atlas/geometry/circle.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Circle;
using atlas::Ray;
using atlas::Vector3F;
using atlas::geometry::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Circle, DefaultConstructorCreatesValidCircle) {
    const Circle<float> circle;

    EXPECT_EQ(circle.type(), GeometryType::Circle);
    EXPECT_TRUE(circle.is_valid());
    EXPECT_NEAR(circle.radius, 1.0f, tol);
}

TEST(Circle, BuilderConstructsConfiguredCircle) {
    const auto circle = Circle<float>::builder()
                            .with_center(Vector3F(1, 2, 3))
                            .with_normal(Vector3F(0, 0, 2))
                            .with_radius(4.0f)
                            .build();

    EXPECT_TRUE(vec_near(circle.center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(circle.normal, Vector3F(0, 0, 2), tol));
    EXPECT_NEAR(circle.radius, 4.0f, tol);
}

TEST(Circle, BuilderRejectsInvalidCircle) {
    EXPECT_THROW(
        Circle<float>::builder()
            .with_normal(Vector3F(0, 0, 0))
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Circle, ClosestPointProjectsToDiskAndRim) {
    const Circle<float> circle(Vector3F(0, 0, 0), Vector3F(0, 0, 1), 2.0f);

    const Vector3F interior_projection = circle.closest_point(Vector3F(1, 0, 3));
    const Vector3F rim_projection = circle.closest_point(Vector3F(3, 0, 0));

    EXPECT_TRUE(vec_near(interior_projection, Vector3F(1, 0, 0), tol));
    EXPECT_TRUE(vec_near(rim_projection, Vector3F(2, 0, 0), tol));
}

TEST(Circle, ClosestNormalAndSignedDistanceAreConsistent) {
    const Circle<float> circle(Vector3F(0, 0, 0), Vector3F(0, 0, 2), 2.0f);

    const Vector3F normal = circle.closest_normal(Vector3F(0, 0, 5));

    EXPECT_TRUE(vec_near(normal, Vector3F(0, 0, 1), tol));
    EXPECT_NEAR(circle.signed_distance(Vector3F(0, 0, 3)), 3.0f, tol);
    EXPECT_NEAR(circle.signed_distance(Vector3F(0, 0, -3)), -3.0f, tol);
}

TEST(Circle, InsideSurfaceCentroidAndBoundWork) {
    const Circle<float> circle(Vector3F(1, 2, 3), Vector3F(0, 0, 1), 2.0f);

    EXPECT_TRUE(circle.is_inside(Vector3F(1, 2, 3), 0.0f));
    EXPECT_TRUE(circle.is_on_surface(Vector3F(3, 2, 3), 0.0f));

    const Vector3F center = circle.centroid();
    const auto bounds = circle.bound();

    EXPECT_TRUE(vec_near(center, Vector3F(1, 2, 3), tol));
    EXPECT_TRUE(vec_near(bounds.lower_corner, Vector3F(-1, 0, 3), tol));
    EXPECT_TRUE(vec_near(bounds.upper_corner, Vector3F(3, 4, 3), tol));
}

TEST(Circle, GeometryOperatorAndTraceWork) {
    const Circle<float> circle(Vector3F(0, 0, 0), Vector3F(0, 0, 1), 2.0f);

    const auto geometry_operator = circle.make_geometry_operator();
    const Ray<float> ray(Vector3F(0, 0, 5), Vector3F(0, 0, -1));
    const auto hit = circle.make_geometry_operator().trace(ray);

    EXPECT_EQ(circle.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, tol);
    EXPECT_TRUE(vec_near(hit.point, Vector3F(0, 0, 0), tol));
}
