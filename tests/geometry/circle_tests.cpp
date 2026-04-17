#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/circle.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Circle, DefaultConstructorCreatesValidCircle) {
    const atlas::geometry::Circle<T> circle;

    EXPECT_EQ(circle.type(), atlas::geometry::GeometryType::Circle);
    EXPECT_TRUE(circle.is_valid());
    EXPECT_NEAR(circle.radius, 1.0f, kEps);
}

TEST(Circle, BuilderConstructsConfiguredCircle) {
    const auto circle = atlas::geometry::Circle<T>::builder()
                            .with_center(Vec3(1, 2, 3))
                            .with_normal(Vec3(0, 0, 2))
                            .with_radius(4.0f)
                            .build();

    EXPECT_TRUE(atlas::test::vec_near(circle.center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(circle.normal, Vec3(0, 0, 2), kEps));
    EXPECT_NEAR(circle.radius, 4.0f, kEps);
}

TEST(Circle, BuilderRejectsInvalidCircle) {
    EXPECT_THROW(
        atlas::geometry::Circle<T>::builder()
            .with_normal(Vec3(0, 0, 0))
            .with_radius(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Circle, ClosestPointProjectsToDiskAndRim) {
    const atlas::geometry::Circle<T> circle(Vec3(0, 0, 0), Vec3(0, 0, 1), 2.0f);

    const Vec3 interior_projection = circle.closest_point(Vec3(1, 0, 3));
    const Vec3 rim_projection = circle.closest_point(Vec3(3, 0, 0));

    EXPECT_TRUE(atlas::test::vec_near(interior_projection, Vec3(1, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(rim_projection, Vec3(2, 0, 0), kEps));
}

TEST(Circle, ClosestNormalAndSignedDistanceAreConsistent) {
    const atlas::geometry::Circle<T> circle(Vec3(0, 0, 0), Vec3(0, 0, 2), 2.0f);

    const Vec3 normal = circle.closest_normal(Vec3(0, 0, 5));

    EXPECT_TRUE(atlas::test::vec_near(normal, Vec3(0, 0, 1), kEps));
    EXPECT_NEAR(circle.signed_distance(Vec3(0, 0, 3)), 3.0f, kEps);
    EXPECT_NEAR(circle.signed_distance(Vec3(0, 0, -3)), -3.0f, kEps);
}

TEST(Circle, InsideSurfaceCentroidAndBoundWork) {
    const atlas::geometry::Circle<T> circle(Vec3(1, 2, 3), Vec3(0, 0, 1), 2.0f);

    EXPECT_TRUE(circle.is_inside(Vec3(1, 2, 3), 0.0f));
    EXPECT_TRUE(circle.is_on_surface(Vec3(3, 2, 3), 0.0f));

    const Vec3 center = circle.centroid();
    const auto bounds = circle.bound();

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.lower_corner, Vec3(-1, 0, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.upper_corner, Vec3(3, 4, 3), kEps));
}

TEST(Circle, GeometryOperatorAndTraceWork) {
    const atlas::geometry::Circle<T> circle(Vec3(0, 0, 0), Vec3(0, 0, 1), 2.0f);

    const auto geometry_operator = circle.make_geometry_operator();
    const atlas::spatial::Ray<T> ray(Vec3(0, 0, 5), Vec3(0, 0, -1));
    const auto hit = circle.make_geometry_operator().trace(ray);

    EXPECT_EQ(circle.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, kEps);
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(0, 0, 0), kEps));
}
