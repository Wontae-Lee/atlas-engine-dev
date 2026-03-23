#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Circle, ConstructorStoresParameters) {
    const geometry::Circle<double> circle(
        Vector3<double>(1.0, 2.0, 3.0),
        Vector3<double>(0.0, 0.0, 2.0),
        4.0);

    EXPECT_TRUE(test::vec_near(circle.center, Vector3<double>(1.0, 2.0, 3.0), eps));
    EXPECT_TRUE(test::vec_near(circle.normal, Vector3<double>(0.0, 0.0, 2.0), eps));
    EXPECT_NEAR(circle.radius, 4.0, eps);
}

TEST(Circle, MakeGeometryOperatorProducesCircleDispatchType) {
    const geometry::Circle<double> circle;
    const auto op = circle.make_geometry_operator();

    EXPECT_EQ(circle.type(), geometry::GeometryType::Circle);
    EXPECT_EQ(op.type, geometry::GeometryType::Circle);
}

TEST(Circle, ClosestPointMatchesGeometryOperator) {
    const geometry::Circle<double> circle(
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(0.0, 0.0, 1.0),
        2.0);

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&circle.center);
    op.normal = atlas::raw_pointer_cast(&circle.normal);
    op.radius = atlas::raw_pointer_cast(&circle.radius);

    const Vector3<double> p(3.0, 0.0, 1.0);
    EXPECT_TRUE(test::vec_near(circle.closest_point(p), op.closest_point(p), eps));
}

TEST(Circle, BoundMatchesGeometryOperator) {
    const geometry::Circle<double> circle(
        Vector3<double>(1.0, -2.0, 3.0),
        Vector3<double>(0.0, 0.0, 1.0),
        2.0);

    geometry::CircleGeometryOperator<double> op;
    op.center = atlas::raw_pointer_cast(&circle.center);
    op.normal = atlas::raw_pointer_cast(&circle.normal);
    op.radius = atlas::raw_pointer_cast(&circle.radius);

    const auto expected = op.bound();
    const auto got = circle.bound();

    EXPECT_TRUE(test::vec_near(got.lower_corner, expected.lower_corner, eps));
    EXPECT_TRUE(test::vec_near(got.upper_corner, expected.upper_corner, eps));
}

TEST(Circle, BuilderRejectsInvalidParameters) {
    EXPECT_THROW(
        geometry::Circle<double>::builder()
            .with_normal(Vector3<double>(0.0, 0.0, 0.0))
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        geometry::Circle<double>::builder()
            .with_radius(0.0)
            .build(),
        std::runtime_error);
}

TEST(Circle, BuilderBuildsValidCircle) {
    const auto circle = geometry::Circle<double>::builder()
                            .with_center(Vector3<double>(1.0, 2.0, 3.0))
                            .with_normal(Vector3<double>(0.0, 3.0, 0.0))
                            .with_radius(5.0)
                            .build();

    EXPECT_TRUE(circle.is_valid());
    EXPECT_EQ(circle.type(), geometry::GeometryType::Circle);
    EXPECT_TRUE(test::vec_near(circle.center, Vector3<double>(1.0, 2.0, 3.0), eps));
    EXPECT_TRUE(test::vec_near(circle.normal, Vector3<double>(0.0, 3.0, 0.0), eps));
    EXPECT_NEAR(circle.radius, 5.0, eps);
}
