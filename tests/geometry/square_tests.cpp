#include "../utilities/tests_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/square.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Square, DefaultConstructorCreatesValidSquare) {
    const atlas::geometry::Square<T> square;

    EXPECT_EQ(square.type(), atlas::geometry::GeometryType::Square);
    EXPECT_TRUE(square.is_valid());
    EXPECT_NEAR(square.side_length, 1.0f, kEps);
}

TEST(Square, BuilderConstructsConfiguredSquare) {
    const auto square = atlas::geometry::Square<T>::builder()
                            .with_center(Vec3(1, 2, 3))
                            .with_normal(Vec3(0, 0, 2))
                            .with_side_length(4.0f)
                            .build();

    EXPECT_TRUE(atlas::test::vec_near(square.center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(square.normal, Vec3(0, 0, 2), kEps));
    EXPECT_NEAR(square.side_length, 4.0f, kEps);
}

TEST(Square, BuilderRejectsInvalidSquare) {
    EXPECT_THROW(
        atlas::geometry::Square<T>::builder()
            .with_normal(Vec3(0, 0, 0))
            .with_side_length(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(Square, ClosestPointProjectsAndClampsToEdges) {
    const atlas::geometry::Square<T> square(Vec3(0, 0, 0), Vec3(0, 0, 1), 2.0f);

    const Vec3 interior_projection = square.closest_point(Vec3(0.25f, 0.50f, 3.0f));
    const Vec3 edge_projection = square.closest_point(Vec3(3.0f, 0.0f, 0.0f));

    EXPECT_TRUE(atlas::test::vec_near(interior_projection, Vec3(0.25f, 0.50f, 0.0f), kEps));
    EXPECT_TRUE(atlas::test::vec_near(edge_projection, Vec3(1.0f, 0.0f, 0.0f), kEps));
}

TEST(Square, ClosestNormalAndSignedDistanceAreConsistent) {
    const atlas::geometry::Square<T> square(Vec3(0, 0, 0), Vec3(0, 0, 2), 2.0f);

    const Vec3 normal = square.closest_normal(Vec3(0, 0, 5));

    EXPECT_TRUE(atlas::test::vec_near(normal, Vec3(0, 0, 1), kEps));
    EXPECT_NEAR(square.signed_distance(Vec3(0, 0, 3)), 3.0f, kEps);
    EXPECT_NEAR(square.signed_distance(Vec3(0, 0, -3)), -3.0f, kEps);
}

TEST(Square, InsideSurfaceCentroidAndBoundWork) {
    const atlas::geometry::Square<T> square(Vec3(1, 2, 3), Vec3(0, 0, 1), 4.0f);

    EXPECT_TRUE(square.is_inside(Vec3(1, 2, 3), 0.0f));
    EXPECT_TRUE(square.is_on_surface(Vec3(3, 4, 3), 0.0f));

    const Vec3 center = square.centroid();
    const auto bounds = square.bound();

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1, 2, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.lower_corner, Vec3(-1, 0, 3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(bounds.upper_corner, Vec3(3, 4, 3), kEps));
}

TEST(Square, GeometryOperatorAndTraceWork) {
    const atlas::geometry::Square<T> square(Vec3(0, 0, 0), Vec3(0, 0, 1), 2.0f);

    const auto geometry_operator = square.make_geometry_operator();
    const atlas::spatial::Ray<T> ray(Vec3(0, 0, 5), Vec3(0, 0, -1));
    const auto hit = geometry_operator.trace(ray);

    EXPECT_EQ(square.type(), geometry_operator.type);
    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 5.0f, kEps);
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(0, 0, 0), kEps));
}
