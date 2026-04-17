#include "../utilities/tests_utils.h"

#include <atlas/logging/logging.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/triangle.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

atlas::geometry::Triangle<T>
make_triangle() {
    return atlas::geometry::Triangle<T>(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0));
}

} // namespace

TEST(Triangle, DefaultAndBuilderCreateValidTriangle) {
    const atlas::geometry::Triangle<T> default_triangle;
    const auto built_triangle = atlas::geometry::Triangle<T>::builder()
                                    .with_vertices(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0))
                                    .build();

    EXPECT_EQ(default_triangle.type(), atlas::geometry::GeometryType::Triangle);
    EXPECT_TRUE(default_triangle.is_valid());
    EXPECT_TRUE(built_triangle.is_valid());
}

TEST(Triangle, BuilderRejectsDegenerateTriangle) {
    EXPECT_THROW(
        atlas::geometry::Triangle<T>::builder()
            .with_vertices(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(2, 0, 0))
            .build(),
        std::runtime_error);
}

TEST(Triangle, ClosestPointNormalAndDistanceWork) {
    const auto triangle = make_triangle();

    const Vec3 closest = triangle.closest_point(Vec3(0.2f, 0.2f, 1.0f));
    const Vec3 normal = triangle.closest_normal(Vec3(0.2f, 0.2f, 1.0f));

    EXPECT_TRUE(atlas::test::vec_near(closest, Vec3(0.2f, 0.2f, 0.0f), kEps));
    EXPECT_TRUE(atlas::test::vec_near(normal, Vec3(0, 0, 1), kEps));
    EXPECT_NEAR(std::abs(triangle.signed_distance(Vec3(0.2f, 0.2f, 1.0f))), 1.0f, kEps);
}

TEST(Triangle, ClassificationCentroidBoundAndOperatorWork) {
    const auto triangle = make_triangle();

    EXPECT_TRUE(triangle.is_on_surface(Vec3(0.2f, 0.2f, 0.0f), kEps));

    const Vec3 center = triangle.centroid();
    const auto bounds = triangle.bound();
    const auto geometry_operator = triangle.make_geometry_operator();

    EXPECT_TRUE(atlas::test::vec_near(center, Vec3(1.0f / 3.0f, 1.0f / 3.0f, 0.0f), 1e-4f));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(triangle.type(), geometry_operator.type);
}

TEST(Triangle, TraceHitsFrontFace) {
    const auto triangle = make_triangle();

    const auto hit = triangle.make_geometry_operator().trace(atlas::spatial::Ray<T>(Vec3(0.2f, 0.2f, 1.0f), Vec3(0, 0, -1)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kEps);
    EXPECT_TRUE(atlas::test::vec_near(hit.point, Vec3(0.2f, 0.2f, 0.0f), kEps));
}
