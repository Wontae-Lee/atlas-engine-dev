#include "../utilities/test_utils.h"

#include <atlas/geometry/geometry_operator.h>
#include <atlas/geometry/triangle.h>
#include <atlas/logging/logging.h>

#include <testkit/testkit.h>

namespace {

using atlas::Ray;
using atlas::Triangle;
using atlas::Vector3F;
using atlas::GeometryType;
using atlas::test::vec_near;
using atlas::tol;

Triangle<float>
make_triangle() {
    return Triangle<float>(Vector3F(0, 0, 0), Vector3F(1, 0, 0), Vector3F(0, 1, 0));
}

} // namespace

TEST(Triangle, DefaultAndBuilderCreateValidTriangle) {
    const Triangle<float> default_triangle;
    const auto built_triangle = Triangle<float>::builder()
                                    .with_vertices(Vector3F(0, 0, 0), Vector3F(1, 0, 0), Vector3F(0, 1, 0))
                                    .build();

    EXPECT_EQ(default_triangle.type(), GeometryType::Triangle);
    EXPECT_FALSE(default_triangle.is_valid());
    EXPECT_TRUE(built_triangle.is_valid());
}

TEST(Triangle, BuilderRejectsDegenerateTriangle) {
    EXPECT_THROW(
        Triangle<float>::builder()
            .with_vertices(Vector3F(0, 0, 0), Vector3F(1, 0, 0), Vector3F(2, 0, 0))
            .build(),
        std::runtime_error);
}

TEST(Triangle, ClosestPointNormalAndDistanceWork) {
    const auto triangle = make_triangle();

    const Vector3F closest = triangle.closest_point(Vector3F(0.2f, 0.2f, 1.0f));
    const Vector3F normal = triangle.closest_normal(Vector3F(0.2f, 0.2f, 1.0f));

    EXPECT_TRUE(vec_near(closest, Vector3F(0.2f, 0.2f, 0.0f), tol));
    EXPECT_TRUE(vec_near(normal, Vector3F(0, 0, 1), tol));
    EXPECT_NEAR(std::abs(triangle.signed_distance(Vector3F(0.2f, 0.2f, 1.0f))), 1.0f, tol);
}

TEST(Triangle, ClassificationCentroidBoundAndOperatorWork) {
    const auto triangle = make_triangle();

    EXPECT_TRUE(triangle.is_on_surface(Vector3F(0.2f, 0.2f, 0.0f), tol));

    const Vector3F center = triangle.centroid();
    const auto bounds = triangle.bound();
    const auto geometry_operator = triangle.make_device_geometry_view();

    EXPECT_TRUE(vec_near(center, Vector3F(1.0f / 3.0f, 1.0f / 3.0f, 0.0f), tol));
    EXPECT_TRUE(bounds.is_valid());
    EXPECT_EQ(triangle.type(), geometry_operator.type);
}

TEST(Triangle, TraceHitsFrontFace) {
    const auto triangle = make_triangle();

    const auto hit = triangle.make_device_geometry_view().trace(Ray<float>(Vector3F(0.2f, 0.2f, 1.0f), Vector3F(0, 0, -1)));

    EXPECT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, tol);
    EXPECT_TRUE(vec_near(hit.point, Vector3F(0.2f, 0.2f, 0.0f), tol));
}
